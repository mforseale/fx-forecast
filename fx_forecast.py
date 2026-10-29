"""Reproducible one-observation-ahead USD/RUB forecasting study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import warnings
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# This project renders matplotlib figures and does not use Prophet's optional Plotly API.
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)
from prophet import Prophet
from cmdstanpy.utils.logging import get_logger as get_cmdstan_logger
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller


ROOT = Path(__file__).resolve().parent
START, END = "2018-01-01", "2025-12-31"
WINDOW, VALIDATION_SIZE, TEST_SIZE, VALIDATION_STEP = 750, 60, 120, 5
ARIMA_CANDIDATES = [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
PROPHET_CANDIDATES = [
    {"changepoint_prior_scale": 0.01, "yearly_seasonality": False},
    {"changepoint_prior_scale": 0.05, "yearly_seasonality": True},
]


def parse_cbr_xml(payload: bytes) -> pd.DataFrame:
    """Normalize the quoted nominal to RUB per one unit, preserving real jumps."""
    root = ET.fromstring(payload)
    if root.tag != "ValCurs" or root.attrib.get("ID") != "R01235":
        raise ValueError("Expected the CBR USD series R01235")
    records = []
    for item in root.findall("Record"):
        nominal_text, value_text = item.findtext("Nominal"), item.findtext("Value")
        if nominal_text is None or value_text is None:
            raise ValueError("Missing nominal or value")
        nominal = float(nominal_text.replace(" ", "").replace("\xa0", "").replace(",", "."))
        value = float(value_text.replace(" ", "").replace("\xa0", "").replace(",", "."))
        if not np.isfinite(nominal) or not np.isfinite(value) or nominal <= 0 or value <= 0:
            raise ValueError("Rates and nominal must be finite and positive")
        records.append({"date": datetime.strptime(item.attrib["Date"], "%d.%m.%Y"), "rate": value / nominal})
    if not records:
        raise ValueError("The source contains no observations")
    frame = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if frame.date.duplicated().any():
        raise ValueError("Duplicate dates: investigate the source instead of silently dropping rows")
    return frame


def load_data(root: Path, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    source_file = raw / "usd_rub_2018_2025.xml"
    metadata_file = raw / "source.json"
    query = urlencode({"date_req1": "01/01/2018", "date_req2": "31/12/2025", "VAL_NM_RQ": "R01235"})
    url = "https://www.cbr.ru/scripts/XML_dynamic.asp?" + query
    if refresh or not source_file.exists():
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "FXForecastPortfolio/1.0"})
                with urlopen(request, timeout=45) as response:
                    payload = response.read()
                parsed = parse_cbr_xml(payload)
                if not parsed.date.between(START, END).all():
                    raise ValueError("Source dates are outside the requested range")
                source_file.write_bytes(payload)
                metadata = {"url": url, "currency": "USD/RUB", "source": "Bank of Russia",
                            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                            "sha256": hashlib.sha256(payload).hexdigest()}
                metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                break
            except (OSError, ET.ParseError) as exc:
                if attempt == 2:
                    raise RuntimeError("Could not download the official CBR series") from exc
                time.sleep(2 ** attempt)
    payload = source_file.read_bytes()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("The cached source checksum changed")
    frame = parse_cbr_xml(payload)
    if not frame.date.between(START, END).all():
        raise ValueError("Cached dates are outside the requested range")
    output = root / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "usd_rub.csv", index=False)
    return frame, metadata


def metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Expected nonempty equally sized arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite observations and forecasts")
    denominator = np.abs(actual).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined for an all-zero series")
    errors = actual - predicted
    return {"wape_pct": float(100 * np.abs(errors).sum() / denominator),
            "mae_rub": float(np.abs(errors).mean()),
            "rmse_rub": float(np.sqrt(np.square(errors).mean()))}


def split_positions(n: int) -> tuple[int, int]:
    validation_start, test_start = n - VALIDATION_SIZE - TEST_SIZE, n - TEST_SIZE
    if validation_start < WINDOW:
        raise ValueError(f"Need at least {WINDOW + VALIDATION_SIZE + TEST_SIZE} observations")
    return validation_start, test_start


def rolling_predictions(frame: pd.DataFrame, positions, predictor, window: int = WINDOW) -> pd.DataFrame:
    rows = []
    for position in positions:
        if position < window or position >= len(frame):
            raise ValueError("Forecast position is outside the valid history window")
        history = frame.iloc[position - window:position].copy()
        target = frame.iloc[position]
        if history.date.max() >= target.date:
            raise ValueError("Forecast history must strictly precede the target date")
        predicted, fit_warnings = predictor(history, target.date)
        if not np.isfinite(predicted) or predicted <= 0:
            raise ValueError("Forecast must be finite and positive")
        rows.append({"date": target.date, "history_end": history.date.iloc[-1],
                     "actual": float(target.rate), "prediction": float(predicted),
                     "fit_warnings": fit_warnings})
    return pd.DataFrame(rows)


def arima_predictor(order: tuple[int, int, int]):
    def predict(history, target_date):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ARIMA(history.rate.to_numpy(), order=order)
            fitted = model.fit(method_kwargs={"maxiter": 200})
            if not fitted.mle_retvals.get("converged", True):
                fitted = model.fit(method_kwargs={"method": "powell", "maxiter": 200})
            if not fitted.mle_retvals.get("converged", True):
                raise RuntimeError(f"ARIMA{order} did not converge before {target_date}")
        warning_text = " | ".join(sorted({str(w.message) for w in caught}))
        return float(fitted.forecast(1)[0]), warning_text
    return predict


def prophet_predictor(config: dict):
    def predict(history, target_date):
        model = Prophet(weekly_seasonality=False, daily_seasonality=False,
                        uncertainty_samples=0, **config)
        model.fit(history.rename(columns={"date": "ds", "rate": "y"}), seed=42)
        predicted = model.predict(pd.DataFrame({"ds": [target_date]}))
        return float(predicted.yhat.iloc[0]), ""
    return predict


def baseline_predictor(history, target_date):
    return float(history.rate.iloc[-1]), ""


def select_models(frame: pd.DataFrame, validation_start: int, test_start: int, results: Path) -> tuple[dict, pd.DataFrame]:
    positions = range(validation_start, test_start, VALIDATION_STEP)
    candidates = [("ARIMA", str(order), arima_predictor(order), {"order": list(order)})
                  for order in ARIMA_CANDIDATES]
    candidates += [("Prophet", json.dumps(config, sort_keys=True), prophet_predictor(config), config)
                   for config in PROPHET_CANDIDATES]
    rows, selections = [], {}
    baseline = rolling_predictions(frame, positions, baseline_predictor)
    rows.append({"model": "Naive", "config": "last value", **metrics(baseline.actual, baseline.prediction)})
    for family, label, predictor, config in candidates:
        print(f"Validation: {family} {label}", flush=True)
        predictions = rolling_predictions(frame, positions, predictor)
        score = metrics(predictions.actual, predictions.prediction)
        rows.append({"model": family, "config": label, **score})
        if family not in selections or score["wape_pct"] < selections[family]["validation_wape_pct"]:
            selections[family] = {"config": config, "validation_wape_pct": score["wape_pct"]}
    validation = pd.DataFrame(rows).sort_values("wape_pct").reset_index(drop=True)
    validation.to_csv(results / "validation_metrics.csv", index=False)
    (results / "selected_models.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    return selections, validation


def evaluate_test(frame: pd.DataFrame, test_start: int, selected: dict, results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = {"Naive": baseline_predictor,
                  "ARIMA": arima_predictor(tuple(selected["ARIMA"]["config"]["order"])),
                  "Prophet": prophet_predictor(selected["Prophet"]["config"])}
    rows, forecasts = [], []
    for name, predictor in predictors.items():
        print(f"Test: {name}, {len(frame) - test_start} rolling forecasts", flush=True)
        predictions = rolling_predictions(frame, range(test_start, len(frame)), predictor)
        predictions["model"] = name
        forecasts.append(predictions)
        rows.append({"model": name, **metrics(predictions.actual, predictions.prediction)})
    scores = pd.DataFrame(rows)
    baseline_mae = float(scores.loc[scores.model.eq("Naive"), "mae_rub"].iloc[0])
    scores["improvement_vs_naive_pct"] = 100 * (1 - scores.mae_rub / baseline_mae)
    scores = scores.sort_values("wape_pct").reset_index(drop=True)
    forecasts = pd.concat(forecasts, ignore_index=True)
    scores.to_csv(results / "test_metrics.csv", index=False)
    forecasts.to_csv(results / "test_predictions.csv", index=False)
    return scores, forecasts


def exploratory_analysis(train: pd.DataFrame, results: Path) -> dict:
    differences = train.rate.diff().dropna()
    diagnostic = {"training_observations": len(train), "training_end": str(train.date.iloc[-1].date())}
    for label, series in [("level", train.rate), ("first_difference", differences)]:
        result = adfuller(series, autolag="AIC", result_object=False)
        diagnostic[f"adf_{label}"] = {"statistic": float(result[0]), "pvalue": float(result[1]), "lags": int(result[2])}
    diagnostic["jumps_above_3std_kept"] = int((np.abs(differences - differences.mean()) > 3 * differences.std()).sum())
    (results / "diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(train.date, train.rate, color="#245ac8", lw=1)
    axes[0].set(title="USD/RUB: training data only", ylabel="RUB per USD")
    axes[1].plot(train.date.iloc[1:], differences, color="#cf6b27", lw=0.7)
    axes[1].set(ylabel="First difference, RUB", xlabel="Effective date in the source")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results / "history.png", dpi=160)
    plt.close(fig)
    monthly = train.set_index("date").rate.resample("MS").mean().iloc[:-1]
    decomposition = STL(monthly, period=12, robust=True).fit()
    fig = decomposition.plot()
    fig.set_size_inches(11, 8)
    fig.suptitle("Exploratory STL: monthly mean, period = 12 (training only)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(results / "decomposition.png", dpi=160)
    plt.close(fig)
    from statsmodels.graphics.tsaplots import plot_acf
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    plot_acf(train.rate, lags=30, ax=axes[0], title="ACF: level (training)")
    plot_acf(differences, lags=30, ax=axes[1], title="ACF: first difference (training)")
    fig.tight_layout()
    fig.savefig(results / "autocorrelation.png", dpi=160)
    plt.close(fig)
    return diagnostic


def plot_predictions(predictions: pd.DataFrame, results: Path):
    colors = {"Naive": "#6b7280", "ARIMA": "#d16a26", "Prophet": "#7651bc"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    actual = predictions[predictions.model.eq("Naive")]
    axes[0].plot(actual.date, actual.actual, color="#172a46", lw=2, label="Actual")
    for model, group in predictions.groupby("model", sort=False):
        axes[0].plot(group.date, group.prediction, color=colors[model], lw=1.1, alpha=0.85, label=model)
        error = np.abs(group.actual - group.prediction).cumsum()
        axes[1].plot(group.date, error, color=colors[model], label=model)
    axes[0].set(title="One-observation-ahead forecasts: untouched test period", ylabel="RUB per USD")
    axes[1].set(title="Cumulative absolute error: lower is better", ylabel="Cumulative error, RUB", xlabel="Effective date in the source")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results / "backtest.png", dpi=170)
    plt.close(fig)


def markdown_table(scores: pd.DataFrame) -> str:
    lines = ["| Модель | WAPE, % | MAE, ₽ | RMSE, ₽ | Улучшение MAE к baseline, % |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in scores.itertuples():
        lines.append(f"| {row.model} | {row.wape_pct:.3f} | {row.mae_rub:.3f} | {row.rmse_rub:.3f} | {row.improvement_vs_naive_pct:+.2f} |")
    return "\n".join(lines)


def write_report(frame, validation_start, test_start, scores, selected, diagnostics, metadata, results):
    winner = scores.iloc[0]
    conclusion = ("На этом test-периоде сложные модели не превзошли последнее известное значение. Для этого горизонта выбираем простой baseline."
                  if winner.model == "Naive" else
                  f"На этом test-периоде минимальная WAPE у {winner.model}; улучшение MAE к baseline — {winner.improvement_vs_naive_pct:.2f}%. Это точечное сравнение, а не доказательство устойчивого преимущества.")
    text = f"""# Можно ли предсказать следующее значение USD/RUB лучше последнего известного курса?

## Результат

{conclusion}

{markdown_table(scores)}

WAPE = 100 × Σ|факт − прогноз| / Σ|факт|. MAE выражена в рублях за один доллар. Положительное улучшение означает меньшую ошибку, отрицательное — ухудшение. Все модели оценены на одних и тех же {TEST_SIZE} наблюдениях. Малая WAPE уровня курса сама по себе не означает, что модель умеет предсказывать изменения курса.

## Данные

- Источник: [XML API Банка России](https://www.cbr.ru/development/sxml/), USD, код R01235.
- Фактический интервал: {frame.date.iloc[0].date()} — {frame.date.iloc[-1].date()}, {len(frame)} наблюдений.
- Обучение до validation: по {frame.date.iloc[validation_start - 1].date()} включительно.
- Validation: {frame.date.iloc[validation_start].date()} — {frame.date.iloc[test_start - 1].date()}, 60 наблюдений; оценка каждого пятого (12 точек).
- Test: {frame.date.iloc[test_start].date()} — {frame.date.iloc[-1].date()}, 120 последовательных прогнозов.
- SHA-256 исходного XML: `{metadata['sha256']}`.

Номинал нормализован к одному доллару. Проверены типы, уникальность дат, положительность и конечность значений. Выходные не дополняются повторениями курса, реальные скачки не удаляются.

## Метод

Для каждой точки используется окно из предыдущих {WINDOW} наблюдений. Горизонт — одно следующее наблюдение. После появления фактического значения окно сдвигается; параметры моделей переоцениваются. Будущие значения не попадают в обучение. Даты являются датами действия официального курса из источника; время его публикации в XML отсутствует. Это исследование последовательности исторических наблюдений, а не воспроизведение доступного в реальном времени обмена валют.

Сетка ARIMA: {ARIMA_CANDIDATES}. Две настройки Prophet: {json.dumps(PROPHET_CANDIDATES)}. Победитель каждой семьи выбран по WAPE на validation и сохранён до расчёта test. Все преобразования ряда для анализа рассчитаны только по исходной обучающей части.

Выбраны: ARIMA{tuple(selected['ARIMA']['config']['order'])}; Prophet `{json.dumps(selected['Prophet']['config'])}`. При несходимости ARIMA повторяет оптимизацию Powell на том же обучающем окне; дальнейшая несходимость завершает расчёт ошибкой. Предупреждения сохраняются вместе с прогнозами.

## Анализ ряда

ADF на уровне курса: p = {diagnostics['adf_level']['pvalue']:.4g}; на первых разностях: p = {diagnostics['adf_first_difference']['pvalue']:.4g}. Нулевая гипотеза ADF — единичный корень. Результат теста зависит от спецификации и структурных сдвигов, поэтому не является доказательством будущей точности модели. Ряд первых разностей построен отдельно; фактические курсы сохранены.

В обучающей части отмечено {diagnostics['jumps_above_3std_kept']} изменений за пределами среднего ± 3 стандартных отклонения; они оставлены в данных. Это диагностическая отметка, а не правило очистки.

STL применяется к среднему курсу по месяцам обучающей части с периодом 12; последний неполный месяц исключён. Она разделяет ряд на тренд, условную сезонную компоненту и остаток, но сама по себе не подтверждает устойчивую сезонность. ACF дополнительно показывает автокорреляцию уровня и разностей.

![Обучающий ряд и разности](history.png)

![Декомпозиция обучающей части](decomposition.png)

![Автокорреляция](autocorrelation.png)

![Прогнозы и накопленная ошибка](backtest.png)

## Вывод для продуктовой задачи

Этот кейс проверяет, стоит ли усложнять прогнозный компонент валютного сервиса. Решение зависит от выигрыша к простому правилу и стоимости сопровождения; качество на одном периоде не гарантирует такого же результата в другом режиме рынка. Для функции уведомлений или планирования бюджета понадобятся другие горизонты, проверка на нескольких периодах и оценка пользы для пользователя.

Внешние регрессоры не используются, поэтому причины движения курса и влияние новостей/ставки не установлены. Прогнозируется официальный справочный курс, а не курс покупки/продажи банка. Интервалы неопределённости и статистическая значимость различий ошибок в эту версию не входят. Результат не переносится автоматически на другие валюты или горизонты.

## Формулировка для анкеты

«Разработал воспроизводимый проект прогнозирования официального USD/RUB на открытых данных Банка России: проверил качество данных, исследовал стационарность и декомпозицию ряда, сравнил ARIMA и Prophet с наивным прогнозом. Выбрал параметры на отдельном validation-периоде и оценил модели на {TEST_SIZE} последовательных test-наблюдениях без утечки будущих значений. Лучшая WAPE на test — {winner.wape_pct:.3f}% ({winner.model}); описал ограничения и целесообразность усложнения модели».
"""
    (results / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download the fixed historical snapshot again")
    args = parser.parse_args()
    get_cmdstan_logger().setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    np.random.seed(42)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    frame, metadata = load_data(ROOT, refresh=args.refresh)
    print(f"Loaded {len(frame)} observations: {frame.date.iloc[0].date()} .. {frame.date.iloc[-1].date()}", flush=True)
    validation_start, test_start = split_positions(len(frame))
    diagnostics = exploratory_analysis(frame.iloc[:validation_start], results)
    selected, _ = select_models(frame, validation_start, test_start, results)
    scores, predictions = evaluate_test(frame, test_start, selected, results)
    plot_predictions(predictions, results)
    write_report(frame, validation_start, test_start, scores, selected, diagnostics, metadata, results)
    manifest = {"window": WINDOW, "horizon_observations": 1, "validation_step": VALIDATION_STEP,
                "validation_size": VALIDATION_SIZE, "test_size": TEST_SIZE,
                "source_sha256": metadata["sha256"], "random_seed": 42,
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (results / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(scores.to_string(index=False), flush=True)
    print(f"Report: {results / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
