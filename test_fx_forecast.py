"""Checks for the material risks: source normalization, metrics and time leakage."""

import unittest

import numpy as np
import pandas as pd

from fx_forecast import metrics, parse_cbr_xml, rolling_predictions, split_positions


class SourceTests(unittest.TestCase):
    def test_normalizes_nominal_and_decimal_comma_and_sorts(self):
        xml = b'''<ValCurs ID="R01235"><Record Date="10.01.2024"><Nominal>10</Nominal><Value>912,3000</Value></Record><Record Date="09.01.2024"><Nominal>1</Nominal><Value>90,50</Value></Record></ValCurs>'''
        frame = parse_cbr_xml(xml)
        self.assertEqual(frame.date.iloc[0], pd.Timestamp("2024-01-09"))
        np.testing.assert_allclose(frame.rate, [90.5, 91.23])

    def test_rejects_duplicate_dates(self):
        record = '<Record Date="10.01.2024"><Nominal>1</Nominal><Value>91,23</Value></Record>'
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_cbr_xml(f'<ValCurs ID="R01235">{record}{record}</ValCurs>'.encode())

    def test_rejects_wrong_currency_and_invalid_rates(self):
        for xml in [b'<ValCurs ID="R01239"/>', b'<ValCurs ID="R01235"/>',
                    b'<ValCurs ID="R01235"><Record Date="10.01.2024"><Nominal>0</Nominal><Value>91</Value></Record></ValCurs>',
                    b'<ValCurs ID="R01235"><Record Date="10.01.2024"><Nominal>1</Nominal><Value>nan</Value></Record></ValCurs>']:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                parse_cbr_xml(xml)


class EvaluationTests(unittest.TestCase):
    def test_wape_is_weighted_and_mae_is_in_source_units(self):
        result = metrics([10, 100], [11, 90])
        self.assertAlmostEqual(result["wape_pct"], 10.0)
        self.assertAlmostEqual(result["mae_rub"], 5.5)
        self.assertAlmostEqual(result["rmse_rub"], np.sqrt(50.5))

    def test_metrics_reject_missing_or_unaligned_values(self):
        for actual, forecast in [([], []), ([0], [1]), ([1, 2], [1]), ([1], [np.nan])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                metrics(actual, forecast)

    def test_rolling_predictor_never_sees_target_or_future(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-09", "2025-01-10", "2025-01-11", "2025-01-14", "2025-01-15"]),
                              "rate": [90., 91., 92., 93., 94.]})
        histories = []
        def spy(history, target_date):
            self.assertLess(history.date.max(), target_date)
            histories.append(history.rate.tolist())
            return float(history.rate.iloc[-1]), ""
        predictions = rolling_predictions(frame, [2, 3, 4], spy, window=2)
        self.assertEqual(histories, [[90., 91.], [91., 92.], [92., 93.]])
        self.assertEqual(predictions.prediction.tolist(), [91., 92., 93.])
        self.assertEqual(predictions.actual.tolist(), [92., 93., 94.])

    def test_rejects_nonchronological_history(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-12", "2025-01-10", "2025-01-11"]), "rate": [90., 91., 92.]})
        with self.assertRaisesRegex(ValueError, "precede"):
            rolling_predictions(frame, [2], lambda h, d: (90., ""), window=2)

    def test_split_reserves_validation_and_test(self):
        self.assertEqual(split_positions(2000), (1820, 1880))
        with self.assertRaises(ValueError):
            split_positions(900)


if __name__ == "__main__":
    unittest.main()
