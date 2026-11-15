import json
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from unittest.mock import patch
from softcas.evaluation import detection_report, detection_ap
from softcas.baselines.common import DetectorTask


class DetectionReportTests(unittest.TestCase):
    def fixture(self):
        p = [
            torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
                    [10.0, 10.0, 18.0, 18.0, 0.8, 1.0],
                    [0.0, 0.0, 10.0, 10.0, 0.7, 2.0],
                ]
            )
        ]
        t = [
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 10.0, 10.0],
                    [1.0, 10.0, 10.0, 20.0, 20.0],
                    [1.0, 30.0, 30.0, 40.0, 40.0],
                ]
            )
        ]
        return p, t

    def test_all_metrics_come_from_same_PR_pool(self):
        p, t = self.fixture()
        r = detection_report(p, t, ["a", "b", "unsupported"])
        self.assertEqual([c["GT_support"] for c in r["per_class"]], [1, 2, 0])
        self.assertIsNone(r["per_class"][2]["AP50_95"])
        self.assertGreater(r["AP50"], r["AP75"])
        self.assertAlmostEqual(
            r["AP50_95"], sum(c["AP50_95"] for c in r["per_class"][:2]) / 2
        )
        self.assertAlmostEqual(r["AP50_95"], detection_ap(p, t), places=10)
        self.assertLessEqual(r["AP50_95"], r["AP50"])

    def test_supported_no_predictions_zero_unsupported_null(self):
        p = [torch.empty((0, 6))]
        t = [torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0]])]
        r = detection_report(p, t, ["present", "absent"])
        self.assertEqual(r["AP50_95"], 0.0)
        self.assertEqual(r["per_class"][0]["AP75"], 0.0)
        self.assertIsNone(r["per_class"][1]["AP75"])

    def test_no_GT_unavailable(self):
        r = detection_report([torch.empty((0, 6))], [torch.empty((0, 5))], ["a"])
        self.assertIsNone(r["AP50_95"])
        self.assertEqual(r["supported_classes"], 0)

    def test_malformed_or_unpaired_evidence_rejected(self):
        p, t = self.fixture()
        for a, b in [
            (p, t + t),
            ([], []),
            ([torch.zeros((2, 5))], t),
            ([torch.tensor([[0.0, 0.0, 10.0, 10.0, float("nan"), 0.0]])], t),
            ([torch.tensor([[0.0, 0.0, 10.0, 10.0, 1.0, 0.5]])], t),
            ([torch.tensor([[1.0, 0.0, 0.0, 10.0, 1.0, 0.0]])], t),
        ]:
            with self.subTest(a=str(a)[:40]), self.assertRaises(ValueError):
                detection_report(a, b)

    def test_saved_prediction_replays_all_metrics(self):
        p, t = self.fixture()
        task = DetectorTask()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(task, "_prediction_evidence", return_value=(p, t)),
        ):
            path = Path(directory) / "evidence.json"
            r = task.report(nn.Linear(1, 1), None, evidence_path=path)
            saved = json.loads(path.read_bytes())
            replay = detection_report(
                [torch.tensor(v).reshape(-1, 6) for v in saved["predictions"]],
                [torch.tensor(v).reshape(-1, 5) for v in saved["targets"]],
                saved["class_names"],
            )
            self.assertEqual(r["per_class"], replay["per_class"])
            self.assertEqual(r["AP50_95"], replay["AP50_95"])
            self.assertEqual(len(r["prediction_GT_SHA256"]), 64)


if __name__ == "__main__":
    unittest.main()
