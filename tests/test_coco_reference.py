"""Compare bbox matching/accumulation to frozen official COCOeval source."""

import contextlib, io, importlib.util, json, sys, types, unittest
from pathlib import Path
import numpy as np
from softcas.coco_evaluation import coco_bbox_report


def bbox_iou(d, g, crowd):
    out = np.zeros((len(d), len(g)))
    for i, (x, y, w, h) in enumerate(d):
        for j, (a, b, c, e) in enumerate(g):
            inter = max(0, min(x + w, a + c) - max(x, a)) * max(
                0, min(y + h, b + e) - max(y, b)
            )
            union = w * h if crowd[j] else w * h + c * e - inter
            out[i, j] = inter / union if union else 0
    return out


def reference(predictions, targets, classes):
    package = "cira_test_coco_reference"
    module = types.ModuleType(package)
    module.__path__ = []
    mask = types.ModuleType(package + ".mask")
    mask.iou = bbox_iou
    sys.modules[package] = module
    sys.modules[package + ".mask"] = mask
    spec = importlib.util.spec_from_file_location(
        package + ".cocoeval", Path(__file__).parent / "coco_reference/cocoeval.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    class NumpyCompat:
        def __getattr__(self, key):
            return float if key == "float" else getattr(np, key)

    evaluator.np = NumpyCompat()

    class API:
        def __init__(self, rows):
            self.rows = rows

        def getImgIds(self):
            return list(range(len(predictions)))

        def getCatIds(self):
            return list(range(classes))

        def getAnnIds(self, imgIds, catIds):
            return [
                r["id"]
                for r in self.rows
                if r["image_id"] in imgIds and r["category_id"] in catIds
            ]

        def loadAnns(self, ids):
            return [r.copy() for r in self.rows if r["id"] in ids]

    gt = []
    dt = []
    for image, (p, t) in enumerate(zip(predictions, targets)):
        for box, crowd in zip(t["boxes"], t["iscrowd"]):
            category, x, y, u, v = box
            gt.append(
                dict(
                    id=len(gt) + 1,
                    image_id=image,
                    category_id=int(category),
                    bbox=[x, y, u - x, v - y],
                    area=(u - x) * (v - y),
                    iscrowd=int(crowd),
                )
            )
        for x, y, u, v, score, category in p:
            dt.append(
                dict(
                    id=len(dt) + 1,
                    image_id=image,
                    category_id=int(category),
                    bbox=[x, y, u - x, v - y],
                    area=(u - x) * (v - y),
                    score=score,
                )
            )
    e = evaluator.COCOeval(API(gt), API(dt), "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
    values = e.eval["precision"][:, :, :, 0, -1]
    return [
        float(values[:, :, c][values[:, :, c] >= 0].mean())
        if (values[:, :, c] >= 0).any()
        else None
        for c in range(classes)
    ]


class CocoReferenceTests(unittest.TestCase):
    def test_frozen_reference_source_hashes(self):
        import hashlib

        root = Path(__file__).parent / "coco_reference"
        for name, binding in json.loads((root / "SOURCE.json").read_text()).items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(),
                binding["sha256"],
            )

    def test_crowd_and_score_matching_agree_on_randomized_pools(self):
        rng = np.random.default_rng(20261114)
        for case in range(24):
            targets = []
            predictions = []
            for image in range(3):
                boxes = []
                crowd = []
                pred = []
                for j in range(8):
                    x, y = rng.uniform(0, 60, 2)
                    w, h = rng.uniform(2, 25, 2)
                    category = int(rng.integers(3))
                    boxes.append([category, x, y, x + w, y + h])
                    crowd.append(int(rng.random() < 0.25))
                    for _ in range(int(rng.integers(3))):
                        delta = rng.normal(0, 0.8, 4)
                        z = np.array([x, y, x + w, y + h]) + delta
                        if z[2] > z[0] and z[3] > z[1]:
                            pred.append([*z, float(rng.random()), category])
                targets.append({"boxes": boxes, "iscrowd": crowd})
                predictions.append(pred)
            expected = reference(predictions, targets, 4)
            actual = coco_bbox_report(predictions, targets, ["a", "b", "c", "absent"])
            for c, value in enumerate(expected):
                if value is None:
                    self.assertIsNone(actual["per_class"][c]["AP50_95"])
                else:
                    self.assertAlmostEqual(
                        actual["per_class"][c]["AP50_95"],
                        value,
                        places=12,
                        msg=str(case),
                    )


if __name__ == "__main__":
    unittest.main()
