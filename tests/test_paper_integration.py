import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import test_mechanism as fixture
from softcas.mechanism import (
    _continuation_initial,
    continuation_comparison,
    receiving_comparison,
)
from softcas.paper_data import make_registry, scalable_group_assignment, convert_bdd
from softcas.paper_protocol import LABELS

torch.set_num_threads(1)


class PaperMechanismTests(unittest.TestCase):
    def test_local_only_reset_preserves_actual_nonzero_peer_logits(self):
        f = fixture.MechanismTests()
        f.setUp()
        branch = _continuation_initial(
            f.parent, f.backend, f.cache, 50, "reset_local_only"
        )
        self.assertEqual(branch.beta, f.parent.beta)
        self.assertNotEqual(branch.phi, f.parent.phi)
        self.assertEqual(branch.C, f.parent.C)

    def test_four_continuation_branches_execute_at_matched_updates(self):
        f = fixture.MechanismTests()
        f.setUp()
        with tempfile.TemporaryDirectory() as t:
            f.backend.output = Path(t)
            r = continuation_comparison(
                f.backend,
                f.cache,
                f.D,
                f.V,
                f.T,
                f.parent,
                {**f.spec, "include_reset_local_only": True},
                f.c,
                t,
            )
            self.assertEqual(r["status"], "completed")
            self.assertEqual(len(r["rows"]), 4)
            self.assertTrue(r["matched_work_eligible"])

    def test_frozen_parent_logit_control_is_a_real_extra_readout(self):
        f = fixture.MechanismTests()
        f.setUp()
        with tempfile.TemporaryDirectory() as t:
            f.backend.output = Path(t)
            r = receiving_comparison(
                f.backend,
                f.cache,
                f.D,
                f.V,
                f.T,
                f.parent,
                {**f.spec, "include_frozen_parent_logits": True},
                f.c,
                t,
            )
            row = next(x for x in r["rows"] if x["branch"] == "frozen_parent_logits")
            self.assertEqual(row["completed_fit_steps"], 0)
            self.assertEqual(row["beta"], dict(f.parent.beta))


class PaperDataTests(unittest.TestCase):
    def test_existing_csv_archive_bridge_preserves_pixels_and_boxes(self):
        import zipfile, csv, io
        from softcas.paper_data import import_allocation

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            buf = io.BytesIO()
            Image.fromarray(np.zeros((10, 20, 3), np.uint8)).save(buf, format="PNG")
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("image.png", buf.getvalue())
            values = {
                "resource_id": "resource1",
                "dataset": "ACDC",
                "domain": "fog",
                "active": "True",
                "role": "D",
                "client": "C1",
                "stage": "1",
                "allocation_group": "known_group",
                "confirmed_identity_group": "identity1",
                "archive": str(archive),
                "entry": "image.png",
                "has_detection_labels": "True",
                "width": "20",
                "height": "10",
                "noncrowd_boxes": "1",
            }
            alloc = root / "allocation.csv"
            with alloc.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=values)
                w.writeheader()
                w.writerow(values)
            ann = root / "boxes.json"
            ann.write_text(
                json.dumps(
                    {"resource1": [{"class_id": 2, "bbox": [2, 1, 8, 6], "iscrowd": 0}]}
                )
            )
            result = import_allocation(
                {
                    "allocations": [str(alloc)],
                    "annotations": [str(ann)],
                    "output_directory": str(root / "converted"),
                    "known_identities_checked": True,
                }
            )
            r = result["records"][0]
            self.assertEqual(r["client"], "0")
            self.assertEqual(r["boxes"][0], [2, 0.3, 0.4, 0.4, 0.6])
            self.assertEqual(Path(r["image"]).read_bytes(), buf.getvalue())

    def test_csv_bridge_rejects_missing_GT_before_extraction(self):
        import csv
        from softcas.paper_data import import_allocation

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = root / "a.csv"
            v = {
                "resource_id": "one",
                "role": "D",
                "active": "True",
                "noncrowd_boxes": "3",
                "has_detection_labels": "True",
                "archive": "not_read.zip",
                "entry": "x.png",
            }
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=v)
                w.writeheader()
                w.writerow(v)
            with self.assertRaisesRegex(ValueError, "Missing human GT"):
                import_allocation(
                    {"allocations": [str(p)], "output_directory": str(root / "out")}
                )
            self.assertFalse((root / "out").exists())

    def test_allocation_preserves_frozen_cells_and_actual_images(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            records = []
            domains = ["fog", "night", "rain", "snow"]
            for client in range(4):
                for stage in range(1, 5):
                    for role in ("D", "V", "A"):
                        i = len(records)
                        p = root / f"{i}.png"
                        Image.fromarray(np.full((8, 12, 3), i, np.uint8)).save(p)
                        records.append(
                            dict(
                                image=str(p),
                                scene_id=str(i),
                                verified_known_identity=True,
                                role=role,
                                client=str(client),
                                stage=stage,
                                domain=domains[stage - 1],
                                boxes=[[i % 8, 0.5, 0.5, 0.5, 0.5]],
                            )
                        )
            src = root / "source.json"
            src.write_text(json.dumps({"labels": LABELS, "records": records}))
            r = make_registry(
                {
                    "source_records": str(src),
                    "output": str(root / "registry.json"),
                    "seed": 1001,
                    "domains": domains,
                    "preserve_allocation": True,
                    "formal": True,
                }
            )
            self.assertFalse(r["missing_online_cells"])
            self.assertEqual(len(r["class_support"]), 48)
            self.assertEqual(
                [(x["client"], x["stage"], x["role"]) for x in records],
                [(x["client"], x["stage"], x["role"]) for x in r["records"]],
            )

    def test_scalable_assignment_covers_all_groups_reproducibly(self):
        groups = [
            {
                "id": str(i),
                "images": 1,
                "class_boxes": [int(k == i % 8) for k in range(8)],
            }
            for i in range(1500)
        ]
        targets = np.full((8, 4), 0.25)
        a = scalable_group_assignment(groups, targets, 1001)
        self.assertEqual(a, scalable_group_assignment(groups, targets, 1001))
        self.assertEqual(len(a["assignment"]), 1500)
        self.assertEqual(set(a["assignment"].values()), {0, 1, 2, 3})

    def test_bdd_conversion_excludes_noncommon_classes_and_reserves_unknowns(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = root / "sample.png"
            Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(p)
            row = {
                "name": p.name,
                "attributes": {"timeofday": "daytime", "weather": "clear"},
                "labels": [
                    {
                        "category": "pedestrian",
                        "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 15},
                    },
                    {"category": "traffic light"},
                ],
            }
            ann = root / "ann.json"
            ann.write_text(json.dumps([row, {"name": "reserve.png", "attributes": {}}]))
            ids = root / "ids.json"
            ids.write_text(
                json.dumps(
                    {p.name: {"scene_id": "rideA", "verified_known_identity": True}}
                )
            )
            r = convert_bdd(
                {
                    "annotations": str(ann),
                    "identity_map": str(ids),
                    "image_root": str(root),
                    "output": str(root / "bdd.json"),
                }
            )
            self.assertEqual(len(r["records"]), 1)
            self.assertEqual(r["records"][0]["boxes"][0][0], 0)
            self.assertEqual(r["excluded_categories"]["traffic light"], 1)
            self.assertEqual(r["reserved_unknown_attributes"], ["reserve.png"])


if __name__ == "__main__":
    unittest.main()
