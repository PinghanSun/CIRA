"""One CPU native-head preparation step on generated images, not a benchmark."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from softcas.initialization import run_yolo_initialization
from softcas.paper_protocol import LABELS


class NativeInitializationInterfaceTest(unittest.TestCase):
    def test_ema_pseudo_box_pipeline_saves_reloadable_eight_class_checkpoint(self):
        from ultralytics import YOLO

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            paths = {}
            for name, role, n in [
                ("L", "SOURCE", 2),
                ("V", "SOURCE_VAL", 1),
                ("U", "U", 2),
            ]:
                rows = []
                for i in range(n):
                    p = root / f"{name}{i}.png"
                    Image.fromarray(np.full((64, 64, 3), 80 + i * 20, np.uint8)).save(p)
                    import hashlib

                    rows.append(
                        {
                            "image": str(p),
                            "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "scene_id": f"{name}{i}",
                            "role": role,
                            "domain": "clear" if name != "U" else "fog",
                            "boxes": [[0, 0.5, 0.5, 0.4, 0.4]] if name != "U" else [],
                        }
                    )
                dest = root / (name + ".json")
                dest.write_text(json.dumps({"labels": LABELS, "records": rows}))
                paths[name] = str(dest)
            result = run_yolo_initialization(
                {
                    "condition": "C2",
                    "detector": "YOLOv8n",
                    "seed": 1001,
                    "device": "cpu",
                    "image_size": 64,
                    "batch_size": 2,
                    "allow_random_initialization": True,
                    "labeled_manifest": paths["L"],
                    "validation_manifest": paths["V"],
                    "unlabeled_manifest": paths["U"],
                    "output": str(root / "run"),
                    "recipe": {
                        "updates": 1,
                        "validation_interval": 1,
                        "lr": 1e-4,
                        "teacher_momentum": 0.9,
                        "pseudo_confidence": 0.001,
                        "unlabeled_weight": 0.1,
                    },
                }
            )
            self.assertEqual(result["actual_optimizer_updates"], 1)
            self.assertEqual(result["U_image_reads"], 2)
            model = YOLO(result["checkpoint"]).model
            self.assertEqual(tuple(model.names.values()), LABELS)
            self.assertTrue((root / "run" / "teacher.pt").is_file())
            from ultralytics.nn.tasks import DetectionModel

            source = DetectionModel("yolov8n.yaml", nc=80, verbose=False)
            source_path = root / "eighty_class_software_fixture.pt"
            torch.save({"model": source}, source_path)
            c1 = run_yolo_initialization(
                {
                    "condition": "C1",
                    "detector": "YOLOv8n",
                    "seed": 1001,
                    "device": "cpu",
                    "initial_checkpoint": str(source_path),
                    "initial_checkpoint_type": "coco_compatible",
                    "image_size": 64,
                    "batch_size": 2,
                    "labeled_manifest": paths["L"],
                    "validation_manifest": paths["V"],
                    "output": str(root / "c1"),
                    "recipe": {
                        "updates": 1,
                        "validation_interval": 1,
                        "lr": 1e-4,
                        "unlabeled_weight": None,
                    },
                }
            )
            self.assertGreater(c1["COCO_transferred_elements"], 0)
            self.assertEqual(c1["U_image_reads"], 0)


if __name__ == "__main__":
    unittest.main()
