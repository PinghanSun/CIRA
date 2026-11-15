import copy, hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from softcas.allocation import hard_group_assignment
from softcas.coco_evaluation import coco_bbox_report
from softcas.paper_protocol import build_matrix, native_config, settings
from softcas.paper_development import lock_stage_budget, freeze_matrix
from softcas.paper_data import single_exposure
from softcas.data import detection_record_batch, DetectionBatches
from softcas.baselines.runner import ImageBatches
from softcas.baselines.common import fitting_batch

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class CurrentNativeTests(unittest.TestCase):
    def test_current_blocks_and_archived_blocks_are_distinct(self):
        c = native_config(ROOT / "configs")
        self.assertEqual(
            (c["fitting"]["local_epochs"], c["fitting"]["receiving_epochs"]), (8, 4)
        )
        old = json.loads((ROOT / "configs/native_r1000.json").read_text())
        self.assertEqual(
            (old["fitting"]["local_epochs"], old["fitting"]["receiving_epochs"]),
            (32, 16),
        )

    def test_bdd_dev_has_complete_grid(self):
        m = build_matrix(blocks=["BDD_dev"])
        self.assertEqual(m["unique_jobs"], 54)
        self.assertEqual({j["recipe"]["seed"] for j in m["jobs"]}, {9001, 9002, 9003})
        self.assertTrue(all(j["recipe"]["phase"] == "development" for j in m["jobs"]))

    def test_single_exposure_jobs_never_reuse_repeated_content(self):
        m = build_matrix(blocks=["primary", "single_exposure"])
        ids = {v["job_id"] for v in m["views"] if v["block"] == "primary"}
        self.assertTrue(
            all(
                v["job_id"] not in ids
                for v in m["views"]
                if v["block"] == "single_exposure"
            )
        )
        self.assertEqual(m["view_counts"]["single_exposure"], 30)

    def compile_dev(self, root, leak=False):
        from softcas.paper_runner import compile_matrix

        m = build_matrix(blocks=["BDD_dev"])
        m["jobs"] = m["jobs"][:1]
        r = m["jobs"][0]["recipe"]
        records = [
            dict(
                image=role + ".png",
                image_sha256=role,
                scene_id=role,
                role=role,
                domain="fog",
                client="0",
                stage=1,
                boxes=[],
            )
            for role in ("D", "V", "A", "T")
        ]
        final = {"records": [{**x, "role": "V" if leak else "D"} for x in records]}
        cell = {
            "checkpoint": write(root / "checkpoint.json", {}),
            "registry": write(root / "final.json", final),
            "development_registry": write(
                root / "dev.json", {"records": records, "phase": "development"}
            ),
            "detector_config": write(root / "detector.json", {}),
            "contact_trace": write(root / "contacts.json", []),
        }
        key = "/".join(
            map(str, (r["dataset"], r["detector"], r["initialization"], r["seed"]))
        )
        return compile_matrix(
            m,
            {"formal": False, "cells": {key: cell}, "runs_root": str(root / "runs")},
            ROOT / "configs",
            root / "compiled",
        )

    def test_DEV_compiles_before_any_DEV_lock_exists(self):
        with tempfile.TemporaryDirectory() as t:
            plan = self.compile_dev(Path(t))
            self.assertEqual(plan["ready"], 1)
            config = json.loads(Path(plan["jobs"][0]["config"]).read_text())
            self.assertEqual(
                (
                    config["fitting"]["local_epochs"],
                    config["fitting"]["receiving_epochs"],
                ),
                (8, 4),
            )

    def test_DEV_cannot_use_any_final_heldout_group(self):
        with tempfile.TemporaryDirectory() as t:
            plan = self.compile_dev(Path(t), True)
            self.assertEqual(plan["ready"], 0)
            self.assertIn("final D", plan["blocked"][0]["reason"])

    def test_native_AdamW_config_is_accepted_by_real_assembly(self):
        import test_mechanism as fixture

        f = fixture.MechanismTests()
        f.setUp()
        model = f.backend.assembly(f.parent, f.cache, mode="refine")
        optimizer = model.optimizer(
            0.001,
            0.01,
            {
                "name": "AdamW",
                "momentum": 0.0,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        )
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        with self.assertRaises(ValueError):
            model.optimizer(0.001, 0.01, {"name": "AdamW", "momentum": 0.9})


class HardQuotaTests(unittest.TestCase):
    def groups(self):
        return [
            {"id": str(i), "images": 1, "class_boxes": [1, 0, 0, 0, 0, 0, 0, 0]}
            for i in range(12)
        ]

    def test_skewed_classes_cannot_change_exact_image_counts(self):
        p = np.tile([0.97, 0.01, 0.01, 0.01], (8, 1))
        a = hard_group_assignment(self.groups(), p, 1001, {"all": [3] * 4})
        self.assertEqual(a["achieved"], {"all": [3] * 4})
        self.assertEqual(
            a, hard_group_assignment(self.groups(), p, 1001, {"all": [3] * 4})
        )

    def test_coupled_weather_keeps_quotas_and_base_identity(self):
        groups = [
            {
                **g,
                "images": 2,
                "cell_images": {"clear|D": 1, "fog|D": 1},
                "cell_classes": {
                    "clear|D": g["class_boxes"],
                    "fog|D": g["class_boxes"],
                },
            }
            for g in self.groups()
        ]
        a = hard_group_assignment(
            groups, np.full((8, 4), 0.25), 1001, {"clear|D": [3] * 4, "fog|D": [3] * 4}
        )
        self.assertEqual(a["achieved"]["clear|D"], a["achieved"]["fog|D"])
        self.assertEqual(len(a["assignment"]), 12)

    def test_infeasible_whole_groups_are_rejected(self):
        groups = [{"id": str(i), "images": 2, "class_boxes": [1] * 8} for i in range(3)]
        with self.assertRaisesRegex(ValueError, "infeasible"):
            hard_group_assignment(
                groups, np.full((8, 4), 0.25), 1001, {"all": [2, 2, 1, 1]}
            )

    def test_quotas_must_match_original_total(self):
        with self.assertRaises(ValueError):
            hard_group_assignment(
                self.groups(), np.full((8, 4), 0.25), 1001, {"all": [4] * 4}
            )


class DevBudgetTests(unittest.TestCase):
    def fixture(self, root):
        m = build_matrix(blocks=["BDD_dev"])
        binding = write(root / "matrix.json", m)
        records = []
        for i, j in enumerate(m["jobs"]):
            r = j["recipe"]
            seen = (
                0.7
                if (r["stage_epochs"], r["attempts"]) in ((50, 8), (100, 16))
                else 0.5
            )
            row = dict(
                paper_job_id=j["id"],
                paper_recipe=r,
                role="DEV",
                complete=True,
                metric="AP50:95",
                selection_metric="final_macro_seen",
                clients=["0", "1", "2", "3"],
                stages=4,
                seen=seen,
                F_total=r["stage_epochs"] * 100.0,
                network_tx_bytes=10.0,
                study_kind="native_detector",
            )
            records.append(write(root / (str(i) + ".json"), row))
        return m, {"matrix": binding, "records": records, "cells": ["BDD100K/YOLOv8n"]}

    def test_uses_all_three_seeds_and_actual_cost_tie(self):
        with tempfile.TemporaryDirectory() as t:
            _, spec = self.fixture(Path(t))
            lock = lock_stage_budget(spec)
            self.assertEqual(
                (
                    lock["cells"]["BDD100K/YOLOv8n"]["stage_epochs"],
                    lock["cells"]["BDD100K/YOLOv8n"]["attempts"],
                ),
                (50, 8),
            )
            self.assertEqual(len(lock["inputs"]), 54)

    def test_missing_candidate_is_not_selected_away(self):
        with tempfile.TemporaryDirectory() as t:
            _, spec = self.fixture(Path(t))
            spec["records"].pop()
            with self.assertRaisesRegex(ValueError, "Missing"):
                lock_stage_budget(spec)

    def test_report_T_or_partial_clients_cannot_lock_DEV(self):
        with tempfile.TemporaryDirectory() as t:
            _, spec = self.fixture(Path(t))
            p = Path(spec["records"][0]["path"])
            row = json.loads(p.read_text())
            row["role"] = "T"
            spec["records"][0] = write(p, row)
            with self.assertRaises(ValueError):
                lock_stage_budget(spec)

    def test_formal_recipe_ids_are_refrozen_after_DEV(self):
        with tempfile.TemporaryDirectory() as t:
            _, spec = self.fixture(Path(t))
            lock = lock_stage_budget(spec)
            m = build_matrix(blocks=["BDD"])
            new = freeze_matrix(m, lock)
            self.assertEqual(new["unique_jobs"], 25)
            self.assertNotEqual(new["jobs"][0]["id"], m["jobs"][0]["id"])
            self.assertEqual({j["recipe"]["stage_epochs"] for j in new["jobs"]}, {50})
            self.assertEqual(
                {v["job_id"] for v in new["views"]}, {j["id"] for j in new["jobs"]}
            )


class CrowdTests(unittest.TestCase):
    def test_score_ordered_matching_and_101_recall(self):
        p = [[[0, 0, 10, 10, 0.9, 0]]]
        t = [[[0, 0, 0, 10, 10]]]
        r = coco_bbox_report(p, t, ["a", "absent"])
        self.assertAlmostEqual(r["AP50_95"], 1.0)
        self.assertIsNone(r["per_class"][1]["AP50_95"])

    def test_crowd_overlap_is_not_a_false_positive(self):
        pred = [[[1, 1, 2, 2, 0.99, 0], [20, 20, 30, 30, 0.9, 0]]]
        target = {
            "boxes": [[0, 0, 0, 10, 10], [0, 20, 20, 30, 30]],
            "iscrowd": [1, 0],
            "ignore": [0, 0],
        }
        ignored = coco_bbox_report(pred, [target], ["a"])
        plain = coco_bbox_report(
            pred, [{"boxes": target["boxes"], "iscrowd": [0, 0]}], ["a"]
        )
        self.assertAlmostEqual(ignored["AP50_95"], 1.0)
        self.assertLess(plain["AP50_95"], 1.0)
        self.assertEqual(ignored["GT_boxes"], 1)

    def test_explicit_ignore_is_excluded_from_support(self):
        r = coco_bbox_report(
            [[]], [{"boxes": [[0, 0, 0, 10, 10]], "ignore": [1]}], ["a"]
        )
        self.assertIsNone(r["AP50_95"])
        self.assertEqual(r["ignored_GT"], 1)

    def test_max_dets_is_per_image_category(self):
        pred = [[20 + i, 20, 20.5 + i, 20.5, 1 - i * 0.001, 0] for i in range(100)] + [
            [0, 0, 10, 10, 0.1, 0]
        ]
        r = coco_bbox_report([pred], [[[0, 0, 0, 10, 10]]], ["a"])
        self.assertEqual(r["AP50_95"], 0.0)

    def test_metadata_survives_geometry_and_is_not_fitting_GT(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.png"
            Image.fromarray(np.zeros((10, 20, 3), np.uint8)).save(p)
            record = {
                "image": str(p),
                "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "boxes": [[0, 0.2, 0.5, 0.2, 0.4]],
                "evaluation_annotations": [
                    {"box": [0, 0.2, 0.5, 0.2, 0.4], "iscrowd": False},
                    {"box": [1, 0.8, 0.5, 0.2, 0.4], "iscrowd": True, "ignore": True},
                ],
            }
            b = detection_record_batch(record, p.parent, 20, view="hflip")
            target = b["evaluation_targets"][0]
            self.assertEqual(len(b["bboxes"]), 1)
            self.assertEqual(len(target["boxes"]), 2)
            self.assertEqual(target["iscrowd"].tolist(), [False, True])
            self.assertGreater(float(target["boxes"][0, 1]), 10)


class ExposureTests(unittest.TestCase):
    def test_each_base_keeps_one_weather_and_reporting_stays_intact(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            rows = []
            domains = ["clear", "fog_005", "fog_010", "fog_020"]
            for base in range(16):
                for stage, domain in enumerate(domains, 1):
                    rows.append(
                        {
                            "image": f"{base}_{domain}.png",
                            "image_sha256": str(base) + "_" + domain,
                            "scene_id": str(base),
                            "role": "D",
                            "client": "0",
                            "stage": stage,
                            "domain": domain,
                            "boxes": [],
                        }
                    )
            write(
                root / "source.json",
                {
                    "records": rows,
                    "exposure_policy": "repeated_content",
                    "grouping_basis": "known_identity",
                },
            )
            r = single_exposure(
                {
                    "registry": str(root / "source.json"),
                    "output": str(root / "out.json"),
                    "domains": domains,
                    "seed": 1001,
                }
            )
            self.assertEqual(len(r["records"]), 16)
            self.assertEqual(len({x["scene_id"] for x in r["records"]}), 16)
            self.assertEqual(
                sorted(r["single_exposure"]["strata"][0]["weather_counts"].values()),
                [4] * 4,
            )

    def test_external_reader_matches_CIRA_views_and_resumes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = root / "x.png"
            im = np.zeros((12, 20, 3), np.uint8)
            im[:, 2:5] = 255
            Image.fromarray(im).save(p)
            record = {
                "image": str(p),
                "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "boxes": [[0, 0.2, 0.5, 0.2, 0.5]],
            }
            external = ImageBatches(
                [record], root, 20, 1, "cpu", "acdc_four_view", 1001
            )
            local = DetectionBatches([record], root, 20, 1, "acdc_four_view", 1001)
            for epoch in range(4):
                b = fitting_batch(external, 0)
                c = local.with_epoch(epoch, 1001)[0]
                self.assertTrue(torch.equal(b["img"], c["img"]))
                self.assertTrue(torch.equal(b["bboxes"], c["bboxes"]))
            state = external.state_dict()
            other = ImageBatches([record], root, 20, 1, "cpu", "acdc_four_view", 1001)
            other.load_state_dict(state)
            self.assertTrue(
                torch.equal(
                    fitting_batch(external, 0)["img"], fitting_batch(other, 0)["img"]
                )
            )

    def test_evaluation_reader_never_advances_fitting_views(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.png"
            Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(p)
            r = {
                "image": str(p),
                "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "boxes": [],
            }
            batches = ImageBatches([r], p.parent, 10, 1, "cpu")
            a = batches[0]
            b = batches[0]
            self.assertTrue(torch.equal(a["img"], b["img"]))
            self.assertEqual(batches.fit_visits, [0])


class AnalysisDeliveryTests(unittest.TestCase):
    def module(self, name, relative):
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_new_endpoint_table_figure_and_paired_errors_are_generated(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            bindings = []
            for method in ("softcas", "serial"):
                for seed in range(1001, 1006):
                    bindings.append(
                        write(
                            root / (method + str(seed) + ".json"),
                            {
                                "complete": True,
                                "cell": "fixture",
                                "method": method,
                                "seed": seed,
                                "seen": 0.6
                                + (seed - 1001) * 0.002
                                + (0.02 if method == "softcas" else 0),
                            },
                        )
                    )
            spec = {
                "evidence_scope": "software_fixture",
                "trajectory_files": bindings,
                "seeds": list(range(1001, 1006)),
                "reference": "softcas",
                "comparators": ["serial"],
            }
            r = self.module("result_probe", "analysis/render_results.py").render(
                spec, root / "rendered"
            )
            self.assertEqual(r["paired_contrasts"][0]["n_pair"], 5)
            self.assertAlmostEqual(r["paired_contrasts"][0]["mean"], 0.02)
            self.assertTrue((root / "rendered/quality.pdf").is_file())
            self.assertTrue((root / "rendered/quality.tex").is_file())

    def test_supplementary_source_manifest_is_frozen(self):
        m = self.module("study_probe", "studies/run.py").verify_sources()
        self.assertEqual((m["feature_cases"], m["feature_main_cases"]), (5502, 420))


if __name__ == "__main__":
    unittest.main()
