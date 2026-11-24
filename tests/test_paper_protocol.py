import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from softcas.paper_protocol import (
    settings,
    build_matrix,
    native_config,
    export,
    bdd_domain,
)
from softcas.augmentation import apply_view, view_for
from softcas.data import validate_registry, DetectionBatches, load_detection_stage
from softcas.fitting import shuffled_batches
from softcas.initialization import train_initialization, ema_update, validate_recipe
from softcas.paper_runner import compile_matrix, execute_job

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class CurrentProtocolTests(unittest.TestCase):
    def test_receiver_study_keeps_the_pilot_separate_and_pairs_each_arm(self):
        from softcas.paper_protocol import build_receiver_study
        from softcas.state import digest

        pilot = build_receiver_study("pilot")
        self.assertEqual(pilot["unique_jobs"], 2)
        self.assertEqual(
            {
                (j["recipe"]["dataset"], j["recipe"]["seed"], j["recipe"]["phase"])
                for j in pilot["jobs"]
            },
            {("ACDC", 9001, "development")},
        )
        core = build_receiver_study("core")
        self.assertEqual(core["unique_jobs"], 48)
        self.assertEqual(core["protocol_sha256"], digest(settings()))
        self.assertTrue(all(j["id"] == digest(j["recipe"]) for j in core["jobs"]))
        for dataset in ("ACDC", "ControlledFog"):
            for seed in range(1001, 1009):
                recipes = [
                    j["recipe"]
                    for j in core["jobs"]
                    if (j["recipe"]["dataset"], j["recipe"]["seed"]) == (dataset, seed)
                ]
                self.assertEqual(
                    {(r["method"], r["ablation"]) for r in recipes},
                    {
                        ("softcas", "none"),
                        ("softcas", "no_handoff"),
                        ("serial", "none"),
                    },
                )
        self.assertEqual(
            build_receiver_study("extended", range(1001, 1009))["unique_jobs"], 80
        )
        with self.assertRaises(ValueError):
            build_receiver_study("core", [9001])

    def test_deduplicated_primary_and_initialization_controls(self):
        p = build_matrix(blocks=["primary", "U", "BDD"])
        self.assertEqual(p["view_counts"], {"BDD": 25, "U": 150, "primary": 420})
        self.assertEqual(p["unique_jobs"], 545)
        self.assertEqual(len({j["id"] for j in p["jobs"]}), 545)
        self.assertFalse(p["execution_started"])

    def test_all_sweeps_register_actual_native_jobs(self):
        p = build_matrix(blocks=["F", "R", "alpha", "robustness", "dev", "ablations"])
        self.assertEqual(
            p["view_counts"],
            {
                "F": 180,
                "R": 180,
                "ablations": 210,
                "alpha": 900,
                "dev": 324,
                "robustness": 600,
            },
        )
        self.assertTrue(all(j["status"] == "not_executed" for j in p["jobs"]))
        self.assertEqual(
            {
                j["recipe"]["stage_epochs"]
                for j in p["jobs"]
                if j["recipe"]["phase"] == "archived_round_protocol"
            },
            {150, 250, 500},
        )

    def test_native_budget_has_no_archived_round_override(self):
        for e in (50, 100, 200):
            for m in (8, 16, 32):
                c = native_config(ROOT / "configs", e, m)
                self.assertEqual(c["fitting"]["stage_epochs"], e)
                self.assertEqual(c["budgets"]["search"]["E"], m + 1)
                self.assertIsNone(c["round_budget"])

    def test_export_and_empty_binding_compile_are_explicit(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "export"
            export(out, ROOT / "configs")
            self.assertEqual(len(list(out.glob("native*.json"))), 9)
            matrix = build_matrix(blocks=["BDD"])
            plan = compile_matrix(matrix, {}, ROOT / "configs", Path(t) / "compiled")
            self.assertEqual((plan["ready"], plan["blocked_count"]), (0, 25))
            self.assertFalse(plan["training_started"])

    def test_config_export_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            Path(t, "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                export(t, ROOT / "configs")

    def test_known_identity_is_not_a_scene_independence_claim(self):
        r = {
            "image": "a",
            "image_sha256": "1",
            "scene_id": "base",
            "role": "D",
            "client": "0",
            "stage": 1,
            "domain": "clear",
            "verified_known_identity": True,
        }
        s = {**r, "image": "b", "image_sha256": "2", "stage": 2, "domain": "fog"}
        with self.assertRaises(ValueError):
            validate_registry([r, s], True)
        validate_registry(
            [r, s],
            True,
            exposure_policy="repeated_content",
            grouping_basis="known_identity",
        )
        with self.assertRaises(ValueError):
            validate_registry(
                [r, {**s, "role": "V"}],
                True,
                exposure_policy="repeated_content",
                grouping_basis="known_identity",
            )
        with self.assertRaises(ValueError):
            validate_registry(
                [r, {**s, "client": "1"}],
                True,
                exposure_policy="repeated_content",
                grouping_basis="known_identity",
            )

    def test_unknown_group_cannot_be_promoted(self):
        r = {
            "image": "a",
            "image_sha256": "1",
            "scene_id": "x",
            "role": "D",
            "client": "0",
            "stage": 1,
            "domain": "fog",
        }
        with self.assertRaises(ValueError):
            validate_registry([r], True, grouping_basis="known_identity")

    def test_coarse_allocation_group_remains_in_one_client(self):
        r = {
            "image": "a",
            "image_sha256": "1",
            "scene_id": "x",
            "allocation_group": "nearby",
            "role": "D",
            "client": "0",
            "stage": 1,
            "domain": "fog",
        }
        with self.assertRaises(ValueError):
            validate_registry(
                [
                    r,
                    {
                        **r,
                        "image": "b",
                        "image_sha256": "2",
                        "scene_id": "y",
                        "client": "1",
                    },
                ]
            )

    def test_alpha_reassignment_keeps_weather_variants_together(self):
        from softcas.conditions import scene_alpha

        records = []
        for i in range(16):
            for d in ("clear", "fog"):
                records.append(
                    {
                        "scene_id": str(i),
                        "role": "D",
                        "domain": d,
                        "client": str(i % 4),
                        "boxes": [[i % 8, 0.5, 0.5, 0.2, 0.2]],
                    }
                )
        result, evidence = scene_alpha(
            {"records": records, "exposure_policy": "repeated_content"},
            ["clear", "fog"],
            ["0", "1", "2", "3"],
            1001,
            0.3,
            2,
        )
        self.assertTrue(evidence["coupled_weather_groups"])
        for i in range(16):
            self.assertEqual(
                len(
                    {r["client"] for r in result["records"] if r["scene_id"] == str(i)}
                ),
                1,
            )

    def test_bdd_domain_priority_and_unknown_reserve(self):
        self.assertEqual(
            bdd_domain({"timeofday": "night", "weather": "rainy"}), "night"
        )
        self.assertEqual(
            bdd_domain({"timeofday": "daytime", "weather": "snowy"}), "adverse"
        )
        self.assertIsNone(bdd_domain({"weather": "undefined"}))

    def test_paper_counts_are_separate_from_views(self):
        p = settings()
        self.assertEqual(p["augmentation"]["view_recipes"], 4864)
        self.assertEqual(p["augmentation"]["originals"], 1216)
        self.assertIsNone(p["initialization"]["pseudo_confidence"])
        self.assertFalse(p["partition"]["scene_independence_verified"])


class GeometricViewTests(unittest.TestCase):
    def test_flip_moves_pixels_and_boxes_together(self):
        image = np.zeros((10, 20, 3), np.float32)
        image[4:6, 2:6] = 1
        p, b = apply_view(image, [[2, 0.2, 0.5, 0.2, 0.2]], "hflip")
        self.assertTrue(np.all(p[4:6, 14:18] == 1))
        self.assertAlmostEqual(float(b[0, 1]), 0.8, places=6)

    def test_rotation_clips_boxes_and_preserves_labels(self):
        p, b = apply_view(
            np.ones((20, 30, 3), np.float32),
            [[3, 0.05, 0.05, 0.1, 0.1]],
            "rotate_plus5",
        )
        self.assertEqual(p.shape, (20, 30, 3))
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(int(b[0, 0]), 3)
        self.assertTrue((b[:, 1:] >= 0).all())
        self.assertTrue((b[:, 1:] <= 1).all())

    def test_four_epochs_cover_each_recipe_once(self):
        self.assertEqual(
            len({view_for("fixed_original", 1001, e) for e in range(4)}), 4
        )
        self.assertEqual(
            view_for("fixed_original", 1001, 0), view_for("fixed_original", 1001, 4)
        )

    def test_loader_visits_originals_once_and_leaves_V_A_unaugmented(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            records = []
            for i, role in enumerate(("D", "V", "A")):
                im = np.zeros((12, 20, 3), np.uint8)
                im[:, 1 + i] = 255
                path = root / f"{role}.png"
                Image.fromarray(im).save(path)
                records.append(
                    dict(
                        image=str(path),
                        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        scene_id=role,
                        role=role,
                        client="0",
                        stage=1,
                        domain="fog",
                        boxes=[[0, 0.2, 0.5, 0.2, 0.2]],
                        verified_known_identity=True,
                    )
                )
            manifest = {
                "labels": settings()["labels"],
                "records": records,
                "grouping_basis": "known_identity",
                "augmentation": {"D": "acdc_four_view", "seed": 1001},
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            data = load_detection_stage(path, "0", 1, 32, manifest["labels"], True)
            self.assertEqual(len(data.D.batches), 1)
            self.assertIsNone(data.V.batches.view_policy)
            v = data.V.batches[0]["img"].clone()
            for e in range(4):
                epoch = shuffled_batches(data.D.batches, 23 + e, e, 1001)
                self.assertEqual(len(epoch.records), 1)
                self.assertEqual(epoch[0]["img"].shape, (1, 3, 32, 32))
                self.assertTrue(torch.equal(v, data.V.batches[0]["img"]))


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.recipe = {
            "updates": 3,
            "lr": 0.01,
            "validation_interval": 1,
            "teacher_momentum": 0.9,
            "pseudo_confidence": 0.8,
            "unlabeled_weight": 1.0,
        }
        self.batch = {"img": torch.ones(2, 1), "target": torch.ones(2, 1)}
        torch.manual_seed(4)
        self.model = torch.nn.Linear(1, 1, bias=False)

    def loss(self, m, b):
        return (m(b["img"]) - b["target"]).square().mean()

    def test_teacher_has_no_grad_and_empty_U_is_not_background(self):
        seen = []
        before = self.model.weight.detach().clone()

        def pseudo(teacher, b, threshold):
            self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
            seen.append(threshold)
            return None

        r = train_initialization(
            self.model,
            [self.batch],
            [self.batch],
            self.recipe,
            "C2",
            self.loss,
            pseudo,
            lambda m: 0.4,
        )
        self.assertEqual(r["U_image_reads"], 6)
        self.assertEqual(r["accepted_pseudo_boxes"], 0)
        self.assertEqual(r["best_update"], 1)
        self.assertEqual(len(seen), 3)
        self.assertFalse(torch.equal(before, self.model.weight))

    def test_supervised_control_has_identical_update_path_when_U_loss_zero(self):
        other = copy.deepcopy(self.model)
        r = {**self.recipe, "unlabeled_weight": 0.0}
        a = train_initialization(
            self.model,
            [self.batch],
            None,
            r,
            "C1",
            self.loss,
            None,
            lambda m: float(m.weight.detach().item()),
        )
        b = train_initialization(
            other,
            [self.batch],
            [self.batch],
            r,
            "C2",
            self.loss,
            lambda *a: None,
            lambda m: float(m.weight.detach().item()),
        )
        self.assertTrue(torch.equal(self.model.weight, other.weight))
        self.assertEqual(a["actual_optimizer_updates"], b["actual_optimizer_updates"])

    def test_missing_teacher_parameters_fail_before_updates(self):
        with self.assertRaises(ValueError):
            validate_recipe({**self.recipe, "pseudo_confidence": None}, "C2")

    def test_ema_uses_exact_moving_average(self):
        teacher = copy.deepcopy(self.model)
        teacher.weight.data.fill_(0)
        self.model.weight.data.fill_(2)
        ema_update(teacher, self.model, 0.75)
        self.assertAlmostEqual(float(teacher.weight.item()), 0.5)

    def test_pseudo_targets_produce_additional_loss(self):
        def pseudo(t, b, c):
            return {**b, "cls": torch.zeros(2, 1), "batch_idx": torch.tensor([0, 1])}

        r = train_initialization(
            self.model,
            [self.batch],
            [self.batch],
            self.recipe,
            "C2",
            self.loss,
            pseudo,
            lambda m: 0.3,
        )
        self.assertEqual(r["accepted_pseudo_boxes"], 6)
        self.assertTrue(all(x["unlabeled_loss"] > 0 for x in r["rows"]))


class RecordedAnalysisTests(unittest.TestCase):
    def rows(self):
        return [
            dict(
                dataset="ACDC",
                detector="v8n",
                seed=1001,
                client="0",
                stage=1,
                attempt=i,
                gain=g,
                prediction=0.1,
                kind="refine",
                retained=True,
            )
            for i, g in enumerate((0.2, 0.4, 0.1))
        ]

    def test_history_reference_uses_only_past_feedback(self):
        from softcas.paper_analysis import prediction_references

        r = prediction_references(self.rows(), "recorded_gain")["observations"]
        self.assertEqual(
            [x["predictions"]["history"] for x in r], [0.0, 0.2, 0.30000000000000004]
        )

    def test_missing_feedback_is_not_relabelled_zero(self):
        from softcas.paper_analysis import prediction_references

        rows = self.rows()
        rows[0]["gain"] = float("nan")
        with self.assertRaises(ValueError):
            prediction_references(rows, "recorded_gain")

    def test_stage_change_resets_history(self):
        from softcas.paper_analysis import prediction_references

        rows = self.rows()
        rows[-1]["stage"] = 2
        r = prediction_references(rows, "recorded_gain")["observations"]
        self.assertEqual(r[-1]["predictions"]["history"], 0.0)

    def test_action_shares_preserve_missing_retention(self):
        from softcas.paper_analysis import action_summary

        rows = self.rows()
        rows[0]["kind"] = "import"
        rows[0]["retained"] = None
        r = action_summary(rows)[0]
        self.assertEqual(r["actions"], 3)
        self.assertEqual(r["retention_observations"], 2)
        self.assertAlmostEqual(r["share_percent"]["Import"], 100 / 3)


if __name__ == "__main__":
    unittest.main()
