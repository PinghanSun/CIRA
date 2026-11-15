"""Controls for confounding and role separation; all inputs here are fixtures."""

import copy
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_invariants import Fixture
from softcas.data import synthetic_split
from softcas.mechanism import (
    fixed_receiving_mass,
    receiving_comparison,
    configuration_composition_comparison,
)
from softcas.state import State, canonical
from softcas.tail_calibration import calibrate_tail


class FixedMassDiagnostics(Fixture):
    def setup_snapshot(self):
        keys = (
            self.put(self.package(port=2, seed=7)),
            self.put(self.package(port=2, seed=8)),
        )
        self.parent = replace(
            self.state,
            C=tuple(sorted(keys)),
            beta=tuple((k, 0.3) for k in sorted(keys)),
        )
        self.T = synthetic_split(
            "T", self.data.D.client, self.data.D.stage, self.data.D.domain, 73
        )
        self.spec = {
            "snapshot_time": 3,
            "fit_epochs": 1,
            "initialization_seed": 50,
            "max_onehot_candidates": 12,
            "total_peer_mass": 0.35,
            "budget": {"F": 1e12, "E": 100, "B": 0, "W": 120},
            "configurations": [
                {
                    "mask": [1, 0, 0, 0, 0, 0],
                    "ratio": 0.125,
                    "lr": 0.001,
                    "residual": 0.001,
                },
                {
                    "mask": [1, 1, 0, 0, 0, 0],
                    "ratio": 0.125,
                    "lr": 0.001,
                    "residual": 0.001,
                },
            ],
        }

    def test_fixed_mass_does_not_depend_on_active_mask_or_member_count(self):
        self.setup_snapshot()
        for parent in (
            self.parent,
            replace(self.parent, C=self.parent.C[:1], beta=self.parent.beta[:1]),
        ):
            state = fixed_receiving_mass(parent, self.cache, 0.35)
            model = self.backend.assembly(state, self.cache, mode="import")
            weights = model.ports["2"].weights()[1]
            self.assertAlmostEqual(float(weights[1:].sum()), 0.35, places=6)
            self.assertEqual(state.phi, parent.phi)
            self.assertEqual(state.C, parent.C)
        with self.assertRaises(ValueError):
            fixed_receiving_mass(self.parent, self.cache, 1)

    def test_four_cells_fit_phi_only_and_lock_choices_before_held_out_readout(self):
        self.setup_snapshot()
        original_score, original_fit = self.backend.score, self.backend.fit_batch
        roles, fit_flags = [], []
        with tempfile.TemporaryDirectory() as tmp:

            def score(model, split):
                roles.append(split.role)
                if split.role == "T":
                    self.assertFalse(
                        json.loads(Path(tmp, "V_choices.json").read_bytes())["T_read"]
                    )
                return original_score(model, split)

            def fit(model, *args, **kwargs):
                fit_flags.append(
                    [
                        b.requires_grad
                        for p in model.ports.values()
                        for b in p.beta.values()
                    ]
                )
                return original_fit(model, *args, **kwargs)

            with (
                patch.object(self.backend, "score", side_effect=score),
                patch.object(self.backend, "fit_batch", side_effect=fit),
            ):
                result = configuration_composition_comparison(
                    self.backend,
                    self.cache,
                    self.data.D,
                    self.data.V,
                    self.T,
                    self.parent,
                    self.spec,
                    self.c,
                    tmp,
                )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["rows"]), 4)
            self.assertTrue(
                all(
                    r["completed_fit_steps"] == len(self.data.D.batches)
                    for r in result["rows"]
                )
            )
            self.assertTrue(all(not any(flags) for flags in fit_flags))
            self.assertEqual(roles, ["V"] * 4 + ["T"] * 4)
            self.assertFalse(result["contract"]["mask_dependent_peer_rescaling"])
            contrasts = result["T_contrast"]
            self.assertEqual(
                contrasts["strict_reversal"],
                contrasts["empty"] * contrasts["peers"] < 0,
            )
            for row in result["rows"]:
                if row["context"] == "peers":
                    state = State.load(
                        Path(tmp, f"config_{row['configuration']}_peers.json")
                    )
                    model = self.backend.assembly(state, self.cache, mode="import")
                    self.assertAlmostEqual(
                        float(model.ports["2"].weights()[1][1:].sum()), 0.35, places=6
                    )

    def test_receiving_grid_is_prespecified_and_does_not_fit_local_parameters(self):
        self.setup_snapshot()
        grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        spec = {**self.spec, "coefficient_grid": grid}
        with tempfile.TemporaryDirectory() as tmp:
            result = receiving_comparison(
                self.backend,
                self.cache,
                self.data.D,
                self.data.V,
                self.T,
                self.parent,
                spec,
                self.c,
                tmp,
            )
            self.assertEqual(result["status"], "completed")
            rows = [r for r in result["rows"] if r["branch"].startswith("fixed_mass_")]
            self.assertEqual([r["total_peer_mass"] for r in rows], grid)
            self.assertTrue(all(r["completed_fit_steps"] == 0 for r in rows))
            for row in rows:
                state = State.load(Path(tmp, row["branch"] + ".json"))
                self.assertEqual(state.phi, self.parent.phi)
                self.assertEqual(state.C, self.parent.C)
            fitted = State.load(Path(tmp, "fitted.json"))
            self.assertEqual(fitted.phi, self.parent.phi)

    def test_failed_cells_keep_cost_and_missing_held_out_scores(self):
        self.setup_snapshot()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                self.backend, "fit_batch", side_effect=RuntimeError("interrupted")
            ),
        ):
            result = configuration_composition_comparison(
                self.backend,
                self.cache,
                self.data.D,
                self.data.V,
                self.T,
                self.parent,
                self.spec,
                self.c,
                tmp,
            )
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all("T" not in row for row in result["rows"]))
        self.assertTrue(all(row["actual"]["W"] > 0 for row in result["rows"]))
        self.assertNotIn("held_out_choice_effect", result)


class TailCalibrationChecks(Fixture):
    def write_stage(self, root, name, closing, seed=9001, runtime="fixture_cpu"):
        directory = Path(root, name)
        directory.mkdir()
        config = {
            **self.c,
            "purpose": "DEVELOPMENT",
            "seed": seed,
            "runtime": {"device": runtime},
        }
        Path(directory, "stage.json").write_bytes(
            canonical(
                {
                    "config": config,
                    "base": self.backend.identity,
                    "runtime_identity": {
                        "device": runtime,
                        "source": "UNIT_TEST_FIXTURE",
                    },
                    "execution": self.backend.execution,
                    "stop": "attempt_limit",
                    "violations": {"search": [], "release": []},
                }
            )
        )
        rows = [
            {"scope": "search", "reason": "predictor_feedback", "cost": {"W": w}}
            for w in (0.01, 0.02, 0.03)
        ]
        rows += [
            {
                "scope": "search",
                "reason": "stage_search_bookkeeping",
                "cost": {"W": closing},
            },
            {"scope": "search", "reason": "wall_clock_unbilled", "cost": {"W": 0.01}},
        ]
        Path(directory, "ledger.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        return directory

    def test_tail_envelope_uses_worst_complete_stage_and_keeps_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages = [
                self.write_stage(tmp, "a", 0.04),
                self.write_stage(tmp, "b", 0.09),
            ]
            result = calibrate_tail(stages, Path(tmp, "tail.json"), 2)
        self.assertAlmostEqual(result["estimates"]["feedback_W"], 0.06)
        self.assertAlmostEqual(result["estimates"]["search_close_W"], 0.2)
        self.assertFalse(result["hard_upper_bound"])
        self.assertEqual(result["status"], "unreviewed_development_envelope")
        self.assertEqual(len(result["inputs"]), 2)

    def test_evaluation_seed_and_mixed_runtime_cannot_calibrate_formal_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluated = self.write_stage(tmp, "eval", 0.04, seed=1001)
            with self.assertRaisesRegex(ValueError, "DEV"):
                calibrate_tail([evaluated], Path(tmp, "bad.json"), 2)
            a, b = (
                self.write_stage(tmp, "a", 0.04),
                self.write_stage(tmp, "b", 0.04, runtime="another_device"),
            )
            with self.assertRaisesRegex(ValueError, "Do not pool"):
                calibrate_tail([a, b], Path(tmp, "bad.json"), 2)
            self.assertFalse(Path(tmp, "bad.json").exists())
