"""Regression tests for the five-risk audit; synthetic behavior, never detector AP."""

import copy
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_invariants import Fixture
from softcas.actions import Proposal
from softcas.controller import Archive, Scored, Ridge, Selector, make_encoder
from softcas.engine import run_stage, export_stage
from softcas.proposals import ProposalFactory
from softcas.resources import Cost, Ledger, ExecutionStopped
from softcas.state import FitSettings, Structure


class RuntimeRiskRepairs(Fixture):
    def test_entry_failure_retains_incumbent_and_exports_without_fake_score(self):
        def fail(*args, **kwargs):
            self.ledger.charge("partial_entry", Cost(F=19, W=0.01))
            raise RuntimeError("failed initial evaluation")

        with (
            patch.object(self.evaluator, "evaluate", side_effect=fail),
            patch.object(self.backend, "fit_batch") as fit,
            patch.object(self.backend, "qualify") as qualify,
        ):
            result = run_stage(
                self.backend,
                self.state,
                self.data,
                self.cache,
                self.exchange,
                self.ledger,
                self.c,
                evaluator=self.evaluator,
            )
        self.assertEqual(result.stop, "initialization_failed")
        self.assertEqual(result.deployed, self.state)
        self.assertEqual(
            (result.actions, result.predictor.n, result.archive), ([], 0, None)
        )
        self.assertNotIn("candidate_score", result.audit)
        self.assertEqual(self.ledger.used["search"].F, 19)
        fit.assert_not_called()
        qualify.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            export_stage(
                result,
                folder,
                self.backend,
                self.data,
                self.c,
                self.evaluator,
                self.cache,
                self.exchange,
            )
            self.assertEqual(
                json.loads(Path(folder, "stage.json").read_bytes())["stop"], result.stop
            )

    def test_known_imports_are_trimmed_before_provider_samples_parents(self):
        self.c["archive_capacity"] = 2
        low = self.state.lineage(saved="low", family="configuration")
        high = self.state.lineage(saved="high", family="collaboration")
        self.evaluator.scores[self.evaluator.key(self.backend, low, self.data.V)] = 0.8
        self.evaluator.scores[self.evaluator.key(self.backend, high, self.data.V)] = 0.9
        pools = []

        def provider(archive, attempt):
            pools.append(set(archive.items))
            self.assertIn(high.hash, archive.items)
            self.assertNotIn(low.hash, archive.items)
            return []

        with patch.object(self.evaluator, "evaluate", return_value=(0.5, False)):
            result = run_stage(
                self.backend,
                self.state,
                self.data,
                self.cache,
                self.exchange,
                self.ledger,
                self.c,
                retained=(low, high),
                evaluator=self.evaluator,
                proposal_provider=provider,
            )
        self.assertEqual(len(pools), 1)
        self.assertEqual(result.actions, [])
        self.assertEqual(result.champion, high)
        self.assertEqual(result.predictor.n, 0)

    def test_import_merge_failure_never_becomes_action_feedback(self):
        saved = self.state.lineage(saved="persist", family="configuration")
        self.evaluator.scores[self.evaluator.key(self.backend, saved, self.data.V)] = (
            0.9
        )
        with (
            patch.object(Archive, "insert", side_effect=OSError("full disk")),
            patch.object(self.evaluator, "evaluate", return_value=(0.5, False)),
        ):
            result = run_stage(
                self.backend,
                self.state,
                self.data,
                self.cache,
                self.exchange,
                self.ledger,
                self.c,
                retained=(saved,),
                evaluator=self.evaluator,
            )
        self.assertEqual(result.stop, "proposal_preparation_failed")
        self.assertEqual((result.actions, result.predictor.n), ([], 0))
        self.assertFalse(result.audit["released"])

    def test_partially_successful_import_merge_rolls_back_the_whole_merge(self):
        archive = Archive(self.state, 0.5, 3, self.cache)
        a = self.state.lineage(family="configuration", tag="a")
        b = self.state.lineage(family="collaboration", tag="b")
        insert = archive.insert
        calls = []

        def fail_second(scored):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("second checkpoint persistence failed")
            return insert(scored)

        with patch.object(archive, "insert", side_effect=fail_second):
            with self.assertRaises(OSError):
                archive.merge(
                    [
                        Scored(a, 0.6, 0.1, "configuration"),
                        Scored(b, 0.7, 0.2, "collaboration"),
                    ]
                )
        self.assertEqual(set(archive.items), {self.state.hash})

    def test_post_operation_search_overrun_blocks_release_and_retains_actual_cost(self):
        def overspend(archive, attempt):
            self.ledger.charge(
                "atomic_overspend", Cost(F=self.ledger.limits["search"].F + 1)
            )
            return []

        with patch.object(self.backend, "qualify") as qualify:
            result = run_stage(
                self.backend,
                self.state,
                self.data,
                self.cache,
                self.exchange,
                self.ledger,
                self.c,
                proposal_provider=overspend,
            )
        self.assertEqual(result.stop, "search_budget_violation")
        self.assertGreater(self.ledger.used["search"].F, self.ledger.limits["search"].F)
        self.assertEqual(result.deployed, self.state)
        self.assertEqual(result.published, ())
        qualify.assert_not_called()

    def test_feedback_reserve_can_prevent_an_otherwise_affordable_attempt(self):
        a = self.proposal("refine")
        remaining = self.ledger.remaining()
        a = replace(a, estimate=Cost(F=remaining.F - 20, E=1, W=1))
        with patch.object(self.backend, "fit_batch") as fit:
            result = run_stage(
                self.backend,
                self.state,
                self.data,
                self.cache,
                self.exchange,
                self.ledger,
                self.c,
                proposal_provider=lambda *_: [a],
            )
        self.assertEqual(result.actions, [])
        self.assertEqual(self.ledger.reservations, {})
        fit.assert_not_called()

    def test_ranking_uses_total_estimated_cost_and_gross_remaining_budget(self):
        a = self.proposal("refine", fidelity=1)
        self.c["attempts"] = 1
        gross = []

        def provider(*args):
            gross.append(self.ledger.limits["search"] - self.ledger.used["search"])
            return [a]

        result = run_stage(
            self.backend,
            self.state,
            self.data,
            self.cache,
            self.exchange,
            self.ledger,
            self.c,
            proposal_provider=provider,
        )
        record = result.actions[0]
        pool = record["candidate_pool"][0]
        self.assertEqual(pool["remaining"]["F"], gross[0].F)
        self.assertEqual(
            pool["estimate"]["F"], a.estimate.F + record["completion_reserve"]["F"]
        )
        self.assertGreater(record["completion_reserve"]["F"], 0)
        self.assertEqual(self.ledger.reservations, {})


class FeatureRiskRepairs(Fixture):
    def encoders(self, vocabulary=()):
        base = make_encoder(self.c["features"])
        extended = make_encoder(
            {
                **self.c["features"],
                "schema": "port_context_v2",
                "donor_vocabulary": list(vocabulary),
            }
        )
        return base, extended

    def encode(self, encoder, proposal):
        return encoder.encode(
            proposal,
            Scored(self.state, 0.5, 0, "incumbent"),
            self.ledger.remaining(),
            self.cache,
            self.exchange.catalogue,
        )

    def test_original_encoding_retained_and_candidate_port_lr_penalty_are_distinguished(
        self,
    ):
        base, extended = self.encoders()
        a = self.proposal("fresh", z=self.state.z, h=self.state.h)
        variants = [
            replace(a, z=Structure((0, 1, 0, 0, 0, 0), 1 / 8)),
            replace(a, h=FitSettings(1e-4, self.state.h.residual)),
            replace(a, h=FitSettings(self.state.h.lr, 1e-4)),
        ]
        self.assertEqual(len(base.fields), 25)
        for b in variants:
            self.assertEqual(self.encode(base, a), self.encode(base, b))
            self.assertNotEqual(self.encode(extended, a), self.encode(extended, b))

    def test_donor_identity_uses_frozen_vocabulary_and_unseen_count(self):
        a_key, b_key = self.put(self.package(seed=8)), self.put(self.package(seed=9))
        base, extended = self.encoders((a_key,))
        a, b = self.proposal("add", add=a_key), self.proposal("add", add=b_key)
        self.assertEqual(self.encode(base, a), self.encode(base, b))
        self.assertNotEqual(self.encode(extended, a), self.encode(extended, b))
        a_values = dict(zip(extended.fields, self.encode(extended, a)))
        b_values = dict(zip(extended.fields, self.encode(extended, b)))
        self.assertEqual(
            (a_values["donor_" + a_key], a_values["unknown_donors"]), (1, 0)
        )
        self.assertEqual(
            (b_values["donor_" + a_key], b_values["unknown_donors"]), (0, 1)
        )

    def test_extension_can_execute_through_real_stage_controller(self):
        self.c["features"] = {
            **self.c["features"],
            "schema": "port_context_v2",
            "donor_vocabulary": [],
        }
        self.c["attempts"] = 2
        result = run_stage(
            self.backend,
            self.state,
            self.data,
            self.cache,
            self.exchange,
            self.ledger,
            self.c,
        )
        self.assertEqual(len(result.actions), 2)
        self.assertTrue(all(len(a["features"]) == 46 for a in result.actions))
        self.assertTrue(all(a["status"] == "complete" for a in result.actions))


class StrictInterventions(Fixture):
    def test_strict_no_cross_retains_import_and_both_action_families(self):
        self.c["ablation"] = "no_handoff"
        self.c.update(
            proposal_protocol="native_r1000",
            fitting={
                "unit": "epochs",
                "stage_epochs": 2,
                "local_epochs": 1,
                "receiving_epochs": 1,
            },
            _fit_steps_remaining=2,
            _fit_batches_per_epoch=1,
            _attempt_index=0,
        )
        self.put(self.package())
        archive = Archive(self.state, 0.5, 8, self.cache)
        conf = self.state.lineage(family="configuration")
        collab = self.state.lineage(family="collaboration")
        archive.insert(Scored(conf, 0.6, 0.1, "configuration"))
        archive.insert(Scored(collab, 0.7, 0.2, "collaboration"))
        retained = self.state.lineage(family="collaboration", saved=True)
        factory = ProposalFactory(self.c, self.backend, self.exchange, (retained,))
        proposals = factory.generate(archive, self.ledger)
        self.assertTrue(any(a.kind == "import" for a in proposals))
        self.assertTrue(any(a.family == "configuration" for a in proposals))
        self.assertTrue(any(a.kind == "add" for a in proposals))
        for a in proposals:
            family = "collaboration" if a.kind == "import" else a.family
            self.assertTrue(
                a.parent == self.state.hash or archive.items[a.parent].family == family
            )
        self.assertEqual(
            retained.hash,
            next(a.imported.hash for a in proposals if a.kind == "import"),
        )

    def test_no_gain_changes_only_mean_in_same_priority_pool(self):
        archive = Archive(self.state, 0.5, 8, self.cache)
        encoder = make_encoder(self.c["features"])
        ridge = Ridge(25, 1)
        proposals = [
            self.proposal("refine"),
            self.proposal("fresh", z=self.state.z, h=self.state.h),
        ]
        f = encoder.encode(
            proposals[0], archive.champion, self.ledger.remaining(), self.cache, {}
        )
        ridge.update(f, 0.3)
        pools = []
        for no_gain in (False, True):
            selector = Selector(
                self.c["cold_order"], self.c["type_order"], 0.1, 0.001, no_gain=no_gain
            )
            selector.local_fresh_offered = True
            selector.cold_index = 4
            selector.choose(
                proposals,
                archive,
                ridge,
                encoder,
                self.ledger.remaining(),
                self.cache,
                {},
            )
            pools.append(selector.last_pool)
        for full, reduced in zip(*pools):
            for name in (
                "features",
                "prediction",
                "coverage",
                "parent",
                "estimate",
                "remaining",
            ):
                self.assertEqual(full[name], reduced[name])
        self.assertTrue(any(a["priority"] != b["priority"] for a, b in zip(*pools)))

    def test_reservations_are_not_spending_and_survive_no_hidden_refund(self):
        ledger = Ledger(Cost(F=10), Cost(), enforce=("F",))
        ledger.reserve("tail", Cost(F=3))
        self.assertEqual((ledger.remaining().F, ledger.used["search"].F), (7, 0))
        with self.assertRaises(ExecutionStopped):
            ledger.guard(Cost(F=8))
        ledger.charge("indivisible", Cost(F=11))
        ledger.unreserve("tail")
        self.assertEqual(ledger.used["search"].F, 11)
        self.assertEqual(ledger.violations()["search"], ["F"])
