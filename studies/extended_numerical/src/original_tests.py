from pathlib import Path
import sys, unittest, json, math, dataclasses
import numpy as np

ROOT = Path("data/CIRA_________20261105")
sys.path.insert(0, str(ROOT / "src"))
from controller import *

G = (1, 0, 0, 0, 0, 0, 1 / 8, 1e-3, 1e-3)


def state(v=0.0, family="entry", history=0, peers=(), logits=()):
    return State(
        G,
        Tensor.from_array(np.array([v], float)),
        peers,
        logits,
        history,
        family,
        (str(v),),
    )


def scored(v=0.0, ap=0.5, family="entry", context="D0"):
    return Scored(state(v, family), ap, context)


def ctrl(**kw):
    return Controller(
        scored(),
        (100, 65, 100, 100),
        (1, 12, 12, 2000, 64, 32 * 2**20, 1, 2, 0.125, 100, 65, 100, 100),
        (9001, "toy", 0, 1),
        **kw,
    )


def proposal(
    c,
    mode="Refine",
    typ="configuration",
    id="p",
    parent=None,
    costs=(1, 1, 0, 1),
    target=None,
):
    return Proposal(
        parent or c.archive.entry,
        typ,
        mode,
        id,
        costs,
        0 if mode == "Import" else 1,
        import_target=target,
    )


class Tests(unittest.TestCase):
    def test_immutable_parameters(self):
        x = state()
        self.assertFalse(x.phi.array().flags.writeable)

    def test_logits_exact_identity(self):
        a = Peer(0, "v1", Tensor.from_array(np.ones(2)))
        b = Peer(0, "v2", Tensor.from_array(np.zeros(2)))
        p = state(peers=(a,), logits=((0, "v1", 2.0),))
        x = initialize(p, "Add", peers=(a, b))
        self.assertEqual(x.logits, ((0, "v1", 2.0), (0, "v2", 0.0)))
        self.assertEqual(x.phi, p.phi)

    def test_softmax_denominator_changes(self):
        a = Peer(0, "v1", Tensor.from_array(np.ones(2)))
        b = Peer(0, "v2", Tensor.from_array(np.zeros(2)))
        p = state(peers=(a,), logits=((0, "v1", 0.0),))
        q = initialize(p, "Add", peers=(a, b))
        self.assertAlmostEqual(p.coefficients(0)[1], 0.5)
        self.assertAlmostEqual(q.coefficients(0)[1], 1 / 3)

    def test_fresh_retains_external_and_history(self):
        a = Peer(0, "v1", Tensor.from_array(np.ones(2)))
        p = state(2.0, history=55, peers=(a,), logits=((0, "v1", 3.0),))
        q = initialize(p, "Fresh", genes=G, fresh_phi=state(8.0).phi)
        self.assertEqual((q.peers, q.logits, q.history_steps), (p.peers, p.logits, 55))
        self.assertNotEqual(q.phi, p.phi)

    def test_import_is_exact(self):
        p = state()
        q = state(9.0, "configuration", 72)
        self.assertIs(initialize(p, "Import", imported=q), q)

    def test_member_capacity(self):
        peers = tuple(Peer(0, str(i), state(i).phi) for i in range(3))
        with self.assertRaises(AssertionError):
            state(peers=peers, logits=tuple((0, str(i), 0.0) for i in range(3)))

    def test_negative_complete_is_not_failure(self):
        c = ctrl()
        p = proposal(c)
        sel = c.choose([p])
        q = State(G, state(1.0).phi, history_steps=1, family="configuration")
        r = c.commit(sel, (1, 1, 0, 1), q, 0.4)
        self.assertEqual(r["status"], "complete")
        self.assertFalse(r["rollback"])
        self.assertAlmostEqual(r["signed_gain"], -0.1)
        self.assertIn(q.digest, c.archive.items)
        self.assertGreater(np.linalg.norm(c.b), 0)

    def test_failure_cost_and_zero_target(self):
        c = ctrl()
        r = c.commit(c.choose([proposal(c)]), (3, 0, 2, 4), status="interrupted")
        self.assertEqual(r["signed_gain"], 0)
        self.assertIsNone(r["child_AP"])
        self.assertEqual(c.remaining.tolist(), [97, 65, 98, 96])
        self.assertEqual(len(c.archive.items), 1)

    def test_overspend_not_hidden(self):
        c = ctrl()
        r = c.commit(c.choose([proposal(c)]), (101, 1, 0, 0), state(2), 0.9)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(c.remaining[0], -1)
        self.assertIsNone(c.choose([proposal(c)]))

    def test_illegal_collaboration_freeze(self):
        c = ctrl()
        p = proposal(c, "Remove", "cached")
        bad = State(G, state(2).phi, history_steps=1, family="collaboration")
        r = c.commit(c.choose([p]), (1, 1, 0, 1), bad, 0.9)
        self.assertEqual(r["status"], "failed")
        self.assertEqual(len(c.archive.items), 1)

    def test_archive_protects_entry_and_best(self):
        a = Archive(scored(), 8)
        for i in range(50):
            a.insert(scored(i + 1, 0.51 + i * 0.005, "configuration"))
        self.assertEqual(len(a.items), 8)
        self.assertIn(a.entry.state.digest, a.items)
        self.assertAlmostEqual(a.best().AP, 0.755)

    def test_archive_context(self):
        a = Archive(scored())
        with self.assertRaises(AssertionError):
            a.insert(scored(1, 0.9, context="D1"))

    def test_duplicate_preserves_family(self):
        a = Archive(scored())
        x = scored(1, 0.6, "configuration")
        a.insert(x)
        a.insert(x)
        self.assertEqual(a.items[x.state.digest].state.family, "configuration")

    def test_family_parent_eligibility(self):
        c = ctrl(restrict_families=True)
        x = scored(1, 0.8, "collaboration")
        c.archive.insert(x)
        p = proposal(c, parent=x)
        self.assertFalse(c.affordable(p))
        c.restrict_families = False
        self.assertTrue(c.affordable(p))

    def test_ridge_reset_stage(self):
        c = ctrl()
        c.b[:] = 4
        c2 = ctrl()
        self.assertTrue(np.array_equal(c2.A, np.eye(25)))
        self.assertEqual(c2.b.sum(), 0)

    def test_rank_exact_gap_and_remaining(self):
        c = ctrl()
        c.archive.insert(scored(1, 0.7, "configuration"))
        p = proposal(c)
        f = features(p, 0.5, c.remaining, c.scales)
        r = c.rank(p, f)
        self.assertAlmostEqual(r["gap"], 0.2)
        self.assertEqual(r["priority"], 0)

    def test_no_gain_preserves_coverage(self):
        c = ctrl(no_gain=True)
        c.b[:] = 1
        p = proposal(c)
        f = features(p, 0.5, c.remaining, c.scales)
        r = c.rank(p, f)
        self.assertEqual(r["used_mu"], 0)
        self.assertGreater(r["coverage"], 0)
        self.assertGreater(r["mu"], 0)

    def test_zero_budget_zero_cost(self):
        c = ctrl()
        c.remaining[2] = 0
        p = proposal(c, costs=(1, 1, 0, 1))
        self.assertTrue(
            math.isfinite(
                c.rank(p, features(p, 0.5, c.remaining, c.scales))["priority"]
            )
        )
        self.assertFalse(c.affordable(proposal(c, costs=(1, 1, 1, 1))))

    def test_keep_zero_priority(self):
        c = ctrl(kappa=0)
        c.cold_visit = 4
        self.assertIsNone(c.choose([proposal(c)]))

    def test_features_saved_before_feedback(self):
        c = ctrl()
        p = proposal(c)
        s = c.choose([p])
        self.assertEqual(s[1].shape, (25,))
        self.assertFalse(s[1].flags.writeable)

    def test_bounded_sampling_unique_K(self):
        c = ctrl()
        p = [proposal(c, id=str(i)) for i in range(20)]
        b = c.bounded(p + p)
        self.assertEqual(len(b), 3)
        self.assertEqual(len({x.identity for x in b}), 3)
        self.assertEqual(c.bounded(p), b)

    def test_cold_skip_does_not_resample(self):
        c = ctrl(kappa=0)
        p = proposal(c, mode="Remove", typ="cached")
        self.assertIsNotNone(c.choose([p]))
        self.assertGreaterEqual(c.cold_visit, 2)
        self.assertIsNone(c.choose([proposal(c, mode="Fresh")]))

    def test_tournament_ties_and_seed(self):
        a = Archive(scored())
        a.insert(scored(1, 0.5, "configuration"))
        x = a.tournament(named_rng((9001,), "t"))
        y = a.tournament(named_rng((9001,), "t"))
        self.assertEqual(x, y)

    def test_mutation_repair_property(self):
        for i in range(100):
            g = crossover(G, G, named_rng((9001,), "g", i))
            self.assertTrue(1 <= sum(g[:6]) <= 2)
            self.assertTrue(all(v in a for v, a in zip(g, ALLELES)))

    def test_archive_increment_property(self):
        a = Archive(scored())
        rng = named_rng((9001,), "property")
        for i in range(200):
            p = a.tournament(rng)
            ap = float(rng.random())
            old = a.best().AP
            gain = ap - p.AP
            gap = old - p.AP
            a.insert(scored(i + 1, ap, "configuration"))
            self.assertAlmostEqual(a.best().AP - old, max(0, gain - gap))

    def test_locked_audit_no_second_best_search(self):
        c = ctrl()
        x = scored(1, 0.9, "configuration")
        c.archive.insert(x)
        calls = []

        def audit(s):
            calls.append(s.digest)
            return 0.8 if s == c.archive.entry.state else 0.7

        released, log = locked_release(c, audit)
        self.assertEqual(released, c.archive.entry)
        self.assertEqual(len(calls), 2)
        self.assertFalse(log["accepted"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    proof = dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        property_cases=300,
        scope="reference-controller software verification only; no new AP performance experiment",
        native_detector_verified=False,
        old_5502_trajectories_reclassified=False,
    )
    (ROOT / "qa/tests.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
