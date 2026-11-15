import copy
from pathlib import Path
from test_invariants import Fixture
from softcas.config import load_config, validate_config
from softcas.fitting import stage_steps
from softcas.controller import Archive
from softcas.proposals import ProposalFactory


class R2000ContractTests(Fixture):
    def test_r2000_stage_allowance_and_actual_loader_size(self):
        c = load_config(
            Path(__file__).parents[1] / "configs/native_r2000.development.json"
        )
        self.assertEqual(
            c["round_budget"], dict(per_client=2000, stages=4, epochs_per_round=2)
        )
        self.assertEqual(stage_steps(c, 5), 5000)
        self.assertEqual(stage_steps(c, 28), 28000)
        self.assertEqual(4 * 4 * stage_steps(c, 5), 80000)
        self.assertEqual(4 * 4 * stage_steps(c, 28), 448000)
        with self.assertRaises(ValueError):
            load_config(
                Path(__file__).parents[1] / "configs/native_r2000.development.json",
                formal=True,
            )

    def test_stale_stage_budget_is_rejected(self):
        c = load_config(
            Path(__file__).parents[1] / "configs/native_r2000.development.json"
        )
        c["fitting"]["stage_epochs"] = 500
        with self.assertRaisesRegex(ValueError, "allowance differ"):
            validate_config(c)

    def test_serial_reverse_changes_family_not_seed(self):
        families = []
        for order in [
            ["configuration", "collaboration"],
            ["collaboration", "configuration"],
        ]:
            c = copy.deepcopy(self.c)
            c.update(
                method="serial",
                proposal_protocol="native_r1000",
                fitting=dict(
                    unit="epochs", stage_epochs=2, local_epochs=1, receiving_epochs=1
                ),
                _fit_batches_per_epoch=1,
                _fit_steps_remaining=2,
                serial=dict(order=order, fraction=0.5, attempt_fraction=0.5),
            )
            factory = ProposalFactory(c, self.backend, self.exchange)
            archive = Archive(self.state, 0.5, 8, self.cache)
            first = factory.serial_family(archive, self.ledger)
            c["_fit_steps_remaining"] = 1
            second = factory.serial_family(archive, self.ledger)
            families.append((first, second))
        self.assertEqual(
            families,
            [("configuration", "collaboration"), ("collaboration", "configuration")],
        )
