"""Snapshot mechanisms with explicit state interventions and held-out readout."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from .backend import work
from .fitting import shuffled_batches
from .resources import Cost, Ledger
from .state import State, Package, canonical, digest
from .storage import atomic_write


CONTINUATION_FIELDS = {
    "complete_state": {
        "z": "retained",
        "h": "retained",
        "C": "retained",
        "phi": "retained",
        "beta": "retained",
    },
    "reset_receiving": {
        "z": "retained",
        "h": "retained",
        "C": "retained",
        "phi": "retained",
        "beta": "zero logits",
    },
    "cold_local": {
        "z": "retained",
        "h": "retained",
        "C": "retained",
        "phi": "reinitialized",
        "beta": "zero logits",
    },
}


def _positive_int(value, name, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (0 if zero else 1)
    ):
        raise ValueError(
            name
            + " must be an explicit "
            + ("nonnegative" if zero else "positive")
            + " integer"
        )
    return value


def _finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _output_directory(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Mechanism output must be new or empty; existing results are preserved"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _roles(D, V, T):
    if (D.role, V.role, T.role) != ("D", "V", "T"):
        raise ValueError("Role-separated D/V/T required")
    if len({(str(s.client), s.stage, s.domain) for s in (D, V, T)}) != 1:
        raise ValueError(
            "Snapshot D/V/T must belong to the same receiver, stage, and current domain"
        )
    if any(not len(s.batches) for s in (D, V, T)):
        raise ValueError("Empty diagnostic split")


def _snapshot(parent, cache, spec, backend):
    if json.loads(parent.nu).get("receiving_rule", "soft") != "soft":
        raise ValueError(
            "These snapshot contrasts require a soft-receiving parent, not a discrete-ablation checkpoint"
        )
    snapshot_time = _finite(spec["snapshot_time"], "snapshot_time")
    if snapshot_time < 0:
        raise ValueError("snapshot_time must not be negative")
    if spec.get("parent_sha256", parent.hash) != parent.hash:
        raise ValueError(
            "Snapshot parent identity differs from the registered checkpoint"
        )
    counts = {}
    for key in parent.C:
        package = cache.get(key)
        if package.created > snapshot_time or package.available > snapshot_time:
            raise ValueError(
                "External version was unavailable at the registered snapshot"
            )
        counts[package.interface.port] = counts.get(package.interface.port, 0) + 1
    if any(n > 2 for n in counts.values()) or len(parent.C) > spec.get(
        "max_external_paths", 12
    ):
        raise ValueError("Snapshot exceeds declared external path capacity")


def _ledger(backend, algorithm, spec):
    limits = spec.get("budget", algorithm["budgets"]["search"])
    if set(limits) != {"F", "E", "B", "W"} or any(
        _finite(v, "budget") < 0 for v in limits.values()
    ):
        raise ValueError("Diagnostic needs explicit nonnegative F/E/B/W allowances")
    ledger = Ledger(Cost(**limits), Cost())
    ledger.runtime = getattr(backend, "runtime", None)
    ledger.meter_policy = algorithm.get("meter")
    return ledger


def _soft(parent):
    return parent.lineage(receiving_rule="soft", discrete_choices={})


def fixed_receiving_mass(parent, cache, total_mass):
    """Keep the total peer coefficient fixed at each occupied port."""
    total_mass = _finite(total_mass, "total_peer_mass")
    if not 0 < total_mass < 1:
        raise ValueError("total_peer_mass must lie strictly between zero and one")
    ports = {}
    for key in parent.C:
        port = cache.get(key).interface.port
        ports.setdefault(port, []).append(key)
    logits = {
        key: math.log(total_mass / (len(keys) * (1 - total_mass)))
        for keys in ports.values()
        for key in keys
    }
    return replace(_soft(parent), beta=tuple(sorted(logits.items())))


def _continuation_initial(parent, backend, cache, seed, branch):
    complete = _soft(parent)
    if branch == "complete_state":
        state = complete
    elif branch == "reset_receiving":
        state = replace(complete, beta=tuple((key, 0.0) for key in complete.C))
    elif branch == "cold_local":
        reset = replace(complete, beta=tuple((key, 0.0) for key in complete.C))
        state = backend.assembly(reset, cache, seed, "fresh").snapshot(reset)
    elif branch == "reset_local_only":
        state = backend.assembly(complete, cache, seed, "fresh").snapshot(complete)
        state = replace(state, beta=complete.beta)
    else:
        raise ValueError("Unregistered continuation branch")
    state = state.lineage(
        diagnostic="continuation", intervention=branch, source=parent.hash
    )
    return (
        state.lineage(initialization_seed=seed)
        if branch in ("cold_local", "reset_local_only")
        else state
    )


def continuation_states(parent, backend, cache, seed):
    """Native complete-state interventions; donor membership is held fixed."""
    _positive_int(seed, "initialization_seed", zero=True)
    return {
        name: _continuation_initial(parent, backend, cache, seed, name)
        for name in CONTINUATION_FIELDS
    }


def _fit(
    backend, cache, state, D, epochs, seed, algorithm, ledger, mode, fit_receiving=True
):
    """Return actual attempted/completed minibatches and partial-state evidence."""
    attempted, completed, losses, epoch_seeds = 0, 0, [], []
    model = None
    try:
        with work(ledger, "mechanism_construct"):
            backend.before_candidate(state, cache)
            model = backend.assembly(state, cache, seed, mode)
            if not fit_receiving:
                for port in model.ports.values():
                    port.beta.requires_grad_(False)
            optimizer = backend.optimizer(model, state, algorithm) if epochs else None
            scaler = backend.new_scaler()
        if optimizer is not None:
            for epoch in range(epochs):
                epoch_value = int(
                    digest(
                        {
                            "seed": seed,
                            "receiver": str(D.client),
                            "stage": D.stage,
                            "domain": D.domain,
                            "epoch": epoch,
                            "split": D.identity,
                        }
                    )[:16],
                    16,
                )
                epoch_seeds.append(epoch_value)
                for batch in shuffled_batches(D.batches, epoch_value, epoch, seed):
                    with work(ledger, "mechanism_fit"):
                        optimizer.zero_grad(set_to_none=True)
                        attempted += 1
                        losses.append(
                            backend.fit_batch(
                                model, batch, state.h.residual, optimizer, scaler
                            )
                        )
                        completed += 1
        state = model.snapshot(state)
        return (
            state,
            model,
            {
                "status": "completed",
                "attempted_fit_steps": attempted,
                "completed_fit_steps": completed,
                "epochs_requested": epochs,
                "epochs_completed": completed / len(D.batches),
                "fit_losses": losses,
                "epoch_seeds": epoch_seeds,
                "optimizer_created": optimizer is not None,
            },
        )
    except Exception as exc:
        if model is not None:
            state = model.snapshot(state)
        return (
            state,
            model,
            {
                "status": "failed",
                "error": type(exc).__name__ + ": " + str(exc),
                "attempted_fit_steps": attempted,
                "completed_fit_steps": completed,
                "epochs_requested": epochs,
                "epochs_completed": completed / len(D.batches),
                "fit_losses": losses,
                "epoch_seeds": epoch_seeds,
                "optimizer_created": model is not None
                and "optimizer" in locals()
                and optimizer is not None,
            },
        )


def _score(backend, model, split, ledger, reason, selection=False):
    with work(ledger, reason, selection=selection):
        return _finite(backend.score(model, split), "diagnostic score")


def _finish_row(row, ledger):
    row["actual"] = asdict(ledger.used["search"])
    row["ledger"] = ledger.events
    row["budget_violations"] = ledger.violations()["search"]
    if row["budget_violations"] and row["status"] == "completed":
        row["status"] = "budget_exceeded"
    return row


def _save_result(result, output):
    atomic_write(output / "mechanism.json", canonical(result))
    return result


def _base_result(kind, backend, parent, D, V, T, algorithm, spec):
    return {
        "kind": kind,
        "purpose": algorithm.get("purpose"),
        "metric": backend.metric,
        "base": backend.identity,
        "parent": parent.hash,
        "data": {s.role: s.identity for s in (D, V, T)},
        "spec": spec,
        "independent_unit": "federation seed; receiver snapshots are nested observations",
        "optimizer_policy": "new optimizer for every fitting branch; no optimizer buffers exist in native State",
        "rows": [],
    }


def receiving_comparison(backend, cache, D, V, T, parent, spec, algorithm, output):
    """Uniform/Fitted soft receiving and exhaustive V-selected member/null one-hot."""
    _roles(D, V, T)
    _snapshot(parent, cache, spec, backend)
    epochs = _positive_int(spec["fit_epochs"], "fit_epochs", zero=True)
    seed = _positive_int(spec["initialization_seed"], "initialization_seed", zero=True)
    cap = _positive_int(spec["max_onehot_candidates"], "max_onehot_candidates")
    grid = tuple(
        _finite(v, "coefficient_grid") for v in spec.get("coefficient_grid", ())
    )
    if len(set(grid)) != len(grid) or any(not 0 < v < 1 for v in grid):
        raise ValueError(
            "coefficient_grid requires distinct prespecified masses in (0,1)"
        )
    choices = [
        (None, *sorted(key for key in parent.C if cache.get(key).interface.port == p))
        for p in range(6)
    ]
    count = math.prod(len(values) for values in choices)
    if count > cap:
        raise ValueError(
            f"Exhaustive one-hot diagnostic requires {count} candidates; registered cap is {cap}. No subsampling performed"
        )
    output = _output_directory(output)
    uniform = replace(_soft(parent), beta=tuple((key, 0.0) for key in parent.C))
    result = _base_result(
        "receiving_snapshot", backend, parent, D, V, T, algorithm, spec
    )
    result["contract"] = {
        "fixed_fields": ["z", "h", "C", "phi"],
        "uniform_fit_epochs": 0,
        "fitted_data": "D",
        "onehot_selection_data": "V",
        "readout": "T after V_choices.json",
        "onehot_candidates": count,
        "same_compute_claim": False,
        "empty_membership": not bool(parent.C),
        "max_onehot_candidates": cap,
    }
    result["contract"]["coefficient_grid"] = list(grid)
    result["contract"]["grid_interpretation"] = (
        "fixed total peer mass per occupied port; no fitting or T-based selection"
    )
    prepared = []
    receiving_branches = (
        ("uniform", "fitted", "frozen_parent_logits")
        if spec.get("include_frozen_parent_logits", False)
        else ("uniform", "fitted")
    )
    for name in receiving_branches:
        ledger = _ledger(backend, algorithm, spec)
        row = {"branch": name, "source_state": parent.hash}
        initial = parent if name == "frozen_parent_logits" else uniform
        state, model, fitted = _fit(
            backend,
            cache,
            initial,
            D,
            epochs if name == "fitted" else 0,
            seed,
            algorithm,
            ledger,
            "collaboration",
        )
        row.update(fitted)
        if (
            state.phi != parent.phi
            or state.C != parent.C
            or state.z != parent.z
            or state.h != parent.h
        ):
            raise RuntimeError("Receiving diagnostic changed a frozen local field")
        if row["status"] == "completed":
            try:
                row["V"] = _score(backend, model, V, ledger, "mechanism_V", True)
            except Exception as exc:
                row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        state = state.lineage(
            diagnostic="receiving", intervention=name, source=parent.hash
        )
        state.save(output / (name + ".json"))
        row["state"] = state.hash
        row["beta"] = dict(state.beta)
        prepared.append((row, state, ledger))
    for mass in grid:
        ledger = _ledger(backend, algorithm, spec)
        state = fixed_receiving_mass(parent, cache, mass)
        row = {
            "branch": f"fixed_mass_{mass:g}",
            "total_peer_mass": mass,
            "source_state": parent.hash,
            "status": "completed",
            "attempted_fit_steps": 0,
            "completed_fit_steps": 0,
            "optimizer_created": False,
        }
        try:
            with work(ledger, "mechanism_grid_construct"):
                model = backend.assembly(state, cache, mode="import")
            row["V"] = _score(backend, model, V, ledger, "mechanism_grid_V", True)
        except Exception as exc:
            row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        state.save(output / (row["branch"] + ".json"))
        row.update(state=state.hash, beta=dict(state.beta))
        prepared.append((row, state, ledger))
    ledger = _ledger(backend, algorithm, spec)
    onehot_row = {
        "branch": "onehot_V_selected",
        "status": "completed",
        "attempted_fit_steps": 0,
        "completed_fit_steps": 0,
        "epochs_completed": 0,
        "optimizer_created": False,
        "V_candidates": [],
        "source_state": parent.hash,
    }
    best, best_value = None, None
    try:
        for index, combination in enumerate(itertools.product(*choices)):
            state = uniform.lineage(
                receiving_rule="discrete",
                discrete_choices={str(p): key for p, key in enumerate(combination)},
            )
            with work(ledger, "mechanism_onehot_construct"):
                model = backend.assembly(state, cache, mode="import")
            value = _score(backend, model, V, ledger, "mechanism_onehot_V", True)
            onehot_row["V_candidates"].append(
                {"index": index, "choices": list(combination), "V": value}
            )

            if best is None or value > best_value:
                best, best_value = state, value
        onehot_row.update(V=best_value, state=best.hash)
        best.save(output / "onehot_V_selected.json")
    except Exception as exc:
        onehot_row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    prepared.append((onehot_row, best, ledger))
    lock = {
        "selection_role": "V",
        "T_read": False,
        "branches": [
            {k: row[k] for k in ("branch", "status", "state", "V") if k in row}
            for row, _, _ in prepared
        ],
        "onehot_tie": "null first, then sorted package hashes in port order",
    }
    atomic_write(output / "V_choices.json", canonical(lock))
    for row, state, ledger in prepared:
        if row["status"] == "completed" and state is not None:
            try:
                with work(ledger, "mechanism_T_construct"):
                    model = backend.assembly(state, cache, mode="import")
                row["T"] = _score(backend, model, T, ledger, "mechanism_independent_T")
            except Exception as exc:
                row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        result["rows"].append(_finish_row(row, ledger))
    result["status"] = (
        "completed"
        if all(row["status"] == "completed" for row in result["rows"])
        else "incomplete"
    )
    return _save_result(result, output)


def configuration_composition_comparison(
    backend, cache, D, V, T, parent, spec, algorithm, output
):
    """Two configurations x empty/fixed-peer context, with local-only D fitting."""
    from .state import Structure, FitSettings

    _roles(D, V, T)
    _snapshot(parent, cache, spec, backend)
    if not parent.C:
        raise ValueError("Composition contrast needs a fixed nonempty donor set")
    configs = spec["configurations"]
    if len(configs) != 2:
        raise ValueError(
            "Register exactly two configurations before the 2x2 diagnostic"
        )
    genes = [
        (Structure(tuple(c["mask"]), c["ratio"]), FitSettings(c["lr"], c["residual"]))
        for c in configs
    ]
    if genes[0] == genes[1] or any(not any(z.mask) for z, _ in genes):
        raise ValueError("Two different, nonempty local configurations required")
    epochs = _positive_int(spec["fit_epochs"], "fit_epochs", zero=True)
    seed = _positive_int(spec["initialization_seed"], "initialization_seed", zero=True)
    peer = fixed_receiving_mass(parent, cache, spec["total_peer_mass"])
    output = _output_directory(output)
    result = _base_result(
        "configuration_composition_fixed_mass",
        backend,
        parent,
        D,
        V,
        T,
        algorithm,
        spec,
    )
    result["contract"] = {
        "fixed_versions": list(parent.C),
        "fitted_data": "D",
        "fitted_fields": "phi only",
        "total_peer_mass_per_port": spec["total_peer_mass"],
        "mask_dependent_peer_rescaling": False,
        "selection_role": "V",
        "readout": "T after V_choices.json",
        "same_minibatch_schedule": True,
        "equal_executed_FLOPs_assumed": False,
        "contrasts": "configuration 1 minus configuration 0",
    }
    prepared = []
    for index, (z, h) in enumerate(genes):
        for context in ("empty", "peers"):
            ledger = _ledger(backend, algorithm, spec)
            cell = replace(peer, z=z, h=h, phi=())
            if context == "empty":
                cell = replace(cell, C=(), beta=())
            state, model, row = _fit(
                backend,
                cache,
                cell,
                D,
                epochs,
                seed,
                algorithm,
                ledger,
                "fresh",
                fit_receiving=False,
            )
            row.update(
                configuration=index,
                context=context,
                state=state.hash,
                beta=dict(state.beta),
            )
            if (
                state.C != cell.C
                or tuple(k for k, _ in state.beta) != tuple(k for k, _ in cell.beta)
                or any(
                    not math.isclose(v, dict(cell.beta)[k], rel_tol=1e-6, abs_tol=1e-7)
                    for k, v in state.beta
                )
            ):
                raise RuntimeError(
                    "Fixed composition changed during local-only fitting"
                )
            if row["status"] == "completed":
                try:
                    row["V"] = _score(backend, model, V, ledger, "mechanism_V", True)
                except Exception as exc:
                    row.update(
                        status="failed", error=type(exc).__name__ + ": " + str(exc)
                    )
            state.save(output / f"config_{index}_{context}.json")
            prepared.append((row, state, ledger))
    choices = {}
    for context in ("empty", "peers"):
        valid = [
            r
            for r, _, ledger in prepared
            if r["context"] == context
            and r["status"] == "completed"
            and not ledger.violations()["search"]
        ]
        if len(valid) == 2:
            choices[context] = max(valid, key=lambda r: (r["V"], -r["configuration"]))[
                "configuration"
            ]
    atomic_write(
        output / "V_choices.json",
        canonical(
            {
                "T_read": False,
                "selection_role": "V",
                "choices": choices,
                "tie": "configuration 0",
            }
        ),
    )
    for row, state, ledger in prepared:
        if row["status"] == "completed" and not ledger.violations()["search"]:
            try:
                with work(ledger, "mechanism_T_construct"):
                    model = backend.assembly(state, cache, mode="import")
                row["T"] = _score(backend, model, T, ledger, "mechanism_independent_T")
            except Exception as exc:
                row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        result["rows"].append(_finish_row(row, ledger))
    result["status"] = (
        "completed"
        if all(r["status"] == "completed" for r in result["rows"])
        else "incomplete"
    )
    result["V_choices"] = choices
    if result["status"] == "completed":
        rows = {(r["context"], r["configuration"]): r for r in result["rows"]}
        for role in ("V", "T"):
            contrasts = {
                c: rows[(c, 1)][role] - rows[(c, 0)][role] for c in ("empty", "peers")
            }
            result[role + "_contrast"] = {
                **contrasts,
                "interaction": contrasts["peers"] - contrasts["empty"],
                "strict_reversal": contrasts["peers"] * contrasts["empty"] < 0,
            }
        result["held_out_choice_effect"] = (
            rows[("peers", choices["peers"])]["T"]
            - rows[("peers", choices["empty"])]["T"]
        )
    return _save_result(result, output)


def continuation_comparison(backend, cache, D, V, T, parent, spec, algorithm, output):
    """Fit three native checkpoint interventions at the same minibatch schedule."""
    _roles(D, V, T)
    _snapshot(parent, cache, spec, backend)
    epochs = _positive_int(spec["fit_epochs"], "fit_epochs", zero=True)
    seed = _positive_int(spec["initialization_seed"], "initialization_seed", zero=True)
    fit_fields = spec.get("fit_fields", "phi_beta")
    if fit_fields not in ("phi", "phi_beta"):
        raise ValueError("Continuation fit_fields must be phi or phi_beta")
    output = _output_directory(output)
    result = _base_result(
        "continuation_snapshot", backend, parent, D, V, T, algorithm, spec
    )
    interventions = dict(CONTINUATION_FIELDS)
    if spec.get("include_reset_local_only", False):
        interventions["reset_local_only"] = {
            "z": "retained",
            "h": "retained",
            "C": "retained",
            "phi": "reinitialized",
            "beta": "retained",
        }
    result["contract"] = {
        "interventions": interventions,
        "fitted_fields": fit_fields,
        "same_epochs_and_minibatches": True,
        "equal_executed_FLOPs_assumed": False,
        "optimizer": "new optimizer in every branch",
        "historical_difference": "The older feature-module WeightsOnly contrast reset momentum. Native State stores no optimizer buffers; reset_receiving here resets beta, and is a different experiment.",
    }
    prepared, branches = [], {}
    for name in interventions:
        ledger = _ledger(backend, algorithm, spec)
        try:
            with work(ledger, "mechanism_" + name + "_initialize"):
                initial = _continuation_initial(parent, backend, cache, seed, name)
            branches[name] = initial
        except Exception as exc:
            row = {
                "branch": name,
                "source_state": parent.hash,
                "status": "failed",
                "error": type(exc).__name__ + ": " + str(exc),
                "failure_phase": "initialization",
                "attempted_fit_steps": 0,
                "completed_fit_steps": 0,
                "epochs_requested": epochs,
                "epochs_completed": 0,
                "optimizer_created": False,
            }
            prepared.append((row, parent, ledger))
            continue
        initial.save(output / (name + "_initial.json"))
        state, model, row = _fit(
            backend,
            cache,
            initial,
            D,
            epochs,
            seed,
            algorithm,
            ledger,
            "refine",
            fit_receiving=fit_fields == "phi_beta",
        )
        row.update(
            branch=name,
            initial=initial.hash,
            state=state.hash,
            source_state=parent.hash,
        )
        if state.z != parent.z or state.h != parent.h or state.C != parent.C:
            raise RuntimeError(
                "Continuation intervention changed shared structure/settings/membership"
            )
        if row["status"] == "completed":
            try:
                row["V"] = _score(backend, model, V, ledger, "mechanism_V", True)
            except Exception as exc:
                row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        state.save(output / (name + ".json"))
        prepared.append((row, state, ledger))
    result["contract"]["initial_intervention_changes"] = {
        "reset_receiving_changes_beta": branches["reset_receiving"].beta
        != branches["complete_state"].beta
        if {"reset_receiving", "complete_state"} <= branches.keys()
        else None,
        "cold_local_changes_phi": branches["cold_local"].phi
        != branches["complete_state"].phi
        if {"cold_local", "complete_state"} <= branches.keys()
        else None,
    }
    atomic_write(
        output / "V_choices.json",
        canonical(
            {
                "selection_role": "none: all prespecified branches retained",
                "T_read": False,
                "branches": [
                    {k: row[k] for k in ("branch", "status", "state", "V") if k in row}
                    for row, _, _ in prepared
                ],
            }
        ),
    )
    for row, state, ledger in prepared:
        if row["status"] == "completed":
            try:
                with work(ledger, "mechanism_T_construct"):
                    model = backend.assembly(state, cache, mode="import")
                row["T"] = _score(backend, model, T, ledger, "mechanism_independent_T")
            except Exception as exc:
                row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        result["rows"].append(_finish_row(row, ledger))
    steps = [row["completed_fit_steps"] for row in result["rows"]]
    expected = epochs * len(D.batches)
    result["matched_work_eligible"] = all(
        row["status"] == "completed" and row["completed_fit_steps"] == expected
        for row in result["rows"]
    )
    result["status"] = (
        "completed"
        if all(row["status"] == "completed" for row in result["rows"])
        else "incomplete"
    )
    result["work"] = {
        "unit": "minibatch optimizer steps",
        "expected_per_branch": expected,
        "actual_per_branch": steps,
    }
    return _save_result(result, output)


def match_completed_work(
    records,
    policies,
    pair_fields=("dataset", "detector", "seed"),
    tolerance=0.05,
    expected_pairs=None,
):
    """Register all pairs; reject missing, failed, or under-target observations."""
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("Distinct registered policies required")
    tolerance = _finite(tolerance, "tolerance")
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must lie in [0,1)")
    groups = {}
    if expected_pairs is not None:
        for pair in expected_pairs:
            if not isinstance(pair, dict) or any(
                name not in pair for name in pair_fields
            ):
                raise ValueError("Expected pair is missing its registered identity")
            key = canonical({name: pair[name] for name in pair_fields}).decode()
            if key in groups:
                raise ValueError("Duplicate expected pair")
            groups[key] = []
    for row in records:
        if (
            any(key not in row for key in (*pair_fields, "policy"))
            or row["policy"] not in policies
        ):
            raise ValueError("Unregistered policy or missing pair identity")
        key = canonical({name: row[name] for name in pair_fields}).decode()
        if expected_pairs is not None and key not in groups:
            raise ValueError("Observed pair is outside the registered expected_pairs")
        groups.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(groups.items()):
        reasons, by_policy, targets, units = [], {}, [], []
        for row in rows:
            policy = row["policy"]
            if policy in by_policy:
                reasons.append("duplicate policy: " + policy)
            by_policy[policy] = row
            missing = [
                name
                for name in (
                    "status",
                    "work_unit",
                    "target_work",
                    "actual_work",
                    "value",
                )
                if row.get(name) is None
            ]
            if missing:
                reasons.append(policy + " missing " + ",".join(missing))
                continue
            try:
                target = _finite(row["target_work"], "target_work")
                actual = _finite(row["actual_work"], "actual_work")
                _finite(row["value"], "value")
                if target <= 0 or actual < 0:
                    raise ValueError(
                        "positive target and nonnegative actual work required"
                    )
                if row["status"] != "completed":
                    reasons.append(policy + " did not complete")
                if not target * (1 - tolerance) <= actual <= target:
                    reasons.append(policy + " outside registered common-work window")
                targets.append(target)
                if not isinstance(row["work_unit"], str) or not row["work_unit"]:
                    reasons.append(policy + " invalid work_unit")
                else:
                    units.append(row["work_unit"])
            except ValueError as exc:
                reasons.append(policy + ": " + str(exc))
        reasons.extend("missing policy: " + p for p in policies if p not in by_policy)
        if targets and len(set(targets)) != 1:
            reasons.append("different registered work targets")
        if units and (
            len(set(units)) != 1 or not isinstance(units[0], str) or not units[0]
        ):
            reasons.append("different or invalid work units")
        eligible = not reasons
        item = {
            "pair": json.loads(key),
            "eligible": eligible,
            "reasons": reasons,
            "observations": rows,
            "differences": None,
        }
        if eligible:
            item["differences"] = {
                p: by_policy[policies[0]]["value"] - by_policy[p]["value"]
                for p in policies[1:]
            }
        result.append(item)
    return {
        "kind": "matched_work_eligibility",
        "policies": list(policies),
        "tolerance": tolerance,
        "pair_fields": list(pair_fields),
        "registered_pairs": len(result),
        "eligible_pairs": sum(item["eligible"] for item in result),
        "pairs": result,
        "expected_pairs_locked": expected_pairs is not None,
        "missing_values_imputed": False,
    }


def run_mechanism_job(spec):
    """Native CLI adapter. Unbound files/calibration remain explicit errors."""
    if spec.get("kind") not in (
        "receiving",
        "continuation",
        "configuration_composition",
        "matched_work",
    ):
        raise ValueError("Unknown mechanism kind")
    if spec.get("kind") == "matched_work":
        records = json.loads(Path(spec["records"]).read_text(encoding="utf-8"))
        result = match_completed_work(
            records,
            spec["policies"],
            tuple(spec.get("pair_fields", ("dataset", "detector", "seed"))),
            spec.get("tolerance", 0.05),
            expected_pairs=spec.get("expected_pairs"),
        )
        return _save_result(result, _output_directory(spec["output"]))
    from .backend import YoloBackend
    from .cache import PackageCache
    from .config import effective_formal, load_config
    from .data import load_detection_split, validate_registry, registry_options
    from .metering import CostProfile
    from .runtime import MemoryGuard

    config = load_config(
        spec["config"],
        formal=spec.get("formal", False),
        overrides=spec.get("overrides"),
    )
    formal = effective_formal(config, spec.get("formal", False))
    detector = json.loads(Path(spec["detector_config"]).read_text(encoding="utf-8"))
    registry = json.loads(Path(spec["registry"]).read_text(encoding="utf-8"))
    validate_registry(registry["records"], formal=formal, **registry_options(registry))
    if (
        tuple(detector["labels"]) != tuple(registry["labels"])
        or len(detector["labels"]) != 8
    ):
        raise ValueError("Eight-class fixed-vocabulary detector/registry required")
    backend = YoloBackend(
        spec["checkpoint"],
        detector["ports"],
        detector["labels"],
        detector["preprocessing"],
        detector["image_size"],
        config["runtime"],
    )
    if config.get("cost_profile"):
        backend.cost_profile = CostProfile(
            config["cost_profile"], backend.runtime.identity(), formal
        )
        profile = backend.cost_profile.data
        if (
            profile.get("base_sha256") != backend.identity
            or profile.get("execution") != backend.execution
            or canonical(profile.get("port_interfaces"))
            != canonical([asdict(p) for p in backend.specs])
        ):
            raise ValueError(
                "Calibration must use this exact base and six port interfaces"
            )
    backend.memory_guard = MemoryGuard(
        config["working_memory"]["process_bytes"],
        config["working_memory"]["minimum_free_bytes"],
        config["working_memory"].get("cuda_bytes"),
    )
    parent = State.load(spec["state"])
    cache = PackageCache(config["cache_bytes"], backend.specs)
    for file in spec.get("packages", []):
        package = Package.read(Path(file).read_bytes())
        cache.put(package.hash, package.wire)
    cache.pin("diagnostic-parent", parent.C)
    splits = [
        load_detection_split(
            spec["registry"],
            spec["client"],
            spec["stage"],
            role,
            detector["image_size"],
            detector["labels"],
            formal=formal,
            batch_size=config["fit_batch_size"]
            if role == "D"
            else config["evaluation_batch_size"],
            domain=spec["domain"] if role == "T" else None,
        )
        for role in ("D", "V", "T")
    ]
    runners = {
        "receiving": receiving_comparison,
        "continuation": continuation_comparison,
        "configuration_composition": configuration_composition_comparison,
    }
    if spec.get("kind") not in runners:
        raise ValueError(
            "Mechanism kind must be receiving, continuation, or matched_work"
        )
    result = runners[spec["kind"]](
        backend, cache, *splits, parent, spec["diagnostic"], config, spec["output"]
    )
    result["effective_formal"] = formal
    return _save_result(result, Path(spec["output"]))
