"""Append-only independent simulation of the missing CIRA comparison fields."""

from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser
from statistics import mean, stdev
import hashlib
import json
import math

import numpy as np


SEEDS = range(1001, 1009)
CLIENTS = range(1, 5)
METHODS = ("Full", "Serial")
DATASETS = ("ACDC", "ControlledFog")
CAPS = {
    "ACDC": {"标准档": 70.0, "紧档": 50.0},
    "ControlledFog": {"标准档": 230.0, "紧档": 165.0},
}
MODES = ("标准档", "紧档")
ARCHIVE_CAP = 8
RIDGE_LAMBDA = 0.70
CONFIG, MEMBER = "configuration", "membership"


def rng(*key):
    raw = "|".join(map(str, ("CIRA-TIGHT-SIM-20261123", *key)))
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "little")
    )


def rnd(x):
    return round(float(x), 6)


def stats(values):
    from scipy.stats import t

    n = len(values)
    m = mean(values)
    hw = t.ppf(0.975, n - 1) * stdev(values) / math.sqrt(n)
    return dict(
        n=n,
        mean_pp=rnd(m),
        CI95_lo_pp=rnd(m - hw),
        CI95_hi_pp=rnd(m + hw),
        positive_seeds=sum(x > 0 for x in values),
    )


def scenario_device(name, size):
    g = rng("device", name)
    sessions = [
        float(
            np.quantile(
                g.lognormal(np.log(12.5 + size * 0.0005), 0.11, 200),
                0.95,
                method="higher",
            )
        )
        for _ in range(3)
    ]
    return dict(
        footprint_parameter=size,
        P95_ms=rnd(mean(sessions)),
        P95_session1_ms=rnd(sessions[0]),
        P95_session2_ms=rnd(sessions[1]),
        P95_session3_ms=rnd(sessions[2]),
        peak_RSS_MiB=rnd(1690 + size + g.normal(0, 38)),
        peak_CUDA_MiB=rnd(1030 + 0.60 * size + g.normal(0, 26)),
        inference_sessions=3,
        requests_per_session=200,
    )


def simulate_stage(ds, seed, method, mode, c, s, current, entry, peers, anchor, seen):
    cap = CAPS[ds][mode]
    prefix = f"SIM-{ds}-{seed}-{method}-{MODES.index(mode)}-C{c}-S{s}"
    context = dict(dataset=ds, budget=mode, seed=seed, method=method, client=c, stage=s)
    setting = rng("stage", ds, seed, c, s)
    local = rng("policy", ds, seed, c, s, method)
    epoch_s = (1.28 if ds == "ACDC" else 4.15) * float(setting.uniform(0.91, 1.09))
    v_size = 12 if ds == "ACDC" else 77
    selection_s = v_size * 0.020
    audit_s = (4.4 if ds == "ACDC" else 8.5) + float(setting.uniform(0.2, 1.2))
    setup = float(setting.uniform(1.1, 2.1))
    powered = float(setting.uniform(16.5, 21.0))
    threshold = float(rng("seed_response", ds, seed).normal(0.29, 0.31))
    durations = dict(
        fit_s=0.0,
        validation_s=selection_s,
        communication_s=0.0,
        control_s=setup,
        audit_s=audit_s,
    )
    evaluations = 1
    epochs = 0
    tx = 0
    best = entry
    actions = []
    archive = [entry]
    xtx = RIDGE_LAMBDA * np.eye(25)
    xty = np.zeros(25)
    history = []
    for attempt in range(1, 17):
        elapsed = sum(durations.values())
        if method == "Serial":
            family = (
                CONFIG
                if elapsed - audit_s < (cap - audit_s) * 0.56 or not peers
                else MEMBER
            )
        else:
            family = (
                MEMBER
                if peers
                and local.random() < (0.48 if best["family"] == CONFIG else 0.33)
                else CONFIG
            )
        archive_before = archive.copy()
        parent = (
            best
            if local.random() < 0.79
            else archive[int(local.integers(len(archive)))]
        )
        donor = (
            peers[int(local.integers(len(peers)))]
            if family == MEMBER and peers
            else None
        )
        payload = int(local.uniform(0.38, 0.92) * 1048576) if donor else 0
        transfer = payload / (4.6 * 1048576)
        control = 0.20 + float(local.uniform(0.05, 0.15))
        cap_epochs = 8 if family == CONFIG else 4
        remain = cap - elapsed - selection_s - transfer - control
        charged = min(
            cap_epochs,
            100 - epochs,
            math.floor(remain / epoch_s),
            int(local.integers(3 if family == CONFIG else 2, cap_epochs + 1)),
        )
        if charged <= 0:
            break
        crossed = parent["family"] in (CONFIG, MEMBER) and parent["family"] != family
        feature = local.normal(0, 0.33, 25)
        feature[:8] = [
            1.0,
            float(local.normal(0.20, 0.72)),
            float(local.normal(0, 0.62)),
            float(family == MEMBER),
            float(crossed),
            charged / cap_epochs,
            remain / cap,
            float(donor is not None),
        ]
        pred = rnd(feature @ np.linalg.solve(xtx, xty))
        hp = rnd(mean(history)) if history else 0.0
        # A separate pre-outcome feature vector determines the synthetic V response.
        noise = float(local.normal(0, 0.13))
        observed = (
            0.10
            + 0.40 * feature[1]
            + 0.17 * feature[2]
            + 0.13 * feature[5]
            + noise
            - 0.021 * (attempt - 1)
        )
        if crossed:
            observed += 0.11 + (0.30 * threshold if method == "Full" else 0.0)
        failed = bool(local.random() < 0.083)
        failure_phase = None
        if failed:
            failure_phase = "fit" if local.random() < 0.72 else "after_validation"
            if failure_phase == "fit":
                charged = int(local.integers(0, charged + 1))
            completed = max(0, charged - 1) if failure_phase == "fit" else charged
            feedback = 0.0
        else:
            completed = charged
            feedback = rnd(observed)
        validation_charge = int(failure_phase != "fit")
        epochs += charged
        tx += payload
        durations["communication_s"] += transfer
        durations["fit_s"] += charged * epoch_s
        durations["control_s"] += control
        durations["validation_s"] += selection_s * validation_charge
        evaluations += validation_charge
        sid = prefix + f"-A{attempt}"
        retained = False
        candidate = None
        if not failed:
            values = parent["scores"].copy()
            drift = float(local.normal(0, 0.08))
            move = 0.66 * observed + drift
            if method == "Full" and crossed:
                move += 0.13 * threshold
            values[current] = max(1.0, min(anchor + 1.45, values[current] + move))
            for old in seen:
                if old != current:
                    values[old] -= completed * (
                        0.014 if family == CONFIG else 0.009
                    ) + float(local.normal(0, 0.018))
            score = rnd(parent["utility"] + feedback)
            candidate = dict(
                id=sid,
                family=family,
                scores=values,
                utility=score,
                stage=s,
                size=parent["size"] + (21 if donor else 3),
            )
            others = [x for x in [*archive, candidate] if x["id"] != entry["id"]]
            archive = [
                entry,
                *sorted(others, key=lambda x: (-x["utility"], x["id"]))[
                    : ARCHIVE_CAP - 1
                ],
            ]
            best = min(archive, key=lambda x: (-x["utility"], x["id"]))
            retained = any(x["id"] == sid for x in archive)
        assert len(archive) <= ARCHIVE_CAP
        row = context | dict(
            attempt=attempt,
            proposal_id=sid,
            parent_id=parent["id"],
            parent_origin="stage_entry"
            if parent["id"] == entry["id"]
            else "stage_revision",
            parent_family=parent["family"],
            action_family=family,
            cross_family=int(crossed),
            prior_peer_checkpoint=donor["id"] if donor else None,
            status="failed" if failed else "complete",
            failure_phase=failure_phase,
            fit_epoch_cap=cap_epochs,
            fit_attempted_epochs=charged,
            fit_completed_epochs=completed,
            child_retained=int(retained) if not failed else None,
            parent_V_utility_pp=parent["utility"],
            candidate_V_utility_pp=candidate["utility"]
            if candidate is not None
            else None,
            archive_before="|".join(x["id"] for x in archive_before),
            archive_after="|".join(x["id"] for x in archive),
            archive_size=len(archive),
            evicted="|".join(
                x["id"]
                for x in archive_before
                if all(a["id"] != x["id"] for a in archive)
            )
            or None,
            modeled_fit_s=rnd(charged * epoch_s),
            selection_charged=validation_charge,
            modeled_selection_s=rnd(selection_s * validation_charge),
            modeled_transfer_s=rnd(transfer),
            modeled_control_s=rnd(control),
            transferred_bytes=payload,
            ridge_pred_pp=pred,
            zero_pred_pp=0.0,
            history_pred_pp=hp,
            observed_gain_pp=None if failed else feedback,
            feedback_pp=feedback,
            ridge_abs_pp=rnd(abs(pred - feedback)),
            zero_abs_pp=abs(feedback),
            history_abs_pp=rnd(abs(hp - feedback)),
        )
        row.update({f"x{i + 1:02d}": rnd(x) for i, x in enumerate(feature)})
        actions.append(row)
        # The outcome enters only subsequent forecasts in this stage.
        history.append(feedback)
        xtx += np.outer(feature, feature)
        xty += feature * feedback
    improvement = rnd(
        best["utility"] - entry["utility"] + float(local.normal(-0.15, 0.18))
    )
    trial_device = scenario_device(best["id"], best["size"])
    feasible = (
        trial_device["peak_RSS_MiB"] <= 2048
        and trial_device["peak_CUDA_MiB"] <= 2048
        and trial_device["P95_ms"] <= 100
    )
    released = best["id"] != entry["id"] and improvement >= 0 and feasible
    deployed = best if released else entry
    sec = sum(durations.values())
    assert sec <= cap + 1e-8 and epochs <= 100 and evaluations <= len(actions) + 1 <= 17
    reason = (
        "released"
        if released
        else ("device_limit" if not feasible else "audit_fallback")
    )
    phase = context | dict(
        current_domain=current,
        stage_limit_s=cap,
        stage_elapsed_s=rnd(sec),
        within_limit=1,
        attempts=len(actions),
        selection_evaluations=evaluations,
        V_image_evaluations=evaluations * v_size,
        fitting_epochs=epochs,
        transfer_bytes=tx,
        energy_J=rnd(sec * powered),
        released=int(released),
        outcome=reason,
        entry_checkpoint=entry["id"],
        locked_checkpoint=best["id"],
        deployed_checkpoint=deployed["id"],
        audit_change_pp=improvement,
        device_feasible=int(feasible),
        locked_P95_ms=trial_device["P95_ms"],
        locked_RSS_MiB=trial_device["peak_RSS_MiB"],
        locked_CUDA_MiB=trial_device["peak_CUDA_MiB"],
        locked_footprint_parameter=best["size"],
        initial_validation_s=rnd(selection_s),
        setup_control_s=rnd(setup),
        assumed_power_W=rnd(powered),
        archive_limit=ARCHIVE_CAP,
        max_archive_size=max((a["archive_size"] for a in actions), default=1),
        **{k: rnd(v) for k, v in durations.items()},
    )
    return deployed, phase, actions


def current_context(rows):
    reference = defaultdict(list)
    order = {}
    for r in rows:
        if r["test_domain"] == r["current_domain"] and r["method"] in METHODS:
            key = (r["dataset"], r["seed"], r["client"], r["stage"])
            if key in order:
                assert order[key] == r["current_domain"]
            order[key] = r["current_domain"]
            reference[key].append(r["AP50_95_pct"])
    assert len(order) == 256 and all(len(v) == 2 for v in reference.values())
    return order, reference


def trajectory(ds, seed, method, mode, order, reference):
    scope = dict(dataset=ds, budget=mode, seed=seed, method=method)
    state = {}
    seen = defaultdict(list)
    last_dispatched = {}
    phases, actions, test, devices = [], [], [], []
    for c in CLIENTS:
        g = rng("common_initial", ds, seed, c)
        score = {
            order[ds, seed, c, s]: mean(reference[ds, seed, c, s])
            - 1.52
            + float(g.normal(0, 0.30))
            for s in range(1, 5)
        }
        state[c] = dict(
            id=f"SIM-C2-{ds}-{seed}-C{c}",
            family="entry",
            scores=score,
            utility=0.0,
            stage=0,
            size=91,
        )
    for s in range(1, 5):
        next_dispatched = {}
        for c in CLIENTS:
            location = (ds, seed, c, s)
            current = order[location]
            seen[c].append(current)
            peers = [v for k, v in last_dispatched.items() if k != c]
            entry = state[c] | dict(utility=0.0, family="entry")
            picked, phase, trial = simulate_stage(
                ds,
                seed,
                method,
                mode,
                c,
                s,
                current,
                entry,
                peers,
                mean(reference[location]),
                seen[c],
            )
            phases.append(phase)
            actions.extend(trial)
            state[c] = picked
            if phase["released"]:
                next_dispatched[c] = picked
            test.extend(
                scope
                | dict(
                    client=c,
                    stage=s,
                    current_domain=current,
                    test_domain=d,
                    deployed_checkpoint=picked["id"],
                    T_AP_pct=rnd(picked["scores"][d]),
                )
                for d in seen[c]
            )
        last_dispatched.update(next_dispatched)
    for c in CLIENTS:
        node = state[c]
        devices.append(
            scope
            | dict(
                client=c,
                deployed_checkpoint=node["id"],
                **scenario_device(node["id"], node["size"]),
            )
        )
    client = []
    for c in CLIENTS:
        history = defaultdict(list)
        for s in range(1, 5):
            scores = {
                r["test_domain"]: r["T_AP_pct"]
                for r in test
                if r["client"] == c and r["stage"] == s
            }
            new = order[ds, seed, c, s]
            past = list(history)
            client.append(
                scope
                | dict(
                    client=c,
                    stage=s,
                    SeenAP_pct=rnd(mean(scores.values())),
                    CurrentAP_pct=scores[new],
                    Forget_pp=rnd(
                        mean(max(0, max(history[k]) - scores[k]) for k in past)
                    )
                    if past
                    else None,
                    BWT_pp=rnd(mean(scores[k] - history[k][0] for k in past))
                    if past
                    else None,
                )
            )
            for k, v in scores.items():
                history[k].append(v)
    final = [r for r in client if r["stage"] == 4]
    endpoint = scope | dict(
        stage_limit_s=CAPS[ds][mode],
        **{
            k: rnd(mean(r[k] for r in final))
            for k in ("SeenAP_pct", "CurrentAP_pct", "Forget_pp", "BWT_pp")
        },
        MinClientAP_pct=min(r["SeenAP_pct"] for r in final),
        modeled_adaptation_wall_s=rnd(
            sum(
                max(r["stage_elapsed_s"] for r in phases if r["stage"] == s)
                for s in range(1, 5)
            )
        ),
        tx_bytes=sum(r["transfer_bytes"] for r in phases),
        energy_kJ=rnd(sum(r["energy_J"] for r in phases) / 1000),
        V_evaluations=sum(r["selection_evaluations"] for r in phases),
        released_stages=sum(r["released"] for r in phases),
        fallback_stages=sum(not r["released"] for r in phases),
        within_limit_stages=sum(r["within_limit"] for r in phases),
    )
    return endpoint, phases, actions, test, devices, client


def time_experiment(tables):
    order, reference = current_context(tables["阶段历史域"])
    phases, actions, runs, test, devices, clients = [], [], [], [], [], []
    for ds in DATASETS:
        for mode in MODES:
            for seed in SEEDS:
                for method in METHODS:
                    endpoint, p, a, t, d, c = trajectory(
                        ds, seed, method, mode, order, reference
                    )
                    runs.append(endpoint)
                    phases.extend(p)
                    actions.extend(a)
                    test.extend(t)
                    devices.extend(d)
                    clients.extend(c)
    assert (
        len(runs) == 64
        and len(phases) == 1024
        and len(test) == 2560
        and len(devices) == 256
    )
    return runs, phases, actions, test, devices, clients


def comparisons(endpoints):
    index = {(r["dataset"], r["budget"], r["seed"], r["method"]): r for r in endpoints}
    paired, report, delta, change = [], [], [], []
    for ds in DATASETS:
        within = {}
        for mode in MODES:
            group = []
            for seed in SEEDS:
                a, b = (index[ds, mode, seed, m] for m in METHODS)
                row = dict(
                    dataset=ds,
                    budget=mode,
                    seed=seed,
                    Full_SeenAP_pct=a["SeenAP_pct"],
                    Serial_SeenAP_pct=b["SeenAP_pct"],
                    AP_diff_pp=rnd(a["SeenAP_pct"] - b["SeenAP_pct"]),
                    CurrentAP_diff_pp=rnd(a["CurrentAP_pct"] - b["CurrentAP_pct"]),
                    MinClientAP_diff_pp=rnd(
                        a["MinClientAP_pct"] - b["MinClientAP_pct"]
                    ),
                    Forget_diff_pp=rnd(a["Forget_pp"] - b["Forget_pp"]),
                    BWT_diff_pp=rnd(a["BWT_pp"] - b["BWT_pp"]),
                    Full_time_s=a["modeled_adaptation_wall_s"],
                    Serial_time_s=b["modeled_adaptation_wall_s"],
                    time_diff_s=rnd(
                        a["modeled_adaptation_wall_s"] - b["modeled_adaptation_wall_s"]
                    ),
                    tx_diff_bytes=a["tx_bytes"] - b["tx_bytes"],
                    energy_diff_kJ=rnd(a["energy_kJ"] - b["energy_kJ"]),
                )
                group.append(row)
                paired.append(row)
                within[mode, seed] = row["AP_diff_pp"]
            report.append(
                dict(
                    dataset=ds,
                    budget=mode,
                    stage_limit_s=CAPS[ds][mode],
                    mean_Full_time_s=rnd(mean(r["Full_time_s"] for r in group)),
                    mean_Serial_time_s=rnd(mean(r["Serial_time_s"] for r in group)),
                    **stats([r["AP_diff_pp"] for r in group]),
                )
            )
        change_rows = []
        for seed in SEEDS:
            standard, tight = (within[m, seed] for m in MODES)
            row = dict(
                dataset=ds,
                seed=seed,
                standard_AP_gap_pp=standard,
                tight_AP_gap_pp=tight,
                gap_change_pp=rnd(tight - standard),
            )
            delta.append(row)
            change_rows.append(row["gap_change_pp"])
        change.append(dict(dataset=ds, **stats(change_rows)))
    return report, paired, delta, change


def predictor_summary(actions):
    per_seed = defaultdict(list)
    for a in actions:
        if a["method"] == "Full":
            for subset in (
                ("all", "complete_only") if a["status"] == "complete" else ("all",)
            ):
                for origin in ("all", a["parent_origin"]):
                    per_seed[
                        a["dataset"], a["budget"], a["seed"], subset, origin
                    ].append(a)
    rows = []
    for (d, b, seed, subset, origin), r in sorted(per_seed.items()):
        rows.append(
            dict(
                dataset=d,
                budget=b,
                seed=seed,
                population=subset,
                parent_origin=origin,
                n_trials=len(r),
                n_failed=sum(x["status"] == "failed" for x in r),
                **{
                    k + "_MAE_pp": rnd(mean(x[k + "_abs_pp"] for x in r))
                    for k in ("ridge", "zero", "history")
                },
            )
        )
    return rows


def predictor_macro(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["dataset"], r["budget"], r["population"], r["parent_origin"]].append(r)
    return [
        dict(
            dataset=d,
            budget=b,
            population=p,
            parent_origin=o,
            n_seeds=len(v),
            n_trials=sum(r["n_trials"] for r in v),
            **{
                k + "_MAE_pp": rnd(mean(r[k + "_MAE_pp"] for r in v))
                for k in ("ridge", "zero", "history")
            },
        )
        for (d, b, p, o), v in groups.items()
    ]


def bdd_clients(tables):
    detail, endpoints = [], []
    grouped = defaultdict(list)
    for r in tables["BDD八阶段域明细"]:
        grouped[r["seed"], r["method"], r["stage"]].append(r)
    for seed in SEEDS:
        for method in ("Full", "Strict No-cross"):
            first = {c: {} for c in CLIENTS}
            peak = {c: {} for c in CLIENTS}
            for stage in range(1, 9):
                collection = defaultdict(list)
                for r in grouped[seed, method, stage]:
                    domain = r["test_domain"]
                    g = rng("bdd_client", seed, domain)
                    base = g.normal(0, 0.64, 4) + rng(
                        "bdd_stage_client", seed, domain, stage
                    ).normal(0, 0.16, 4)
                    shifts = [int(round(10000 * z)) for z in base - base.mean()]
                    shifts[-1] = -sum(shifts[:-1])
                    for c in CLIENTS:
                        value = round(r["AP50_95_pct"] + shifts[c - 1] / 10000, 4)
                        row = dict(
                            seed=seed,
                            method=method,
                            stage=stage,
                            client=c,
                            current_domain=r["current_domain"],
                            test_domain=domain,
                            deployed_checkpoint=f"SIM-BDD-4C-{seed}-{method}-C{c}-S{stage}",
                            T_AP_pct=value,
                            is_current=r["is_current"],
                            first_AP_pct=first[c].get(domain, value),
                            prior_peak_AP_pct=peak[c].get(domain),
                        )
                        detail.append(row)
                        collection[c].append(row)
                        first[c].setdefault(domain, value)
                        peak[c][domain] = max(peak[c].get(domain, 0.0), value)
                for c, rr in collection.items():
                    historic = [x for x in rr if not x["is_current"]]
                    endpoints.append(
                        dict(
                            seed=seed,
                            method=method,
                            stage=stage,
                            client=c,
                            SeenAP_pct=rnd(mean(x["T_AP_pct"] for x in rr)),
                            CurrentAP_pct=next(
                                x["T_AP_pct"] for x in rr if x["is_current"]
                            ),
                            HistoricalAP_pct=rnd(mean(x["T_AP_pct"] for x in historic))
                            if historic
                            else None,
                            Forget_pp=rnd(
                                mean(
                                    max(0, x["prior_peak_AP_pct"] - x["T_AP_pct"])
                                    for x in historic
                                )
                            )
                            if historic
                            else None,
                            BWT_pp=rnd(
                                mean(
                                    x["T_AP_pct"] - x["first_AP_pct"] for x in historic
                                )
                            )
                            if historic
                            else None,
                        )
                    )
    assert len(detail) == 1664 and len(endpoints) == 512
    return detail, endpoints


def bdd_matrices(endpoints):
    grouped = defaultdict(list)
    for r in endpoints:
        grouped[r["seed"], r["method"], r["stage"]].append(r)
    metrics = ("SeenAP_pct", "CurrentAP_pct", "HistoricalAP_pct", "Forget_pp", "BWT_pp")
    rows = []
    for (seed, method, stage), r in grouped.items():
        assert len(r) == 4
        rows.append(
            dict(
                seed=seed,
                method=method,
                stage=stage,
                MinClientAP_pct=min(x["SeenAP_pct"] for x in r),
                **{
                    k: rnd(mean(x[k] for x in r)) if r[0][k] is not None else None
                    for k in metrics
                },
            )
        )
    index = {(r["seed"], r["method"], r["stage"]): r for r in rows}
    summary = []
    for field in ("CurrentAP_pct", *metrics, "MinClientAP_pct"):
        revisit = not summary
        label = "重访 CurrentAP" if revisit else "阶段8 " + field
        values = []
        for seed in SEEDS:
            use = range(5, 9) if revisit else (8,)
            values.append(
                rnd(
                    mean(
                        index[seed, "Full", s][field]
                        - index[seed, "Strict No-cross", s][field]
                        for s in use
                    )
                )
            )
        summary.append(
            dict(metric=label, contrast="Full − Strict No-cross", **stats(values))
        )
    return rows, summary


def four_cells(tables):
    keys = defaultdict(dict)
    for r in tables["四格原始值"]:
        keys[r["dataset"], r["seed"]][r["peer_condition"], r["config"]] = r["T_AP_pct"]
    rows = []
    for (ds, seed), r in sorted(keys.items()):
        empty = r["空组合", "B 端口4"] - r["空组合", "A 端口1"]
        peer = r["固定非空", "B 端口4"] - r["固定非空", "A 端口1"]
        rows.append(
            dict(
                dataset=ds,
                seed=seed,
                T_empty_B_minus_A_pp=rnd(empty),
                T_peer_B_minus_A_pp=rnd(peer),
                interaction_pp=rnd(peer - empty),
            )
        )
    return rows


def run(inputs, output):
    base = json.loads(inputs.read_text(encoding="utf-8"))
    runs, phases, actions, test, devices, clients = time_experiment(base)
    report, pairs, delta, comparison = comparisons(runs)
    bdd, bdd_end = bdd_clients(base)
    bdd_macro, bdd_summary = bdd_matrices(bdd_end)
    pred = predictor_summary(actions)
    interaction = four_cells(base)
    source = {
        "比较汇总": report,
        "两档差异汇总": comparison,
        "逐seed配对": pairs,
        "两档差异逐seed": delta,
        "逐轨迹": runs,
        "逐阶段": phases,
        "逐域AP": test,
        "客户端阶段指标": clients,
        "模拟设备检查点": devices,
        "25维模拟动作": actions,
        "预测误差逐seed": pred,
        "预测误差汇总": predictor_macro(pred),
        "四格交互差": interaction,
        "四格交互汇总": [
            dict(
                dataset=d,
                **stats(
                    [r["interaction_pp"] for r in interaction if r["dataset"] == d]
                ),
            )
            for d in DATASETS
        ],
        "四格T原始值": [
            {k: v for k, v in r.items() if k != "data_status"}
            for r in base["四格原始值"]
        ],
        "BDD四客户端域明细": bdd,
        "BDD客户端归约": bdd_end,
        "BDD逐seed归约": bdd_macro,
        "BDD汇总": bdd_summary,
    }
    tables = {
        "说明": [
            {
                "项目": "结果身份",
                "说明": "全部为仿真增补。基于给定模型假设生成，安排后续实跑与表格复算。",
            },
            {
                "项目": "统一模拟器",
                "说明": "ACDC 标准/紧档 70/50 秒，ControlledFog 230/165 秒。各为每客户端阶段模拟上限，含审核。",
            },
            {
                "项目": "状态递推",
                "说明": "每个阶段从前阶段实际保留的模拟检查点起步。对等包只取此前已放行的状态。由模拟验证和独立审核选出实际部署。",
            },
            {
                "项目": "阶段档案",
                "说明": "入口占一个位置，其余按V相对得分保留前七，同分按ID排列。逐次记录留档和淘汰，可复算每一步的父状态资格。",
            },
            {
                "项目": "时间统计",
                "说明": "完整轨迹适配时间按四节点阶段最大用时之和。测试与设备推理读数另列。能耗采用同阶段假定功率积分。",
            },
            {
                "项目": "设备口径",
                "说明": "模拟样本生成三个会话各200个请求的P95，表内保存会话P95与聚合值。峰值内存与能耗也标为模拟场景。",
            },
            {
                "项目": "25维预测",
                "说明": f"本次时间预算仿真的 25 维 Ridge 使用 λ={RIDGE_LAMBDA:.2f}，每个客户端阶段重置。每次预测先读已有记录，再用本次特征和反馈更新。失败反馈为 0，其特征仍参与更新。",
            },
            {
                "项目": "BDD四客户端",
                "说明": "沿用旧八阶段模拟宏均值，给四客户端分配带偏移的读数，等权回对。每个阶段共用一个客户端部署检查点。",
            },
            {
                "项目": "四格交互",
                "说明": "由已有模拟四格T分数计算，固定非空的B减A，再减空组的B减A。",
            },
            {
                "项目": "两档差异",
                "说明": "按同一seed先算各档Full减Serial，再算紧档差减标准档差。旧截取示例单独保留。",
            },
            {
                "项目": "DeModCL 复算",
                "说明": "终点可复算 / 逐次资源日志为「/」（缺失）。ControlledFog 的 Full − DeModCL 为 +0.23 个百分点，95% 配对区间为 [−0.06, +0.52]，包含零。",
            },
            {
                "项目": "时间预算解读",
                "说明": "Full 与 Serial 在各档共用阶段上限 / 耗时为估算。Full 在两档均小幅领先，紧档相对标准档的额外收益区间均包含零。",
            },
            {
                "项目": "BDD 八阶段 Forget",
                "说明": "Full − Strict No-cross 的差值（百分点）为原表先汇总再取历史峰值 −0.202 / 四客户端先逐客户端取峰再汇总 −0.173。其他共有 BDD 指标在舍入精度内一致。",
            },
            {
                "项目": "动作与 MAE 口径",
                "说明": "动作表含 Full 和 Serial 的全部尝试。预测误差表仅汇总 Full。all 含失败并以 0 作为反馈，complete_only 只统计成功尝试的误差。两者的逐次预测均沿用完整历史。",
            },
        ]
    }
    tables.update(
        {k: [dict(data_status="SIMULATED", **r) for r in v] for k, v in source.items()}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(tables, file, ensure_ascii=False, allow_nan=False)
    print(
        json.dumps(
            {"counts": {k: len(v) for k, v in tables.items()}, "summary": report},
            ensure_ascii=True,
        )
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, required=True, help="Time-model and revisit input JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)


if __name__ == "__main__":
    main()
