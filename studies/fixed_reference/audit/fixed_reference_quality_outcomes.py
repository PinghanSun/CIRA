"""Freeze retrospective DEV-only quality policy, then describe formal endpoints."""

from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, math, statistics
from dev_selection_closure import DATA, OUT, sig, config, sha, display, dump_csv
from d05_quality_availability import METHODS, LEVELS, DEV_SEEDS, EVAL_SEEDS


def save_frozen(path, obj):
    encoded = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode(
        "utf-8"
    )
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(
                "Refusing to overwrite conflicting DEV freeze: " + str(path)
            )
    else:
        path.write_bytes(encoded)
    return sha(path)


def exact_by_seed(pool, setting, seeds):
    out = defaultdict(list)
    for r in pool:
        if r["setting"]["seed"] in seeds and sig(r["setting"]) == sig(setting):
            out[r["setting"]["seed"]].append(r)
    if set(out) != seeds:
        return None
    chosen = []
    for seed in sorted(seeds):
        group = sorted(out[seed], key=lambda r: r["run_id"])
        if len({tuple(r["metrics"]) for r in group}) != 1:
            raise ValueError("Conflicting repeated setting/seed " + repr(setting))
        chosen.append(group[0])
    return chosen


def known_mean(values):
    return statistics.mean(values) if values else None


def known_sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def dev_freeze(records, registry_sha):
    dev = [r for r in records if r["setting"]["block"] == "development"]
    main = defaultdict(dict)
    for r in records:
        s = r["setting"]
        if s["experiment"] == "N01" and s["method"] in METHODS:
            main[s["dataset"], s["profile"]].setdefault(s["method"], s)
    assert len(main) == 6
    targets = []
    policies = []
    support = {}
    for (ds, pr), settings in sorted(main.items()):
        target_setting = dict(settings["Serial"], F_scale=1.0)
        t = exact_by_seed(dev, target_setting, DEV_SEEDS)
        qstar = statistics.mean(r["metrics"][0] for r in t) if t else None
        if t:
            support.update({r["run_id"]: r for r in t})
        targets.append(
            dict(
                dataset=ds,
                profile=pr,
                Qstar=qstar,
                reference_method="Serial",
                F_scale=1.0,
                reference_settings=config(target_setting),
                DEV_support_run_ids=[r["run_id"] for r in t] if t else [],
                DEV_SeenAP_values=[r["metrics"][0] for r in t] if t else [],
                denominator=3 if t else None,
                unavailable_reason=None if t else "missing_exact_DEV_Serial_F1",
            )
        )
        for m in METHODS:
            candidates = []
            for F in LEVELS:
                desired = dict(settings[m], F_scale=F)
                matched = exact_by_seed(dev, desired, DEV_SEEDS)
                if matched:
                    support.update({r["run_id"]: r for r in matched})
                mean = (
                    statistics.mean(r["metrics"][0] for r in matched)
                    if matched
                    else None
                )
                candidates.append(
                    dict(
                        F_scale=F,
                        settings=config(desired),
                        DEV_support_run_ids=[r["run_id"] for r in matched]
                        if matched
                        else [],
                        DEV_SeenAP_values=[r["metrics"][0] for r in matched]
                        if matched
                        else [],
                        mean_DEV_SeenAP=mean,
                        denominator=3 if matched else None,
                        qualifies=mean >= qstar
                        if mean is not None and qstar is not None
                        else None,
                    )
                )
            complete = qstar is not None and all(
                c["denominator"] == 3 for c in candidates
            )
            admissible = [c for c in candidates if c["qualifies"] is True]
            selected = (
                min(admissible, key=lambda c: c["F_scale"])
                if complete and admissible
                else None
            )
            policies.append(
                dict(
                    dataset=ds,
                    profile=pr,
                    method=m,
                    Qstar=qstar,
                    selected_F=selected["F_scale"] if selected else None,
                    selected_settings=selected["settings"] if selected else None,
                    selection_status="DEV_selected_budget"
                    if selected
                    else "DEV_no_qualifying_budget"
                    if complete
                    else "DEV_inestimable",
                    candidate_levels=candidates,
                    formal_seed_labels=sorted(EVAL_SEEDS),
                    planned_formal_denominator=5,
                )
            )
    assert len(targets) == 6 and len(policies) == 36 and len(support) == 324

    support_list = [
        dict(
            run_id=r["run_id"],
            setting=r["setting"],
            metrics=r["metrics"],
            history_AP=r["history_AP"],
        )
        for rid, r in sorted(support.items())
    ]
    support_path = OUT / "dev_support.json"
    support_hash = save_frozen(support_path, support_list)
    frozen = dict(
        schema_version=1,
        review_date="2026-11-07",
        identity="retrospective_classroom_fixed_reference_DEV_policy",
        native_DEV_lock=False,
        timing="Written before this formal endpoint summary; not claimed to predate original generation/evaluation.",
        scope_methods=METHODS,
        scope_cells=6,
        DEV_seeds=sorted(DEV_SEEDS),
        formal_seed_labels=sorted(EVAL_SEEDS),
        target_rule="Mean final SeenAP of three exact-setting DEV Serial F1 records in each dataset-profile cell.",
        selection_rule="Smallest F in{0.5,1,2} with three-seed mean DEV SeenAP>=Qstar; allDEVgridpoints must be estimable.",
        no_qualifying_budget_rule="selected_F=null; do not choose F from formal outcomes; formal_success_rate remainsnull.",
        formal_scores_used_to_select=False,
        source_registry_sha256=registry_sha,
        DEV_support_file=support_path.name,
        DEV_support_file_sha256=support_hash,
        targets=targets,
        policies=policies,
    )
    freeze_path = OUT / "dev_policy.json"
    freeze_sha = save_frozen(freeze_path, frozen)

    prior = json.loads((OUT / "quality_availability.json").read_bytes())
    old_choices = {
        (c["dataset"], c["profile"], c["method"]): c["selected_F"]
        for c in prior["choices"]
    }
    assert all(
        old_choices[p["dataset"], p["profile"], p["method"]] == p["selected_F"]
        for p in policies
    )
    return frozen, freeze_sha


def endpoint_row(policy, record, seed, freeze_sha):
    if record is None:
        return dict(
            dataset=display(policy["dataset"]),
            profile=policy["profile"],
            method=policy["method"],
            seed=seed,
            Qstar=policy["Qstar"],
            selected_F=None,
            formal_run_id=None,
            source_experiment=None,
            formal_evaluated=False,
            SeenAP=None,
            margin_pp=None,
            pass_target=None,
            observed_GFLOP=None,
            observed_online_time_s=None,
            observed_energy_J=None,
            successful_endpoint_GFLOP=None,
            successful_endpoint_online_time_s=None,
            successful_endpoint_energy_J=None,
            failed_endpoint_GFLOP=None,
            failed_endpoint_online_time_s=None,
            failed_endpoint_energy_J=None,
            target_qualifying_endpoint_cost_observed=False,
            missing_or_censored_reason=policy["selection_status"],
            cost_interpretation="No DEV-qualified budget: no formal outcome selected; not a failed evaluated seed.",
            DEV_policy_sha256=freeze_sha,
        )
    v = record["metrics"]
    ok = v[0] >= policy["Qstar"]
    return dict(
        dataset=display(policy["dataset"]),
        profile=policy["profile"],
        method=policy["method"],
        seed=seed,
        Qstar=policy["Qstar"],
        selected_F=policy["selected_F"],
        formal_run_id=record["run_id"],
        source_experiment=record["setting"]["experiment"],
        formal_evaluated=True,
        SeenAP=v[0],
        margin_pp=v[0] - policy["Qstar"],
        pass_target=ok,
        observed_GFLOP=v[15],
        observed_online_time_s=v[18],
        observed_energy_J=v[11],
        successful_endpoint_GFLOP=v[15] if ok else None,
        successful_endpoint_online_time_s=v[18] if ok else None,
        successful_endpoint_energy_J=v[11] if ok else None,
        failed_endpoint_GFLOP=None if ok else v[15],
        failed_endpoint_online_time_s=None if ok else v[18],
        failed_endpoint_energy_J=None if ok else v[11],
        target_qualifying_endpoint_cost_observed=ok,
        missing_or_censored_reason=None if ok else "fixed_budget_endpoint_below_Qstar",
        cost_interpretation="Observed cost of the frozen-F endpoint; not minimum/first-hit cost. Failed endpoint does not imply target was never crossed earlier.",
        DEV_policy_sha256=freeze_sha,
    )


def summarize(policy, g):
    assessed = [r for r in g if r["formal_evaluated"]]
    passed = [r for r in assessed if r["pass_target"]]
    failed = [r for r in assessed if not r["pass_target"]]
    evaluated = bool(assessed)
    out = dict(
        dataset=display(policy["dataset"]),
        profile=policy["profile"],
        method=policy["method"],
        Qstar=policy["Qstar"],
        selected_F=policy["selected_F"],
        selection_status=policy["selection_status"],
        planned_denominator=5,
        evaluated_denominator=len(assessed),
        formal_success_denominator=5 if evaluated else None,
        success_n=len(passed) if evaluated else None,
        failure_n=len(failed) if evaluated else None,
        formal_success_rate=len(passed) / 5 if evaluated else None,
        formal_success_percent=100 * len(passed) / 5 if evaluated else None,
        not_evaluated_seed_slots=5 - len(assessed),
        target_qualifying_cost_n=len(passed),
        failed_endpoint_cost_n=len(failed),
        reason=policy["selection_status"]
        if not evaluated
        else "fixed_budget_endpoint_below_Qstar"
        if failed
        else None,
        source_formal_run_ids=",".join(r["formal_run_id"] for r in assessed),
        paired_seed_labels=",".join(str(r["seed"]) for r in g),
        mean_SeenAP=known_mean([r["SeenAP"] for r in assessed]),
        sample_SD_SeenAP=known_sd([r["SeenAP"] for r in assessed]),
    )
    for name, key in [
        ("GFLOP", "observed_GFLOP"),
        ("online_time_s", "observed_online_time_s"),
        ("energy_J", "observed_energy_J"),
    ]:
        out["mean_all_evaluated_" + name] = known_mean([r[key] for r in assessed])
        out["SD_all_evaluated_" + name] = known_sd([r[key] for r in assessed])
        out["mean_successful_endpoint_" + name] = known_mean([r[key] for r in passed])
        out["mean_failed_endpoint_" + name] = known_mean([r[key] for r in failed])
    out["cost_claim"] = (
        "Frozen-budget endpoint cost by outcome; not time-to-first-attainment or a native device measurement."
    )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry_hash = sha(DATA / "results.json")
    records = json.loads((DATA / "results.json").read_bytes())
    freeze, freeze_sha = dev_freeze(records, registry_hash)

    formal = [r for r in records if r["setting"]["block"] == "evaluation"]
    per_seed = []
    summaries = []
    all_budget = []
    history_error = 0.0
    for p in freeze["policies"]:
        matched = (
            exact_by_seed(formal, p["selected_settings"], EVAL_SEEDS)
            if p["selected_F"] is not None
            else None
        )
        if p["selected_F"] is not None and matched is None:
            raise ValueError("Frozen choice lacks full formal exact-match " + repr(p))
        chosen = {r["setting"]["seed"]: r for r in matched} if matched else {}
        group = []
        for seed in sorted(EVAL_SEEDS):
            r = chosen.get(seed)
            row = endpoint_row(p, r, seed, freeze_sha)
            per_seed.append(row)
            group.append(row)
            if r is not None:
                from_history = statistics.mean(
                    x for client in r["history_AP"][-1] for x in client
                )
                history_error = max(history_error, abs(from_history - r["metrics"][0]))
        summaries.append(summarize(p, group))
        for candidate in p["candidate_levels"]:
            matches = exact_by_seed(formal, candidate["settings"], EVAL_SEEDS)
            if matches is None:
                raise ValueError(
                    "Descriptive exact-match grid missing " + repr(candidate)
                )
            for r in matches:
                v = r["metrics"]
                F = candidate["F_scale"]
                all_budget.append(
                    dict(
                        dataset=display(p["dataset"]),
                        profile=p["profile"],
                        method=p["method"],
                        seed=r["setting"]["seed"],
                        F_scale=F,
                        frozen_selected_F=p["selected_F"],
                        is_selected_budget=F == p["selected_F"],
                        Qstar=p["Qstar"],
                        run_id=r["run_id"],
                        source_experiment=r["setting"]["experiment"],
                        SeenAP=v[0],
                        pass_Qstar=v[0] >= p["Qstar"],
                        observed_GFLOP=v[15],
                        observed_online_time_s=v[18],
                        observed_energy_J=v[11],
                        policy_use="descriptive_all_budget_readout_not_used_for_F_selection",
                        DEV_policy_sha256=freeze_sha,
                    )
                )
    assert len(per_seed) == 180 and len(summaries) == 36 and len(all_budget) == 540

    dev_support = json.loads((OUT / freeze["DEV_support_file"]).read_bytes())
    dev_index = {r["run_id"]: r for r in dev_support}
    errors = []
    for t in freeze["targets"]:
        values = [dev_index[rid]["metrics"][0] for rid in t["DEV_support_run_ids"]]
        if statistics.mean(values) != t["Qstar"]:
            errors.append("Qstar DEV reconstruction")
    for p in freeze["policies"]:
        valid = []
        for c in p["candidate_levels"]:
            vals = [dev_index[rid]["metrics"][0] for rid in c["DEV_support_run_ids"]]
            if statistics.mean(vals) != c["mean_DEV_SeenAP"]:
                errors.append("F candidate DEV reconstruction")
            if statistics.mean(vals) >= p["Qstar"]:
                valid.append(c["F_scale"])
        expected = min(valid) if valid else None
        if expected != p["selected_F"]:
            errors.append("Frozen F not reconstructed solely from DEV")
    for row in per_seed:
        if not row["formal_evaluated"]:
            if not (
                row["pass_target"] is None
                and row["SeenAP"] is None
                and row["selected_F"] is None
                and row["observed_GFLOP"] is None
            ):
                errors.append("Null policy slot changed to an outcome")
        elif row["pass_target"] != (row["SeenAP"] >= row["Qstar"]):
            errors.append("Endpoint pass mismatch")
    for p, s in zip(freeze["policies"], summaries):
        if p["selected_F"] is None:
            if (
                s["formal_success_rate"] is not None
                or s["evaluated_denominator"] != 0
                or s["formal_success_denominator"] is not None
            ):
                errors.append("Unselected policy has non-null success rate")
        elif s["evaluated_denominator"] != 5 or s["success_n"] + s["failure_n"] != 5:
            errors.append("Evaluated denominator mismatch")
    if history_error > 6e-8:
        errors.append("SeenAP history reconstruction")
    freeze_path = OUT / "dev_policy.json"
    if sha(freeze_path) != freeze_sha:
        errors.append("DEV policy mutated during formal summary")
    if sha(DATA / "results.json") != registry_hash:
        errors.append("Registry changed")
    dump_csv(OUT / "quality_by_seed.csv", per_seed)
    dump_csv(OUT / "quality_summary.csv", summaries)
    dump_csv(OUT / "budget_endpoints.csv", all_budget)
    actual = [r for r in per_seed if r["formal_evaluated"]]
    cirafail = [
        {
            k: r[k]
            for k in (
                "dataset",
                "profile",
                "seed",
                "formal_run_id",
                "Qstar",
                "SeenAP",
                "margin_pp",
                "selected_F",
            )
        }
        for r in actual
        if r["method"] == "CIRA" and not r["pass_target"]
    ]
    counts = dict(
        methods=6,
        cells=6,
        policy_cells=36,
        planned_seed_slots=180,
        evaluated_policy_cells=sum(s["evaluated_denominator"] == 5 for s in summaries),
        actual_formal_readouts=len(actual),
        DEV_no_qualifying_policy_cells=sum(
            p["selected_F"] is None for p in freeze["policies"]
        ),
        DEV_no_qualifying_seed_slots=180 - len(actual),
        formal_successes=sum(r["pass_target"] for r in actual),
        formal_failures=sum(not r["pass_target"] for r in actual),
        descriptive_all_F_rows=540,
        CIRA_formal_failures=len(cirafail),
    )
    report = dict(
        status="PASS_FROZEN_DEV_FORMAL_QUALITY_SUMMARY"
        if not errors
        else "FAIL_FROZEN_DEV_FORMAL_QUALITY_SUMMARY",
        errors=errors,
        scope="Six fixed-reference classroom methods only; not all fourteen methods.",
        model_training_executed=False,
        source_data_modified=False,
        formal_scores_used_for_selection=False,
        native_DEV_lock=False,
        DEV_policy_sha256=freeze_sha,
        source_registry_sha256=registry_hash,
        counts=counts,
        max_formal_SeenAP_history_error=history_error,
        CIRA_test_nonattainment_retained=cirafail,
        output_sha256={
            name: sha(OUT / name)
            for name in [
                "dev_policy.json",
                "dev_support.json",
                "quality_by_seed.csv",
                "quality_summary.csv",
                "budget_endpoints.csv",
            ]
        },
        interpretation=[
            "The policy is frozen retrospectively for this classroom analysis, not a prior native devlock.",
            "The five evaluation seeds are the denominator when a DEV-qualified budget exists. No-budget slots retain null and are not failed observed seeds.",
            "Failures retain all observed endpoint costs. Their target-qualifying endpoint cost is unknown/null; no replacement F is chosen on formal scores.",
            "Costs describe the selected-budget endpoint, not first time/cost to hit the target; no survival/right-censoring analysis is claimed.",
            "All-F formal results are descriptive only and never feed budget selection.",
            "GFLOP/energy/online time are modeled classroom quantities, not measured device costs.",
        ],
    )
    (OUT / "quality_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = "Study summary.\n"
    md += "Study summary" + json.dumps(counts, ensure_ascii=False) + "\n\n"
    md += "Study summary.\n"
    md += "Study summary.\n"
    md += "Study summary" + json.dumps(cirafail, ensure_ascii=False) + "\n\n"
    md += "Study summary.\n"
    (OUT / "quality_summary.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": counts,
                "CIRA_failures": cirafail,
                "DEV_policy_sha256": freeze_sha,
            },
            ensure_ascii=False,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
