"""Exact-setting DEV target/quality-cost availability; no formal-score selection."""

from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, statistics
from dev_selection_closure import DATA, OUT, sig, config, sha, display, dump_csv

METHODS = ["CIRA", "Serial", "Fixed-collab", "Local-NAS-HPO", "DeModCL", "DSpodFL"]
LEVELS = [0.5, 1.0, 2.0]
DEV_SEEDS = {9001, 9002, 9003}
EVAL_SEEDS = {1001, 1002, 1003, 1004, 1005}


def match(pool, setting, seeds):
    found = [
        r
        for r in pool
        if sig(r["setting"]) == sig(setting) and r["setting"]["seed"] in seeds
    ]
    per_seed = defaultdict(list)
    for r in found:
        per_seed[r["setting"]["seed"]].append(r)
    conflicts = []
    for seed, g in per_seed.items():
        if len({tuple(r["metrics"]) for r in g}) > 1:
            conflicts.append(seed)
    complete = set(per_seed) == seeds and not conflicts
    chosen = (
        [
            sorted(per_seed[seed], key=lambda r: r["run_id"])[0]
            for seed in sorted(per_seed)
        ]
        if not conflicts
        else []
    )
    return dict(
        record_count=len(found),
        independent_seed_count=len(per_seed),
        seeds=sorted(per_seed),
        complete=complete,
        conflicts=conflicts,
        source_experiments=dict(Counter(r["setting"]["experiment"] for r in found)),
        run_ids=[r["run_id"] for r in found],
        chosen_seed_run_ids=[r["run_id"] for r in chosen],
        mean_DEV_SeenAP=statistics.mean(r["metrics"][0] for r in chosen)
        if complete and seeds == DEV_SEEDS
        else None,
        sample_DEV_SD=statistics.stdev(r["metrics"][0] for r in chosen)
        if complete and seeds == DEV_SEEDS
        else None,
        mean_modeled_GFLOP=statistics.mean(r["metrics"][15] for r in chosen)
        if complete and seeds == DEV_SEEDS
        else None,
        mean_modeled_time_s=statistics.mean(r["metrics"][18] for r in chosen)
        if complete and seeds == DEV_SEEDS
        else None,
    )


def main():
    data_hash = sha(DATA / "results.json")
    records = json.loads((DATA / "results.json").read_bytes())
    dev = [r for r in records if r["setting"]["block"] == "development"]
    formal = [r for r in records if r["setting"]["block"] == "evaluation"]
    mains = defaultdict(dict)
    for r in formal:
        s = r["setting"]
        if s["experiment"] == "N01" and s["method"] in METHODS:
            mains[s["dataset"], s["profile"]].setdefault(s["method"], s)
    assert len(mains) == 6 and all(set(g) == set(METHODS) for g in mains.values())
    targets = []
    grid = []
    choices = []
    errors = []
    for (ds, pr), settings in sorted(mains.items()):
        target_setting = dict(settings["Serial"], F_scale=1.0)
        target_match = match(dev, target_setting, DEV_SEEDS)
        qstar = target_match["mean_DEV_SeenAP"] if target_match["complete"] else None
        targets.append(
            dict(
                dataset=ds,
                profile=pr,
                target_kind="retrospective_fixed_reference_DEV_Serial_F1",
                target_qstar=qstar,
                target_settings=config(target_setting),
                availability=target_match,
                original_native_DEV_lock=False,
                reason="three exact-setting DEV seeds available"
                if qstar is not None
                else "missing or conflicted exact-setting DEV Serial F1",
            )
        )
        for m in METHODS:
            rows = []
            for F in LEVELS:
                setting = dict(settings[m], F_scale=F)
                dm = match(dev, setting, DEV_SEEDS)
                tm = match(formal, setting, EVAL_SEEDS)
                d5 = [
                    r
                    for r in dev
                    if r["setting"]["experiment"] == "D05"
                    and r["setting"]["method"] == m
                    and r["setting"]["dataset"] == ds
                    and r["setting"]["profile"] == pr
                    and r["setting"]["F_scale"] == F
                ]
                mismatches = []
                for r in d5:
                    diff = {
                        k: {"expected": setting.get(k), "observed": r["setting"].get(k)}
                        for k in setting
                        if k
                        not in {
                            "case_id",
                            "matched_case_id",
                            "experiment",
                            "condition",
                            "block",
                            "seed",
                        }
                        and setting.get(k) != r["setting"].get(k)
                    }
                    if diff:
                        mismatches.append({"run_id": r["run_id"], "diff": diff})
                qualifies = (
                    (dm["mean_DEV_SeenAP"] >= qstar)
                    if qstar is not None and dm["complete"]
                    else None
                )
                row = dict(
                    dataset=ds,
                    profile=pr,
                    method=m,
                    F_scale=F,
                    main_fixed_settings_except_F=config(setting),
                    target_qstar=qstar,
                    DEV=dm,
                    formal_availability=tm,
                    D05_direct_records=len(d5),
                    D05_exact_records=len(d5) - len(mismatches),
                    D05_mismatches=mismatches,
                    DEV_reuse_anchor=not bool(d5) and dm["complete"],
                    qualifies_DEV_target=qualifies,
                    unresolved_reason=None
                    if qstar is not None and dm["complete"]
                    else (
                        "Qstar_unavailable"
                        if qstar is None
                        else "missing_or_conflicted_DEV_setting"
                    ),
                )
                rows.append(row)
                grid.append(row)
            eligible = [r for r in rows if r["qualifies_DEV_target"] is True]
            choice = min(eligible, key=lambda r: r["F_scale"]) if eligible else None
            complete_grid = all(r["DEV"]["complete"] for r in rows)
            choice_known = qstar is not None and complete_grid
            choices.append(
                dict(
                    dataset=ds,
                    profile=pr,
                    method=m,
                    target_qstar=qstar,
                    selected_F=choice["F_scale"] if choice_known and choice else None,
                    selection_status=(
                        "attained_in_DEV_grid"
                        if choice
                        else "not_attained_at_any_DEV_F"
                    )
                    if choice_known
                    else "inestimable",
                    selection_rule="smallest F in{0.5,1,2} with mean exact-setting three-seed DEV SeenAP >= fixed-reference Serial F1 target",
                    matched_formal_availability=choice["formal_availability"]
                    if choice_known and choice
                    else None,
                    can_estimate_five_seed_formal_attainment=bool(
                        choice_known
                        and choice
                        and choice["formal_availability"]["complete"]
                    ),
                    formal_scores_used_to_choose=False,
                    caveat="The target/choice is a retrospective classroom calculation, not a timestamped native DEV lock.",
                )
            )
    counts = dict(
        targets_complete=sum(r["target_qstar"] is not None for r in targets),
        targets_total=len(targets),
        grid_cells=len(grid),
        DEV_grid_complete=sum(r["DEV"]["complete"] for r in grid),
        formal_grid_complete=sum(r["formal_availability"]["complete"] for r in grid),
        D05_direct_exact_records=sum(r["D05_exact_records"] for r in grid),
        D05_direct_mismatched_records=sum(len(r["D05_mismatches"]) for r in grid),
        reused_F1_DEV_settings=sum(r["DEV_reuse_anchor"] for r in grid),
        reused_F1_DEV_records=sum(
            r["DEV"]["record_count"] for r in grid if r["DEV_reuse_anchor"]
        ),
        method_cell_choices=len(choices),
        attained_choices=sum(
            r["selection_status"] == "attained_in_DEV_grid" for r in choices
        ),
        not_attained_choices=sum(
            r["selection_status"] == "not_attained_at_any_DEV_F" for r in choices
        ),
        inestimable_choices=sum(
            r["selection_status"] == "inestimable" for r in choices
        ),
        selected_F_formal_complete=sum(
            r["can_estimate_five_seed_formal_attainment"] for r in choices
        ),
    )
    if counts["D05_direct_mismatched_records"]:
        errors.append("SomeD05records do not match fixed main parameters exceptF")
    report = dict(
        status="PASS_FIXED_REFERENCE_QUALITY_AVAILABILITY"
        if not errors
        else "FAIL_FIXED_REFERENCE_QUALITY_AVAILABILITY",
        errors=errors,
        source_registry_sha256=data_hash,
        source_data_modified=False,
        model_training_executed=False,
        scope="Numerical classroom fixed-reference target and exact-setting availability; native DEV-selected lock remains pending.",
        formal_scores_used_for_target_or_F_selection=False,
        matching_rule="All setting fields exceptcase_id/matched_case_id/experiment/condition/block/seed equal; F_scale varies only at the prespecified grid level.",
        counts=counts,
        targets=targets,
        grid=grid,
        choices=choices,
        native_DEV_selected_chain_not_closed_reason="The global eligible controller winnerQ6 haseta_beta=.03; existingSerialD03 uses.01 and selectedLocalN-L06 formal coverage is absent. These fixed-reference calculations do not close that different chain.",
        interpretation="An unmatchedorunattainedmethod remainsnull/unattained. Existing formal scores may be summarized after choosingF solely fromDEV; do not redefineQstar fromformal results or a differentSerial configuration.",
    )
    assert sha(DATA / "results.json") == data_hash
    (OUT / "quality_availability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csvgrid = []
    for r in grid:
        csvgrid.append(
            dict(
                dataset=display(r["dataset"]),
                profile=r["profile"],
                method=r["method"],
                F_scale=r["F_scale"],
                target_qstar=r["target_qstar"],
                DEV_SeenAP=r["DEV"]["mean_DEV_SeenAP"],
                DEV_SD=r["DEV"]["sample_DEV_SD"],
                DEV_independent_seeds=r["DEV"]["independent_seed_count"],
                DEV_complete=r["DEV"]["complete"],
                DEV_source_experiments=json.dumps(
                    r["DEV"]["source_experiments"], ensure_ascii=False
                ),
                formal_independent_seeds=r["formal_availability"][
                    "independent_seed_count"
                ],
                formal_complete=r["formal_availability"]["complete"],
                formal_source_experiments=json.dumps(
                    r["formal_availability"]["source_experiments"], ensure_ascii=False
                ),
                D05_exact_records=r["D05_exact_records"],
                reused_DEV_anchor=r["DEV_reuse_anchor"],
                qualifies_DEV_target=r["qualifies_DEV_target"],
                unresolved_reason=r["unresolved_reason"],
            )
        )
    dump_csv(OUT / "quality_grid.csv", csvgrid)
    dump_csv(
        OUT / "dev_targets.csv",
        [
            dict(
                dataset=display(r["dataset"]),
                profile=r["profile"],
                target_qstar=r["target_qstar"],
                DEV_seeds=",".join(map(str, r["availability"]["seeds"])),
                source_experiments=json.dumps(r["availability"]["source_experiments"]),
                native_DEV_lock=False,
                reason=r["reason"],
            )
            for r in targets
        ],
    )
    dump_csv(
        OUT / "dev_budgets.csv",
        [
            {k: v for k, v in r.items() if k != "matched_formal_availability"}
            for r in choices
        ],
    )
    md = "Study summary.\n"
    md += "\n".join(
        f"- {display(r['dataset'])}/{r['profile']}：Q*={r['target_qstar'] if r['target_qstar'] is not None else 'null'}'Study summary'{r['availability']['source_experiments']}'Study summary'{r['availability']['seeds']}。"
        for r in targets
    )
    md += "Study summary.\n" + json.dumps(counts, ensure_ascii=False) + "\n\n"
    md += "Study summary.\n"
    (OUT / "quality_availability.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": counts,
                "targets": [
                    {k: r[k] for k in ("dataset", "profile", "target_qstar")}
                    for r in targets
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
