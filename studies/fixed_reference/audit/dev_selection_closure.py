"""Read-only DEV ranking and setting-availability audit; never choose on T."""

from pathlib import Path
from collections import Counter, defaultdict
import csv, hashlib, json, math, statistics

DATA = Path("data/data")
PRIOR = Path("data/_________.json")
MANUSCRIPT = Path("data/S03_comparisons_statistics.tex")
OUT = Path(__file__).resolve().parent
IGNORE = {"case_id", "matched_case_id", "experiment", "condition", "block", "seed"}
SERIAL_ORDER = ["cfg25_collab75", "cfg50_collab50", "cfg75_collab25", "collab50_cfg50"]
Q_ELIGIBLE = {f"Q{i}" for i in range(7)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sig(setting):
    return json.dumps(
        {k: v for k, v in setting.items() if k not in IGNORE}, sort_keys=True
    )


def cell(setting):
    return setting["dataset"], setting["profile"]


def config(setting):
    return {k: v for k, v in setting.items() if k not in IGNORE}


def display(name):
    return name.replace("ControlledNATIVE_RUN", "ControlledFog")


def dump_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(g):
    return {
        "n": len(g),
        "seeds": sorted({r["setting"]["seed"] for r in g}),
        "mean_SeenAP": statistics.mean(r["metrics"][0] for r in g),
        "mean_GFLOP": statistics.mean(r["metrics"][15] for r in g),
        "mean_communication_MiB": statistics.mean(r["metrics"][17] for r in g),
        "run_ids": [r["run_id"] for r in g],
        "settings": [config(r["setting"]) for r in g[:1]],
    }


def available(formal, setting):
    found = [r for r in formal if sig(r["setting"]) == sig(setting)]
    return {
        "record_count": len(found),
        "distinct_evaluation_seeds": sorted({r["setting"]["seed"] for r in found}),
        "experiments": dict(Counter(r["setting"]["experiment"] for r in found)),
        "run_ids": [r["run_id"] for r in found],
        "all_five_evaluation_seeds_present": {r["setting"]["seed"] for r in found}
        == set(range(1001, 1006)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [DATA / "results.json", DATA / "study_cases.json", PRIOR, MANUSCRIPT]
    hashes = {str(p): sha(p) for p in inputs}
    assert (
        hashes[str(MANUSCRIPT)]
        == "d6f5c2a5fa59981cc8da174539fafdd71726b3647a4b137050171ef610970051"
    )
    records = json.loads((DATA / "results.json").read_bytes())
    prior = json.loads(PRIOR.read_bytes())
    formal = [
        r
        for r in records
        if r["setting"]["block"] == "evaluation"
        and r["setting"]["seed"] in range(1001, 1006)
    ]
    dev = [r for r in records if r["setting"]["experiment"] in {"D01", "D02", "D03"}]
    assert len(dev) == 378 and all(r["setting"]["block"] == "development" for r in dev)
    errors = []
    history_errors = []
    for r in dev:
        h = r["history_AP"][-1]
        val = statistics.mean(v for client in h for v in client)
        history_errors.append(abs(val - r["metrics"][0]))
        if not math.isclose(val, r["metrics"][0], abs_tol=6e-8, rel_tol=0):
            errors.append("DEV SeenAP does not reconstruct " + r["run_id"])

    by_exp = defaultdict(list)
    for r in dev:
        by_exp[r["setting"]["experiment"]].append(r)
    ranks = []
    local = []
    serial = []
    for exp, output in [("D01", local), ("D03", serial)]:
        by_cell = defaultdict(lambda: defaultdict(list))
        for r in by_exp[exp]:
            by_cell[cell(r["setting"])][r["setting"]["condition"]].append(r)
        for (ds, pr), grid in sorted(by_cell.items()):
            summaries = {k: summarize(v) for k, v in grid.items()}
            for v in summaries.values():
                assert v["n"] == 3 and v["seeds"] == [9001, 9002, 9003]
            if exp == "D01":
                order = sorted(
                    summaries,
                    key=lambda k: (
                        -summaries[k]["mean_SeenAP"],
                        summaries[k]["mean_GFLOP"],
                        summaries[k]["mean_communication_MiB"],
                        k,
                    ),
                )
                main_method = "Local"
            else:
                order = sorted(
                    summaries,
                    key=lambda k: (-summaries[k]["mean_SeenAP"], SERIAL_ORDER.index(k)),
                )
                main_method = "Serial"
            winner = order[0]
            summary = summaries[winner]
            chosen = grid[winner][0]["setting"]
            main = [
                r
                for r in formal
                if r["setting"]["experiment"] == "N01"
                and cell(r["setting"]) == (ds, pr)
                and r["setting"]["method"] == main_method
            ]
            assert len(main) == 5 and len({sig(r["setting"]) for r in main}) == 1
            main_setting = main[0]["setting"]
            matching_candidate_ids = [
                k for k, g in grid.items() if sig(g[0]["setting"]) == sig(main_setting)
            ]
            gap = summary["mean_SeenAP"] - summaries[order[1]]["mean_SeenAP"]
            tie_groups = defaultdict(list)
            for k, v in summaries.items():
                tie_groups[v["mean_SeenAP"]].append(k)
            row = {
                "dataset": ds,
                "profile": pr,
                "experiment": exp,
                "DEV_winner": winner,
                "DEV_mean_SeenAP": summary["mean_SeenAP"],
                "runner_up": order[1],
                "winner_gap_pp": gap,
                "winner_unique": gap > 0,
                "winner_settings": config(chosen),
                "main_settings": config(main_setting),
                "main_candidate_matches": matching_candidate_ids,
                "main_matches_DEV_winner": sig(chosen) == sig(main_setting),
                "main_run_ids": [r["run_id"] for r in main],
                "DEV_run_ids": summary["run_ids"],
                "matching_formal_available": available(formal, chosen),
                "all_candidate_summaries": summaries,
                "exact_mean_ties": [v for v in tie_groups.values() if len(v) > 1],
            }
            output.append(row)
            for candidate in order:
                q = summaries[candidate]
                ranks.append(
                    dict(
                        experiment=exp,
                        dataset=display(ds),
                        profile=pr,
                        candidate=candidate,
                        eligible=True,
                        n=q["n"],
                        seeds="9001,9002,9003",
                        macro_SeenAP=q["mean_SeenAP"],
                        rank=order.index(candidate) + 1,
                        GFLOP=q["mean_GFLOP"],
                        communication_MiB=q["mean_communication_MiB"],
                    )
                )
    qgrid = defaultdict(list)
    for r in by_exp["D02"]:
        qgrid[r["setting"]["condition"]].append(r)
    qsummaries = {}
    for q, g in sorted(qgrid.items()):
        cells = defaultdict(list)
        for r in g:
            cells[cell(r["setting"])].append(r)
        assert len(cells) == 6 and len(g) == 18
        for v in cells.values():
            assert sorted(r["setting"]["seed"] for r in v) == [9001, 9002, 9003]
        m = statistics.mean(
            statistics.mean(r["metrics"][0] for r in v) for v in cells.values()
        )
        qsum = summarize(g)
        qsum["mean_SeenAP"] = m
        qsum["eligible"] = q in Q_ELIGIBLE
        qsum["settings_per_cell"] = {
            ds + "/" + pr: config(v[0]["setting"])
            for (ds, pr), v in sorted(cells.items())
        }
        qsum["matching_formal_per_cell"] = {
            ds + "/" + pr: available(formal, v[0]["setting"])
            for (ds, pr), v in sorted(cells.items())
        }
        qsummaries[q] = qsum
        if not math.isclose(m, prior["D02_means"][q], abs_tol=1e-9, rel_tol=0):
            errors.append("Prior Q mean mismatch " + q)
    qorder = sorted(
        Q_ELIGIBLE,
        key=lambda q: (-qsummaries[q]["mean_SeenAP"], qsummaries[q]["mean_GFLOP"], q),
    )
    for q, qsum in qsummaries.items():
        ranks.append(
            dict(
                experiment="D02",
                dataset="six-cell macro",
                profile="three-profile macro",
                candidate=q,
                eligible=q in Q_ELIGIBLE,
                n=18,
                seeds="9001,9002,9003",
                macro_SeenAP=qsum["mean_SeenAP"],
                rank=qorder.index(q) + 1 if q in Q_ELIGIBLE else "",
                GFLOP=qsum["mean_GFLOP"],
                communication_MiB=qsum["mean_communication_MiB"],
            )
        )
    assert qorder[0] == "Q6"
    qmain = []
    for ds, pr in sorted({cell(r["setting"]) for r in by_exp["D02"]}):
        m = [
            r
            for r in formal
            if r["setting"]["experiment"] == "N01"
            and cell(r["setting"]) == (ds, pr)
            and r["setting"]["method"] == "CIRA"
        ]
        d = [r for r in by_exp["D02"] if cell(r["setting"]) == (ds, pr)]
        match = [
            q
            for q in qgrid
            if sig(next(r["setting"] for r in d if r["setting"]["condition"] == q))
            == sig(m[0]["setting"])
        ]
        qmain.append(
            dict(
                dataset=ds,
                profile=pr,
                declared_main_Q_matches=match,
                main_settings=config(m[0]["setting"]),
                main_run_ids=[r["run_id"] for r in m],
            )
        )
    selected_eta = next(
        r for r in by_exp["D02"] if r["setting"]["condition"] == qorder[0]
    )["setting"]["eta_beta"]
    selected_local = {(r["dataset"], r["profile"]): r["winner_settings"] for r in local}
    serial_chain = []
    for row in serial:
        ds, pr = row["dataset"], row["profile"]
        base = dict(row["winner_settings"])
        beta = dict(base, eta_beta=selected_eta)
        strict = dict(
            beta, z=selected_local[ds, pr]["z"], h=selected_local[ds, pr]["h"]
        )
        serial_chain.append(
            dict(
                dataset=ds,
                profile=pr,
                existing_D03_eta_beta=base["eta_beta"],
                copied_Q6_eta_beta=selected_eta,
                D03_candidates_at_Q6_eta=sum(
                    r["setting"]["eta_beta"] == selected_eta
                    and cell(r["setting"]) == (ds, pr)
                    for r in by_exp["D03"]
                ),
                formal_selected_rule_at_existing_beta=available(formal, base),
                formal_selected_rule_at_Q6_beta=available(formal, beta),
                formal_selected_rule_at_Q6_beta_and_selected_Local=available(
                    formal, strict
                ),
            )
        )
    for exp, derived in [("D01", local), ("D03", serial)]:
        old = {(r["dataset"], r["profile"]): r for r in prior[exp]}
        for r in derived:
            p = old[r["dataset"], r["profile"]]
            if r["DEV_winner"] != p["DEV_best"] or not math.isclose(
                r["DEV_mean_SeenAP"], p["DEV_best_mean"], abs_tol=1e-9, rel_tol=0
            ):
                errors.append("Prior winner mismatch " + exp)
    matches = {
        "Local_main_matches_DEV": sum(r["main_matches_DEV_winner"] for r in local),
        "Local_main_mismatches": sum(not r["main_matches_DEV_winner"] for r in local),
        "Serial_main_matches_conditional_DEV": sum(
            r["main_matches_DEV_winner"] for r in serial
        ),
        "Serial_main_mismatches": sum(not r["main_matches_DEV_winner"] for r in serial),
    }
    statement = (
        "The current numerical main comparison uses fixed declared reference settings: Local Z24-H3 in five cells and Z33-H2 for ACDC/YOLO11n; "
        "the recorded Serial schedules in the per-cell table; and CIRA Q2 (lambda=10, kappa=0.01, eta_beta=0.01, M=64, E=65). "
        "The DEV grids are reported descriptively and were not used to retroactively relabel these main results as DEV-selected. "
        "Their recomputed numerical ranks favor N-L06, Q6, and cfg25_collab75 conditional on the existing beta learning rate. "
        "A complete native DEV-selected comparison and its lock remain pending."
    )
    report = dict(
        status="PASS_DEV_SETTING_AUDIT" if not errors else "FAIL_DEV_SETTING_AUDIT",
        errors=errors,
        source_hashes=hashes,
        source_data_modified=False,
        model_training_executed=False,
        scope="Retrospective descriptive ranking of disclosed classroom DEV records; no T scores used for selection; not a native DEV lock.",
        manuscript_lines={
            "current_references": 31,
            "native_DEV_procedure": 66,
            "Serial_tie_order": 149,
            "Q_eligibility": 169,
        },
        native_qualification_or_devlock_present=False,
        DEV_records=len(dev),
        DEV_counts=dict(Counter(r["setting"]["experiment"] for r in dev)),
        DEV_seeds=[9001, 9002, 9003],
        formal_seed_labels=[1001, 1002, 1003, 1004, 1005],
        score="Final equal-client/equal-domain SeenAP reconstructed as mean of16 final history cells, then mean across three DEV seeds and six equally weighted cells for Q.",
        max_SeenAP_history_reconstruction_error=max(history_errors),
        eligibility={
            "CIRA_eligible": sorted(Q_ELIGIBLE),
            "CIRA_excluded_M_diagnostics": ["Q7", "Q8"],
            "Local_candidates": [f"N-L{i:02d}" for i in range(1, 9)],
            "Serial_order": SERIAL_ORDER,
        },
        tie_rules={
            "Local": "higher SeenAP, lower computation, lower communication, then fixed ID",
            "CIRA": "higher six-cell macro SeenAP, lower computation, then fixed ID",
            "Serial": "higher per-cell DEV SeenAP, then listed rule order",
            "winner_ties_present": False,
            "tie_tolerance": "exact full-precision stored-value tie; no invented practical threshold",
        },
        Local=local,
        Serial=serial,
        CIRA=dict(
            winner=qorder[0],
            eligible_order=qorder,
            points=qsummaries,
            main=qmain,
            Q6_minus_Q2_DEV_pp=qsummaries["Q6"]["mean_SeenAP"]
            - qsummaries["Q2"]["mean_SeenAP"],
        ),
        main_match_counts=matches,
        Serial_dependent_selection_chain=serial_chain,
        recommended_option="Preserve primary results and explicitly use fixed-reference identity; retain DEV ranks as descriptive diagnostics.",
        no_rerun_resolution_complete="Content/setting identity can be reconciled now. A full native DEV-selected execution claim cannot be closed by renaming existing results.",
        proposed_wording=statement,
        alternative_DEV_selected_option={
            "CIRA": "Q6 has30 exact-setting formal records in N03; may be reported as a separate sensitivity result without replacing the frozen primary.",
            "Local": "N-L06 has no exact-setting formal Local records across evaluation seeds1001-1005.",
            "Serial": "At beta=.01 the cfg25 rule has30 exact-setting formal records (20N11+10N01); at the Q6-copied beta=.03 both DEV rule selection and formal matching records are absent.",
            "restriction": "Do not choose the interpretation using T rankings; do not copy test values from a nonmatching configuration.",
        },
    )
    assert {str(p): sha(p) for p in inputs} == hashes
    (OUT / "dev_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dump_csv(OUT / "dev_ranking.csv", ranks)
    table = []
    for label, group in [("Local", local), ("Serial", serial)]:
        for r in group:
            table.append(
                dict(
                    method=label,
                    dataset=display(r["dataset"]),
                    profile=r["profile"],
                    DEV_winner=r["DEV_winner"],
                    DEV_mean_SeenAP=r["DEV_mean_SeenAP"],
                    main_reference=";".join(r["main_candidate_matches"]),
                    main_matches_DEV=r["main_matches_DEV_winner"],
                    matching_formal_count=r["matching_formal_available"][
                        "record_count"
                    ],
                    matching_formal_experiments=json.dumps(
                        r["matching_formal_available"]["experiments"],
                        ensure_ascii=False,
                    ),
                )
            )
    dump_csv(OUT / "dev_comparison.csv", table)
    md = f"""'Study summary.\n'{qsummaries["Q6"]["mean_SeenAP"]:.10f}，Q2={qsummaries["Q2"]["mean_SeenAP"]:.10f}'Study summary'{report["CIRA"]["Q6_minus_Q2_DEV_pp"]:.10f}'Study summary.\n'{statement}'Study summary.\n'"""
    (OUT / "dev_selection.md").write_text(md, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "CIRA_DEV_winner": qorder[0],
                "Q6_macro_SeenAP": qsummaries["Q6"]["mean_SeenAP"],
                "Q2_macro_SeenAP": qsummaries["Q2"]["mean_SeenAP"],
                "main_match_counts": matches,
                "errors": errors,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
