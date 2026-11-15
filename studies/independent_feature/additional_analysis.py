"""Cost-selected trajectory comparisons; no score-selected checkpoints."""

from pathlib import Path
import argparse, json, csv
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "output")
    a = p.parse_args()
    out = a.out
    records = json.loads((out / "results.json").read_bytes())
    groups = {}
    for r in records:
        c = r["setting"]
        method = (
            "Full"
            if c["experiment"] == "N01" and c["method"] == "CIRA"
            else c["method"]
            if c["experiment"] == "N14"
            and c["method"] in ["No-Refine", "No-cross-family-parent"]
            else None
        )
        if method:
            groups.setdefault((c["dataset"], c["profile"], c["seed"]), {})[method] = r
    rows = []
    matched = 0
    for key, g in sorted(groups.items()):
        if len(g) != 3:
            continue
        cap = min(r["metrics"][15] for r in g.values())
        block = []
        for method in ["Full", "No-Refine", "No-cross-family-parent"]:
            r = g[method]
            with np.load(out / r["trajectory"]) as t:
                eligible = np.flatnonzero(
                    t["AP_evaluated"]
                    & (t["round"] >= r["setting"]["round_budget"] * 0.75)
                    & (t["values"][:, 15] <= cap + 1e-12)
                )
                if len(eligible):
                    j = int(eligible[-1])
                    rr = int(t["round"][j])
                    cost = float(t["values"][j, 15])
                    ap = float(t["values"][j, 0])
                else:
                    rr = cost = ap = None
            block.append(
                dict(
                    dataset=key[0],
                    profile=key[1],
                    seed=key[2],
                    method=method,
                    run_id=r["run_id"],
                    common_cost_ceiling_GFLOP=cap,
                    round=rr,
                    actual_online_GFLOP=cost,
                    AllDomainAP=ap,
                )
            )
        ok = all(
            x["actual_online_GFLOP"] is not None
            and x["actual_online_GFLOP"] >= 0.95 * cap
            for x in block
        )
        matched += int(ok)
        for x in block:
            x["status"] = (
                "matched_within_5pct"
                if ok
                else "no_stage4_observed_point_within_ceiling"
                if x["round"] is None
                else "cost_or_coverage_gap_exceeds_rule"
            )
            rows.append(x)
    with (out / "matched_work.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = dict(
        matched_seed_groups_within_5pct=matched,
        planned_seed_groups=len(groups),
        rows=len(rows),
        selection_uses_AP=False,
        no_interpolation=True,
        stage4_only=False,
        stage_window="third-stage completed boundary and subsequent fourth-stage points; not necessarily equal stage",
        unsupported_groups_retained=len(groups) - matched,
    )
    (out / "matched_work_coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
