from pathlib import Path
from collections import defaultdict
import csv, json, statistics as st, shutil, hashlib, math

R = Path(__file__).resolve().parents[1]
S = Path("data/recovered_figure_source_data")
D = R / "references/old_prediction_records"
D.mkdir(exist_ok=True)
for name in ["gain_predictions.csv", "recorded_actions.csv"]:
    if not (D / name).is_file():
        shutil.copy2(S / name, D / name)


def read(n):
    with (D / n).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


a = read("gain_predictions.csv")
b = read("recorded_actions.csv")
key = lambda x: tuple(x[k] for k in ["run_id", "client", "stage", "round", "action"])
bm = {key(x): x for x in b}
assert len(bm) == len(b) == len(a) == 7920 and {key(x) for x in a} == set(bm)
groups = defaultdict(list)
for x in a:
    y = bm[key(x)]
    assert abs(float(x["predicted_gain_pp"]) - 100 * float(y["predicted_mu"])) < 1e-10
    assert abs(float(x["observed_gain_pp"]) - float(y["signed_parent_gain_pp"])) < 1e-10
    groups[x["run_id"], x["client"], x["stage"]].append(x)
details = []
run_errors = defaultdict(list)
for context, rows in groups.items():
    rows.sort(key=lambda x: int(bm[key(x)]["sequence_index"]))
    past = []
    for x in rows:
        target = float(x["observed_gain_pp"])
        history = st.mean(past) if past else 0.0
        predictions = {
            "recorded_predictor": float(x["predicted_gain_pp"]),
            "zero": 0.0,
            "past_feedback_mean": history,
        }
        for name, pred in predictions.items():
            err = abs(pred - target)
            run_errors[
                x["dataset"], x["profile"], x["run_id"], int(x["seed"]), name
            ].append(err)
            details.append(
                dict(
                    run_id=x["run_id"],
                    dataset=x["dataset"],
                    profile=x["profile"],
                    seed=int(x["seed"]),
                    client=int(x["client"]),
                    stage=int(x["stage"]),
                    sequence_index=int(bm[key(x)]["sequence_index"]),
                    past_observations=len(past),
                    reference=name,
                    prediction_pp=pred,
                    observed_parent_V_gain_pp=target,
                    absolute_error_pp=err,
                    scope="Study summary",
                )
            )
        past.append(target)
seed_rows = []
for (ds, pr, run, seed, name), err in run_errors.items():
    seed_rows.append(
        dict(
            dataset=ds,
            profile=pr,
            run_id=run,
            seed=seed,
            reference=name,
            executed_actions=len(err),
            MAE_pp=st.mean(err),
        )
    )
conds = defaultdict(list)
for x in seed_rows:
    conds[x["dataset"], x["profile"], x["reference"]].append(x["MAE_pp"])
summary = []
for (ds, pr, name), values in sorted(conds.items()):
    assert len(values) == 5
    m = st.mean(values)
    sd = st.stdev(values)
    ci = 2.7764451051977987 * sd / math.sqrt(5)
    summary.append(
        dict(
            dataset=ds,
            profile=pr,
            reference=name,
            n_seed=5,
            MAE_mean_pp=m,
            MAE_sample_SD=sd,
            CI95_lo=m - ci,
            CI95_hi=m + ci,
            scope="Study summary",
        )
    )


def write(file, rows):
    with (R / "tables" / file).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


write("old_prediction_reference_detail.csv", details)
write("old_prediction_reference_seed.csv", seed_rows)
write("old_prediction_reference_summary.csv", summary)
audit = [
    dict(
        item=k,
        present=sum(bool(x.get(k)) for x in b),
        records=len(b),
        meaning="Study summary",
    )
    for k in ["parent_hash", "child_hash", "retained_hash", "predicted_mu", "priority"]
]
audit += [
    dict(
        item="cross_family_labels",
        present=0,
        records=len(b),
        meaning="Study summary",
    ),
    dict(
        item="released_checkpoint_mapping",
        present=0,
        records=len(b),
        meaning="Study summary",
    ),
]
write("old_parent_field_audit.csv", audit)
p = json.loads((R / "code/payload.json").read_bytes())
for name, file, rows in [
    ("Study summary", "old_prediction_reference_summary.csv", summary),
    ("Study summary", "old_parent_field_audit.csv", audit),
]:
    p["tables"][name] = {
        "header": list(rows[0]),
        "rows": [[r[k] for k in rows[0]] for r in rows],
    }
    if name not in p["order"]:
        p["order"].insert(p["order"].index("Study summary"), name)
index = []
for f in (R / "references").rglob("*"):
    if f.is_file():
        index.append(
            dict(
                path=f.relative_to(R).as_posix(),
                bytes=f.stat().st_size,
                SHA256=hashlib.sha256(f.read_bytes()).hexdigest(),
                scope="Study summary",
            )
        )
write("evidence_aligned.csv", index)
p["tables"]["Study summary"] = {
    "header": list(index[0]),
    "rows": [[r[k] for k in index[0]] for r in index],
}
(R / "code/payload.json").write_text(
    json.dumps(p, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
)
q = dict(
    status="PASS_PAST_ONLY_REFERENCE_RECOMPUTATION",
    executed_proposals=7920,
    joined_recorded_actions=7920,
    contexts=len(groups),
    predictors=3,
    seed_summaries=len(seed_rows),
    condition_summaries=len(summary),
    historical_mean_uses_previous_only=True,
    cold_start_prediction_pp=0,
    predicted_mu_conversion_to_pp=100,
    native_or_strict_policy_validation=False,
)
(R / "qa/prediction_reference_check.json").write_text(
    json.dumps(q, indent=2), encoding="utf-8"
)
print(json.dumps(q), flush=True)
