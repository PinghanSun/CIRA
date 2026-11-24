"""Recompute manuscript summaries using the accompanying numerical records."""

from collections import Counter, defaultdict
from csv import DictReader
from hashlib import sha256
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean, stdev
from math import isclose, sqrt
import json


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def set_data_root(directory):
    global ROOT, DATA
    ROOT = directory.resolve()
    DATA = ROOT / "data"


T975 = {5: 2.776445105, 8: 2.364624252}


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as file:
        return list(DictReader(file))


def key(row, *fields):
    return tuple(row[k] for k in fields)


def group(rows, *fields):
    output = defaultdict(list)
    for row in rows:
        output[key(row, *fields)].append(row)
    return output


def number(row, field):
    return float(row[field])


def paired_interval(values):
    n = len(values)
    avg = mean(values)
    half = T975[n] * stdev(values) / sqrt(n)
    return [round(x, 6) for x in (avg, avg - half, avg + half)]


def rounded(exact, printed, digits=2):
    assert abs(exact - printed) <= 0.5 * 10 ** (-digits) + 1e-8, (exact, printed)


def confirm_interval(summary, published):
    for actual, expected in zip(summary, published, strict=True):
        rounded(actual, expected)


def check_manifest():
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    with (ROOT / "data_catalog.csv").open(encoding="utf-8", newline="") as file:
        for row in DictReader(file):
            assert len(read(row["file"])) == int(row["rows"]), row["file"]


def allocations():
    rows = read("detector_role_assignments.csv")
    expected = {
        ("ACDC", "D"): 1216,
        ("ACDC", "V"): 192,
        ("ACDC", "A"): 192,
        ("ACDC", "T"): 406,
        ("ControlledFog", "D"): 7036,
        ("ControlledFog", "V"): 1232,
        ("ControlledFog", "A"): 1232,
        ("ControlledFog", "T"): 2000,
    }
    seeds = group(rows, "seed")
    assert len(seeds) == 5
    for seed, assigned in seeds.items():
        assert Counter(key(r, "dataset", "role") for r in assigned) == expected, seed
        assert len({key(r, "dataset", "resource_id") for r in assigned}) == len(
            assigned
        )
    field_order = ("seed", "dataset", "client", "stage", "role")
    matched = group([r for r in rows if r["role"] in ("D", "V", "A")], *field_order)
    cells = read("detector_role_class_counts.csv")
    assert len(matched) == len(cells) == 480
    for c in cells:
        assert int(c["RGB"]) == len(matched[key(c, *field_order)])
        assert int(c["noncrowd_boxes"]) == sum(
            int(c[col])
            for col in (
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            )
        )
    return {
        "seeds": list(k[0] for k in sorted(seeds)),
        "role_cells": len(cells),
        "active_assignments_per_seed": sum(expected.values()),
    }


def r1000():
    rows = read("r1000_endpoints.csv")
    by_cell = group(rows, "dataset", "profile", "method")
    assert len(by_cell) == 84 and all(len(g) == 5 for g in by_cell.values())
    for published in read("table2_printed_reference.csv"):
        cell = by_cell[key(published, "dataset", "profile", "method")]
        vals = [number(r, "seen_score_original") for r in cell]
        rounded(mean(vals), number(published, "mean"))
        rounded(stdev(vals), number(published, "sd"))

    summaries = {}
    published = {
        ("ACDC", "YOLOv8n"): (1.87, 1.81, 1.94),
        ("ACDC", "YOLO11n"): (3.06, 2.94, 3.19),
        ("ACDC", "YOLO26n"): (3.93, 3.83, 4.04),
        ("ControlledFog", "YOLOv8n"): (0.78, 0.66, 0.90),
        ("ControlledFog", "YOLO11n"): (2.21, 2.08, 2.35),
        ("ControlledFog", "YOLO26n"): (2.49, 2.35, 2.63),
    }
    for (dataset, profile), summary in published.items():
        full = {
            r["seed"]: number(r, "seen_score_original")
            for r in by_cell[dataset, profile, "CIRA"]
        }
        reference = {
            r["seed"]: number(r, "seen_score_original")
            for r in by_cell[dataset, profile, "pFedDIL"]
        }
        assert full.keys() == reference.keys()
        differences = [full[s] - reference[s] for s in sorted(full)]
        interval = paired_interval(differences)
        confirm_interval(interval, summary)
        summaries[f"{dataset}/{profile}"] = interval
    costs = {r["run_id"]: r for r in read("r1000_ledger.csv")}
    assert costs.keys() == {r["run_id"] for r in rows}
    resources = {}
    for label in ("CIRA", "Serial"):
        cost_rows = [costs[r["run_id"]] for r in rows if r["method"] == label]
        resources[label] = {
            "fit_work": mean(number(c, "fit_work") for c in cost_rows),
            "validation_work": mean(number(c, "V_work") for c in cost_rows),
            "total_work": mean(number(c, "total_work") for c in cost_rows),
            "controller_work": mean(number(c, "controller_work") for c in cost_rows),
            "transfer_MiB": mean(number(c, "transfer_bytes") for c in cost_rows)
            / 2**20,
        }
    assert all(
        isclose(resources[label]["fit_work"], 9040, abs_tol=1e-7) for label in resources
    )
    assert isclose(
        resources["CIRA"]["total_work"] - resources["Serial"]["total_work"],
        2132,
        abs_tol=1e-7,
    )
    rounded(resources["CIRA"]["transfer_MiB"], 28.33)
    rounded(resources["Serial"]["transfer_MiB"], 42.23)

    mapping = read("r1000_label_attribution.csv")
    labels = {"Fixed-collab": "Fixed collab.", "Local-NAS-HPO": "Local NAS/HPO"}
    indexed = {key(r, "dataset", "profile", "seed", "method"): r for r in rows}
    for r in mapping:
        assert r["displayed_method"] == r["ledger_method"]
        assert r["displayed_method"] != r["score_label"]
        label = labels.get(r["displayed_method"], r["displayed_method"])
        published_row = indexed[(*key(r, "dataset", "profile", "seed"), label)]
        assert isclose(
            number(published_row, "seen_score_original"),
            number(r, "seen_score_original"),
            abs_tol=1e-8,
        )
    return {
        "verified_table2_cells": len(by_cell),
        "table_s11": summaries,
        "table4_selected_resources": resources,
        "verified_label_attribution_rows": len(mapping),
    }


def continuation():
    strict = read("r1000_strict_preset_pairs.csv")
    strict_intervals = {}
    for k, g in group(strict, "dataset", "profile", "control").items():
        values = [number(r, "full_score") - number(r, "control_score") for r in g]
        assert all(
            isclose(v, number(r, "full_minus_control"), abs_tol=1e-8)
            for r, v in zip(g, values)
        )
        strict_intervals["/".join(k)] = paired_interval(values)
    assert len(strict_intervals) == 12
    for label, reported in [
        ("Strict No-cross", (1.03, 1.39)),
        ("Strict No-gain", (0.79, 1.06)),
    ]:
        all_means = [v[0] for k, v in strict_intervals.items() if k.endswith(label)]
        rounded(min(all_means), reported[0])
        rounded(max(all_means), reported[1])

    endpoints = read("yolov8n_continuation_preset_pairs.csv")
    reported = {"ACDC": (1.00, 0.54, 1.46), "ControlledFog": (0.83, 0.42, 1.24)}
    paired = {}
    adjunct = {}
    for (name,), rows in group(endpoints, "dataset").items():
        d = [
            number(r, "full_seen_ap_pct") - number(r, "strict_seen_ap_pct")
            for r in rows
        ]
        assert all(
            isclose(v, number(r, "preset_difference_pp"), abs_tol=1e-8)
            for r, v in zip(rows, d)
        )
        paired[name] = paired_interval(d)
        confirm_interval(paired[name], reported[name])
        assert sorted(int(r["seed"]) for r in rows) == list(range(1001, 1009))
        forget = paired_interval([number(r, "forget_difference_pp") for r in rows])
        validations = [
            number(r, "validation_image_evaluation_difference") for r in rows
        ]
        transfer = [number(r, "transfer_difference_bytes") / 2**20 for r in rows]
        assert all(x > 0 for x in validations) and all(x < 0 for x in transfer)
        confirm_interval(
            forget, (-0.04, -0.35, 0.26) if name == "ACDC" else (-0.09, -0.22, 0.05)
        )
        rounded(mean(validations), 172.5 if name == "ACDC" else 1347.5, 1)
        rounded(mean(transfer), -1.90 if name == "ACDC" else -1.77)
        adjunct[name] = {
            "forget_difference_pp": forget,
            "validation_image_evaluation_difference": mean(validations),
            "transfer_difference_mib": mean(transfer),
        }
    events = read("continuation_events.csv")
    counts = read("continuation_event_counts.csv")
    flags = ("retained", "on_deployment_lineage")
    assert len(
        {key(r, "dataset", "seed", "client", "stage", "attempt") for r in events}
    ) == len(events)
    assert len({r["candidate_id"] for r in events}) == len(events)
    for row in events:
        assert row["policy"] == "Full" and row["parent_origin"] == "stage_revision"
        assert row["candidate_id"].startswith("SIM-") and row["parent_id"].startswith(
            "SIM-"
        )
        assert all(row[k] in ("0", "1") for k in flags)
    assert (
        len(counts)
        == len({key(r, "dataset", "seed", "direction") for r in counts})
        == 32
    )
    by_seed = group(events, "dataset", "seed", "direction")
    aggregate = {}
    for r in counts:
        group_rows = by_seed.get(key(r, "dataset", "seed", "direction"), [])
        assert int(r["completed"]) == len(group_rows)
        for flag in flags:
            assert int(r[flag]) == sum(int(a[flag]) for a in group_rows)
    for k, rows in group(events, "dataset", "direction").items():
        aggregate["/".join(k)] = [
            len(rows),
            *(sum(int(r[f]) for r in rows) for f in flags),
        ]
    assert set(by_seed) <= {key(r, "dataset", "seed", "direction") for r in counts}
    totals = Counter(r["direction"] for r in events)
    assert totals == {
        "configuration_to_membership": 327,
        "membership_to_configuration": 200,
    }
    assert len(events) == 527 and all(
        int(r[flags[0]]) >= int(r[flags[1]]) for r in events
    )
    assert [sum(int(r[f]) for r in events) for f in flags] == [291, 106]
    return {
        "figure_12": strict_intervals,
        "figure_s2": paired,
        "figure_s3_completed_retained_lineage": aggregate,
        "s5_1_additional_contrasts": adjunct,
    }


def replay():
    events = read("r1000_search_replay.csv")
    outcomes = Counter(r["status"] for r in events)
    assert outcomes == {"completed": 5298, "failed": 462}
    assert sum(r["action"] == "Import" for r in events) == 366
    assert sum(r["kept_as_incumbent"] in ("True", "1") for r in events) == 1279
    assert all(
        number(r, "feedback_gain_pp") == 0 for r in events if r["status"] == "failed"
    )
    metrics = {}
    for k, rows in group(events, "dataset", "profile", "seed").items():
        assert len(rows) == 192
        metrics["/".join(k)] = {
            "ridge_mae_pp": mean(
                abs(number(r, "predicted_gain_pp") - number(r, "feedback_gain_pp"))
                for r in rows
            ),
            "zero_mae_pp": mean(abs(number(r, "feedback_gain_pp")) for r in rows),
        }
    return {
        "n": len(events),
        "outcomes": dict(outcomes),
        "retained": 1279,
        "seed_mae": metrics,
    }


def timing():
    pairs = read("time_budget_pairs.csv")
    reports = {
        ("ACDC", "standard"): (0.22, 0.03, 0.42),
        ("ACDC", "tight"): (0.26, 0.15, 0.36),
        ("ControlledFog", "standard"): (0.27, 0.01, 0.52),
        ("ControlledFog", "tight"): (0.37, 0.15, 0.58),
    }
    indexed = group(pairs, "dataset", "budget")
    assert set(indexed) == set(reports)
    trajectories = read("time_budget_trajectories.csv")
    by_trajectory = group(
        read("time_budget_client_stages.csv"), "dataset", "budget", "seed", "method"
    )
    lookup = {}
    for r in trajectories:
        k = key(r, "dataset", "budget", "seed", "method")
        stages = group(by_trajectory[k], "stage")
        assert len(stages) == 4 and all(
            len(clients) == 4 for clients in stages.values()
        )
        elapsed = sum(
            max(number(c, "stage_elapsed_s") for c in clients)
            for clients in stages.values()
        )
        assert isclose(elapsed, number(r, "modeled_adaptation_wall_s"), abs_tol=3e-6)
        lookup[k] = r
    summaries = {}
    for k, rs in indexed.items():
        d = [number(r, "Full_SeenAP_pct") - number(r, "Serial_SeenAP_pct") for r in rs]
        time = [number(r, "Full_time_s") - number(r, "Serial_time_s") for r in rs]
        assert len(rs) == 8
        for r in rs:
            for label in ("Full", "Serial"):
                trajectory = lookup[(*k, r["seed"], label)]
                assert isclose(
                    number(r, label + "_time_s"),
                    number(trajectory, "modeled_adaptation_wall_s"),
                    abs_tol=3e-6,
                )
                assert isclose(
                    number(r, label + "_SeenAP_pct"),
                    number(trajectory, "SeenAP_pct"),
                    abs_tol=3e-6,
                )
        assert all(
            isclose(x, number(r, "AP_diff_pp"), abs_tol=3e-6) for x, r in zip(d, rs)
        )
        summary = paired_interval(d)
        confirm_interval(summary, reports[k])
        time_summary = paired_interval(time)
        if k[1] == "tight":
            confirm_interval(
                time_summary,
                (-0.18, -0.63, 0.28) if k[0] == "ACDC" else (-0.04, -1.75, 1.68),
            )
        summaries["/".join(k)] = {
            "final_seen_ap_difference_pp": summary,
            "modeled_time_difference_s": time_summary,
        }
    return summaries


def bdd():
    rows = read("bdd_revisit_seed_stages.csv")
    client_rows = read("bdd_revisit_client_stages.csv")
    domain_groups = group(
        read("bdd_revisit_client_domains.csv"), "seed", "method", "stage", "client"
    )
    for r in client_rows:
        domains = domain_groups[key(r, "seed", "method", "stage", "client")]
        current = [d for d in domains if d["is_current"] == "1"]
        history = [d for d in domains if d["is_current"] == "0"]
        assert len(current) == 1
        values = {
            "SeenAP_pct": mean(number(d, "T_AP_pct") for d in domains),
            "CurrentAP_pct": number(current[0], "T_AP_pct"),
        }
        if history:
            values.update(
                HistoricalAP_pct=mean(number(d, "T_AP_pct") for d in history),
                Forget_pp=mean(
                    max(0, number(d, "prior_peak_AP_pct") - number(d, "T_AP_pct"))
                    for d in history
                ),
                BWT_pp=mean(
                    number(d, "T_AP_pct") - number(d, "first_AP_pct") for d in history
                ),
            )
        else:
            assert all(r[k] == "" for k in ("HistoricalAP_pct", "Forget_pp", "BWT_pp"))
        for field, value in values.items():
            assert isclose(number(r, field), value, abs_tol=0.00011)
    grouped_clients = group(client_rows, "seed", "method", "stage")
    for r in rows:
        clients = grouped_clients[key(r, "seed", "method", "stage")]
        assert len(clients) == 4
        for k in (
            "SeenAP_pct",
            "CurrentAP_pct",
            "HistoricalAP_pct",
            "Forget_pp",
            "BWT_pp",
        ):
            values = [number(c, k) for c in clients if c[k] != ""]
            if values:
                assert isclose(number(r, k), mean(values), abs_tol=0.00011)
            else:
                assert r[k] == ""
        assert isclose(
            number(r, "MinClientAP_pct"),
            min(number(c, "SeenAP_pct") for c in clients),
            abs_tol=0.00011,
        )
    lookup = {key(r, "seed", "method", "stage"): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    methods = {r["method"] for r in rows}
    assert len(seeds) == 8 and len(methods) == 2 and "Full" in methods
    control = next(iter(methods - {"Full"}))

    def diff(seed, stage, field):
        return number(lookup[seed, "Full", stage], field) - number(
            lookup[seed, control, stage], field
        )

    out = {
        "current_ap_revisit_difference_pp": paired_interval(
            [mean(diff(s, str(t), "CurrentAP_pct") for t in range(5, 9)) for s in seeds]
        ),
        "final_seen_ap_difference_pp": paired_interval(
            [diff(s, "8", "SeenAP_pct") for s in seeds]
        ),
        "final_forget_difference_pp": paired_interval(
            [diff(s, "8", "Forget_pp") for s in seeds]
        ),
    }
    for v, published in zip(
        out.values(),
        [(0.57, 0.40, 0.74), (0.78, 0.62, 0.94), (-0.17, -0.24, -0.11)],
        strict=True,
    ):
        confirm_interval(v, published)
    bdd_endpoints = group(
        [r for r in read("bdd100k_numerical_endpoints.csv") if r["method"] == "CIRA"],
        "profile",
    )
    for profile, printed in (
        ("YOLOv8n", 32.29),
        ("YOLO11n", 33.43),
        ("YOLO26n", 34.26),
    ):
        rounded(
            mean(number(r, "seen_score_rescaled") for r in bdd_endpoints[(profile,)]),
            printed,
        )
    out["client_stage_reductions_verified"] = len(client_rows)
    return out


def reduced_features():
    masks = read("feature_model_configurations.csv")
    contexts = ("empty", "fixed_peer")
    z_labels = ("z0", "z1")
    paired = defaultdict(list)
    flips = Counter()
    for identifiers, branches in group(
        masks, "dataset", "profile", "seed", "client"
    ).items():
        cells = {key(r, "context", "configuration"): r for r in branches}
        assert set(cells) == {(c, z) for c in contexts for z in z_labels}

        def gap(c, field):
            return number(cells[c, "z1"], field) - number(cells[c, "z0"], field)

        g0, g1 = (gap(c, "T_current_AP") for c in contexts)
        flips["V"] += gap(contexts[0], "V_AP") * gap(contexts[1], "V_AP") < 0
        flips["T"] += g0 * g1 < 0

        def selected(c):
            return max(z_labels, key=lambda z: number(cells[c, z], "V_AP"))

        choices = {c: selected(c) for c in contexts}
        effects = []
        for c, other in ((contexts[0], contexts[1]), (contexts[1], contexts[0])):
            effects.append(
                number(cells[c, choices[c]], "T_current_AP")
                - number(cells[c, choices[other]], "T_current_AP")
            )
        paired[identifiers[:3]].append((g1 - g0, mean(effects)))
    assert flips == {"V": 12, "T": 8}
    assert len(paired) == 30 and all(len(x) == 4 for x in paired.values())
    by_condition = defaultdict(list)
    for identifiers, receivers in paired.items():
        by_condition[identifiers[:2]].append(
            (mean(x[0] for x in receivers), mean(x[1] for x in receivers))
        )
    summaries = {}
    for k, scores in sorted(by_condition.items()):
        summaries["/".join(k)] = {
            "interaction_pp": paired_interval([r[0] for r in scores]),
            "context_selection_pp": paired_interval([r[1] for r in scores]),
        }
    expected = {
        "ACDC/YOLOv8n": 0.039,
        "ACDC/YOLO11n": 0.023,
        "ACDC/YOLO26n": -0.195,
        "ControlledFog/YOLOv8n": -0.047,
        "ControlledFog/YOLO11n": 0.008,
        "ControlledFog/YOLO26n": -0.055,
    }
    for k, v in expected.items():
        rounded(summaries[k]["context_selection_pp"][0], v, 3)
    mixing = group(
        read("feature_model_mixing.csv"), "dataset", "profile", "seed", "client"
    )
    assert len(mixing) == 120
    assert all(
        len(x) == 2 and len({r["T_current_AP"] for r in x}) == 1
        for x in mixing.values()
    )
    return {
        "strict_reversals": dict(flips),
        "test_ap_unchanged_pairs": len(mixing),
        "table_s14": summaries,
    }


def class_allocation():
    records = read("class_allocation_pairs.csv")
    assert len(records) == 750
    changes = defaultdict(list)
    for k, values in group(records, "setting", "dataset", "profile", "method").items():
        d = [
            number(r, "seen_score_rescaled") - number(r, "reference_rescaled")
            for r in values
        ]
        assert d == [number(r, "paired_difference") for r in values]
        assert len(d) == 5
        changes[k[0]].append(mean(d))
    for setting, expected in (("a0.1", (-0.82, -0.60)), ("aequal", (0.09, 0.14))):
        rounded(min(changes[setting]), expected[0])
        rounded(max(changes[setting]), expected[1])
    return {
        level: [min(values), max(values)] for level, values in sorted(changes.items())
    }


def additional_controls():
    frozen = {}
    for k, rows in group(
        read("frozen_feature_pairs.csv"), "dataset", "profile", "comparison"
    ).items():
        frozen["/".join(k)] = paired_interval([number(r, "value") for r in rows])
    curves = {}
    for k, rows in group(
        read("budget_curve_pairs.csv"), "dataset", "profile", "method", "axis", "level"
    ).items():
        curves["/".join(k)] = paired_interval([number(r, "paired_delta") for r in rows])
    equal_work = read("matched_work_pairs.csv")
    for r in equal_work:
        assert number(r, "F_fit_work") == number(r, "R_fit_work")
        assert isclose(
            number(r, "R_deltaScore") - number(r, "F_deltaScore"),
            number(r, "R_minus_F"),
            abs_tol=1e-8,
        )
    matched = {}
    for k, rows in group(equal_work, "dataset", "profile", "method").items():
        matched["/".join(k)] = paired_interval([number(r, "R_minus_F") for r in rows])
    assert all(x[1] <= 0 <= x[2] for x in matched.values())
    robust = {}
    records = read("perturbation_records.csv")
    normal = {
        key(r, "dataset", "profile", "method", "seed"): number(r, "score")
        for r in records
        if r["stress"] == "normal"
    }
    for k, rows in group(records, "dataset", "profile", "method", "stress").items():
        deltas = [
            number(r, "score") - normal[key(r, "dataset", "profile", "method", "seed")]
            for r in rows
        ]
        assert all(
            isclose(v, number(r, "own_normal_difference"), abs_tol=1e-8)
            for r, v in zip(rows, deltas)
        )
        robust["/".join(k)] = paired_interval(deltas)
    return {
        "figure13": frozen,
        "figure15_budget": curves,
        "figure15_matched_work": matched,
        "figure16": robust,
    }


def main():
    parser = ArgumentParser(
        description="Recompute paper results from the companion data."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    set_data_root(args.data_root)
    check_manifest()
    report = {
        "detector_protocol_allocations": allocations(),
        "r1000": r1000(),
        "continuation": continuation(),
        "replay": replay(),
        "time_budget": timing(),
        "reduced_feature_model": reduced_features(),
        "bdd_revisits": bdd(),
        "class_allocation": class_allocation(),
        "additional_controls": additional_controls(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
