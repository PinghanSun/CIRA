"""Generate a labeled synthetic planning example from the registered CIRA matrix."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from scipy import stats


from softcas.reporting import macro_endpoints, paired_effects

LABEL = "SIMULATED"
DATASETS = ("ACDC", "ControlledFog")
SEEDS = tuple(range(1001, 1009))
CORE = ("Strict No-cross", "Serial")
EXTENDED = ("Strict No-gain", "DeModCL")
BASE = {"ACDC": 32.4, "ControlledFog": 34.9}
SEED_SHIFTS = {
    "ACDC": [-0.83, 0.35, -0.30, 1.03, 0.54, -0.36, 0.18, -0.61],
    "ControlledFog": [0.42, -0.85, 1.14, -0.29, -0.60, 0.28, 0.73, -0.83],
}
DIFFERENCES = {
    "ACDC": {
        "Strict No-cross": [0.92, 1.43, 0.64, 1.30, 1.62, 0.75, -0.06, 1.39],
        "Serial": [0.34, 0.55, -0.12, 0.74, 0.62, 0.29, 0.12, 0.87],
        "Strict No-gain": [0.21, 0.51, 0.09, 0.56, 0.42, -0.14, 0.15, 0.36],
        "DeModCL": [0.24, 0.69, -0.29, 0.63, 0.71, -0.16, 0.35, 0.40],
    },
    "ControlledFog": {
        "Strict No-cross": [1.12, 0.76, 1.31, 0.55, 1.16, -0.16, 0.68, 1.22],
        "Serial": [0.17, 0.48, 0.63, -0.20, 0.50, 0.37, 0.02, 0.66],
        "Strict No-gain": [0.37, 0.20, 0.58, -0.31, 0.32, 0.19, 0.34, 0.10],
        "DeModCL": [0.52, -0.16, 0.61, 0.17, 0.40, -0.33, 0.09, 0.56],
    },
}
DOMAINS = {
    "ACDC": ("Fog", "Night", "Rain", "Snow"),
    "ControlledFog": ("clear", "fog_005", "fog_010", "fog_020"),
}
DOMAIN_OFFSETS = {"ACDC": [5, -12, 1, 6], "ControlledFog": [8, 3, -1, -10]}
METHODS = {
    ("softcas", "none"): "Full",
    ("softcas", "no_handoff"): "Strict No-cross",
    ("softcas", "no_gain"): "Strict No-gain",
    ("serial", "none"): "Serial",
    ("DeModCL", "none"): "DeModCL",
}


def rng(*key):
    token = "|".join(map(str, ("CIRA_SIM_20261123", *key)))
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
    )


def titled(rows):
    return [dict(data_status=LABEL, **row) for row in rows]


def simulate_run(recipe):
    ds, seed = recipe["dataset"], recipe["seed"]
    method = METHODS[recipe["method"], recipe["ablation"]]
    si = SEEDS.index(seed)
    group = "核心" if method in ("Full", *CORE) else "扩展"
    run = f"SIM-{ds}-{seed}-{recipe['method']}-{recipe['ablation']}"
    context = dict(sim_id=run, group=group, dataset=ds, seed=seed, method=method)
    shared = rng(ds, seed)
    local = rng(ds, seed, method)
    target_mean = BASE[ds] + SEED_SHIFTS[ds][si]
    if method != "Full":
        target_mean -= DIFFERENCES[ds][method][si]

    # Shared domain shifts and Latin-square domain exposure for the paired methods.
    order = shared.permutation(4)
    client_offsets = shared.normal(0, 0.75, (4, 1))
    domain_noise = shared.normal(0, 0.42, (4, 4))
    epoch_nominal = shared.integers(78, 96, (4, 4))
    proposal_nominal = shared.integers(8, 14, (4, 4))
    package_mib = shared.uniform(1.45, 2.8, (4, 4))
    audit_draw = shared.random((4, 4))
    thresholds = {
        "Full": 0.83,
        "Strict No-cross": 0.77,
        "Serial": 0.78,
        "Strict No-gain": 0.78,
        "DeModCL": 0.76,
    }
    releases = audit_draw < thresholds[method]
    finals = (
        np.array(DOMAIN_OFFSETS[ds])[None, :]
        + client_offsets
        + domain_noise
        + local.normal(0, 0.22, (4, 4))
    )
    orders = [np.roll(order, c) for c in range(4)]
    for c in range(4):
        finals[c, orders[c][-1]] += 0.35 if releases[c, -1] else -1.25
    finals -= finals.mean()
    finals += target_mean

    raw, resources = [], []
    for c in range(4):
        declines = local.normal(0.80 if method == "Full" else 0.89, 0.46, (4, 4))
        declines[~releases[c, :]] = 0
        for s in range(4):
            ceiling = recipe["stage_epochs"]
            attempt_limit = recipe["attempts"]
            used = int(
                np.clip(
                    epoch_nominal[c, s]
                    + local.integers(-5, 6)
                    - 2 * (method == "Serial"),
                    1,
                    ceiling,
                )
            )
            attempts = int(
                np.clip(
                    proposal_nominal[c, s]
                    + local.integers(-1, 2)
                    + 1 * (method == "Full"),
                    1,
                    attempt_limit,
                )
            )
            failures = int(local.binomial(attempts, 0.075))
            evaluations = 1 + attempts - int(local.integers(0, min(3, attempts) + 1))
            bytes_used = int(
                package_mib[c, s]
                * 1048576
                * local.uniform(0.91, 1.09)
                * (0.94 if method == "Full" else 1)
            )
            if "Strict No-cross" == method:
                directions = [0, 0]
            elif method in ("Full", "Strict No-gain"):
                directions = list(
                    map(
                        int,
                        local.multinomial(attempts - failures, (0.12, 0.075, 0.805))[
                            :2
                        ],
                    )
                )
            else:
                directions = [None, None]
            retained = [
                int(local.binomial(n, 0.56)) if n is not None else None
                for n in directions
            ]
            deployed = [
                int(local.binomial(n, 0.48))
                if releases[c, s] and n is not None
                else (0 if n is not None else None)
                for n in retained
            ]
            if deployed[0] is not None:
                deployed = [
                    min(deployed[0], deployed[1] + 1),
                    min(deployed[1], deployed[0] + 1),
                ]
            resources.append(
                context
                | dict(
                    client=c + 1,
                    stage=s + 1,
                    F_used_epochs=used,
                    F_limit_epochs=ceiling,
                    attempts=attempts,
                    attempt_limit=attempt_limit,
                    failed_attempts=failures,
                    E_selection=evaluations,
                    E_limit=attempt_limit + 1,
                    V_images_per_eval=12 if ds == "ACDC" else 77,
                    search_payload_bytes=bytes_used,
                    released=int(releases[c, s]),
                    cfg_to_mem=directions[0],
                    mem_to_cfg=directions[1],
                    retained_cfg_to_mem=retained[0],
                    retained_mem_to_cfg=retained[1],
                    deployed_cfg_to_mem=deployed[0],
                    deployed_mem_to_cfg=deployed[1],
                )
            )
            for domain in orders[c][: s + 1]:
                points = round(
                    float(finals[c, domain] + declines[s + 1 :, domain].sum()), 4
                )
                raw.append(
                    dict(
                        seed=seed,
                        client=c + 1,
                        stage=s + 1,
                        deployed=f"{run}-C{c + 1}-S{s + 1}",
                        current_domain=DOMAINS[ds][orders[c][s]],
                        domain=DOMAINS[ds][domain],
                        score=points / 100,
                    )
                )
    stages = macro_endpoints(raw, expected_clients=range(1, 5))
    assert len(stages) == 4 and abs(stages[-1]["seen"] * 100 - target_mean) < 0.0001
    endpoint = {
        field: round(stages[-1][nl] * 100, 4)
        for field, nl in [
            ("SeenAP_pct", "seen"),
            ("CurrentAP_pct", "current"),
            ("MinClientAP_pct", "minimum_client_seen"),
            ("Forget_pp", "forget"),
            ("BWT_pp", "bwt"),
        ]
    }
    totals = {
        k: sum(r[k] for r in resources)
        for k in (
            "F_used_epochs",
            "attempts",
            "failed_attempts",
            "E_selection",
            "search_payload_bytes",
            "released",
        )
    }
    totals["V_image_evaluations"] = sum(
        r["E_selection"] * r["V_images_per_eval"] for r in resources
    )
    totals["release_pct"] = 100 * totals["released"] / len(resources)
    for column in (
        "cfg_to_mem",
        "mem_to_cfg",
        "deployed_cfg_to_mem",
        "deployed_mem_to_cfg",
    ):
        totals[column] = (
            None if resources[0][column] is None else sum(r[column] for r in resources)
        )
    domain_data = [
        context
        | dict(
            client=r["client"],
            stage=r["stage"],
            current_domain=r["current_domain"],
            test_domain=r["domain"],
            AP50_95_pct=round(r["score"] * 100, 4),
        )
        for r in raw
    ]
    return context | endpoint | totals, domain_data, resources


def interval(values):
    n = len(values)
    m = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return m, sd, m - margin, m + margin


def summarize(runs):
    indexed = {(r["dataset"], r["method"], r["seed"]): r for r in runs}
    comparisons, per_seed = [], []
    field_pairs = [
        ("SeenAP_pct", "SeenAP_diff_pp"),
        ("CurrentAP_pct", "CurrentAP_diff_pp"),
        ("Forget_pp", "Forget_diff_pp"),
        ("BWT_pp", "BWT_diff_pp"),
        ("V_image_evaluations", "V_images_diff"),
        ("search_payload_bytes", "payload_diff_bytes"),
        ("release_pct", "release_diff_pp"),
    ]
    for method in (*CORE, *EXTENDED):
        a = {
            (ds, s): indexed[ds, "Full", s]["SeenAP_pct"]
            for ds in DATASETS
            for s in SEEDS
        }
        b = {
            (ds, s): indexed[ds, method, s]["SeenAP_pct"]
            for ds in DATASETS
            for s in SEEDS
        }
        effects = paired_effects(a, b, DATASETS, seeds=SEEDS)
        for effect in effects:
            ds = effect["cell"]
            local = []
            for seed in SEEDS:
                left, right = (indexed[ds, m, seed] for m in ("Full", method))
                pair = dict(
                    dataset=ds,
                    comparator=method,
                    seed=seed,
                    Full_SeenAP_pct=left["SeenAP_pct"],
                    comparator_SeenAP_pct=right["SeenAP_pct"],
                )
                pair.update(
                    {
                        dest: round(left[src] - right[src], 6)
                        for src, dest in field_pairs
                    }
                )
                local.append(pair)
                per_seed.append(pair)
            lo, hi = effect["ci95_unadjusted"]
            base_v = np.mean(
                [indexed[ds, method, s]["V_image_evaluations"] for s in SEEDS]
            )
            base_b = np.mean(
                [indexed[ds, method, s]["search_payload_bytes"] for s in SEEDS]
            )
            comparisons.append(
                dict(
                    dataset=ds,
                    comparison=f"Full − {method}",
                    n=effect["n_pair"],
                    mean_diff_pp=round(effect["mean"], 4),
                    sd_diff_pp=round(effect["sample_sd"], 4),
                    CI95_lo_pp=round(lo, 4),
                    CI95_hi_pp=round(hi, 4),
                    positive_seeds=sum(r["SeenAP_diff_pp"] > 0 for r in local),
                    p_holm=round(effect["p_holm"], 6) if method in CORE else None,
                    Forget_diff_pp=round(
                        float(np.mean([r["Forget_diff_pp"] for r in local])), 4
                    ),
                    BWT_diff_pp=round(
                        float(np.mean([r["BWT_diff_pp"] for r in local])), 4
                    ),
                    V_work_change_pct=round(
                        float(np.mean([r["V_images_diff"] for r in local]))
                        / base_v
                        * 100,
                        2,
                    ),
                    payload_change_pct=round(
                        float(np.mean([r["payload_diff_bytes"] for r in local]))
                        / base_b
                        * 100,
                        2,
                    ),
                )
            )
    return comparisons, per_seed


def mechanism():
    cells, paired, summary = [], [], []
    v_target = {
        "ACDC": [0.58, 0.44, -0.22, 0.67, 0.38, 0.16, -0.23, 0.41],
        "ControlledFog": [0.41, -0.18, 0.52, 0.24, -0.13, 0.44, -0.12, 0.36],
    }
    t_target = {
        "ACDC": [0.54, 0.45, -0.30, 0.47, 0.27, -0.16, 0.09, 0.39],
        "ControlledFog": [0.46, -0.27, 0.39, 0.27, 0.11, 0.34, -0.20, -0.13],
    }
    for ds in DATASETS:
        ds_pairs = []
        for i, seed in enumerate(SEEDS):
            group = dict(
                dataset=ds,
                seed=seed,
                snapshot_id=f"SIM-E3-{ds}-{seed}",
                client=1 + i % 4,
                stage=2 + i % 3,
                fit_epochs=8,
            )
            g = rng("E3", ds, seed)
            v_base = float(g.normal(BASE[ds] - 1.4, 0.57))
            t_base = float(g.normal(BASE[ds] - 1.9, 0.58))
            scores = {}
            for peer, mass, v_gap, t_gap in [
                (
                    "空组合",
                    0.0,
                    -float(g.uniform(0.25, 0.65)),
                    -float(g.uniform(0.07, 0.51)),
                ),
                ("固定非空", 0.35, v_target[ds][i], t_target[ds][i]),
            ]:
                offset = 0.33 if mass else 0
                scores[peer] = {}
                for name, sign in [("A 端口1", -1), ("B 端口4", 1)]:
                    v = round(v_base + offset + sign * v_gap / 2, 4)
                    t = round(t_base + offset + sign * t_gap / 2, 4)
                    cells.append(
                        group
                        | dict(
                            peer_condition=peer,
                            peer_mass=mass,
                            config=name,
                            V_AP_pct=v,
                            T_AP_pct=t,
                        )
                    )
                    scores[peer][name] = (v, t)
            empty, filled = scores["空组合"], scores["固定非空"]
            a, b = (
                max(empty, key=lambda k: empty[k][0]),
                max(filled, key=lambda k: filled[k][0]),
            )
            carry, selected = filled[a][1], filled[b][1]
            oracle = max(value[1] for value in filled.values())
            row = group | dict(
                V_empty_choice=a,
                V_nonempty_choice=b,
                V_rank_reversal=int(a != b),
                T_rank_reversal=int(
                    max(empty, key=lambda k: empty[k][1])
                    != max(filled, key=lambda k: filled[k][1])
                ),
                T_carry_pct=carry,
                T_selected_pct=selected,
                T_oracle_pct=oracle,
                heldout_reselect_pp=round(selected - carry, 4),
                remaining_oracle_pp=round(oracle - selected, 4),
            )
            paired.append(row)
            ds_pairs.append(row)
        avg, sd, lo, hi = interval([r["heldout_reselect_pp"] for r in ds_pairs])
        summary.append(
            dict(
                dataset=ds,
                n=len(ds_pairs),
                V_reversals=sum(r["V_rank_reversal"] for r in ds_pairs),
                T_reversals=sum(r["T_rank_reversal"] for r in ds_pairs),
                heldout_mean_pp=round(avg, 4),
                CI95_lo_pp=round(lo, 4),
                CI95_hi_pp=round(hi, 4),
                remaining_oracle_mean_pp=round(
                    float(np.mean([r["remaining_oracle_pp"] for r in ds_pairs])), 4
                ),
            )
        )
    return cells, paired, summary


def run(output, core_matrix, extended_matrix):
    source = extended_matrix
    jobs = json.loads(source.read_text(encoding="utf-8"))["jobs"]
    core = json.loads(core_matrix.read_text(encoding="utf-8"))["jobs"]
    assert len(jobs) == 80 and len(core) == 48
    assert {j["id"] for j in core} <= {j["id"] for j in jobs}
    runs, domains, resource = [], [], []
    for job in jobs:
        target, raw, cost = simulate_run(job["recipe"])
        runs.append(target)
        domains.extend(raw)
        resource.extend(cost)
    comparisons, pairs = summarize(runs)
    cells, diagnostics, diag_summary = mechanism()
    assert (
        len(runs) == 80
        and len(domains) == 3200
        and len(resource) == 1280
        and len(cells) == 64
    )
    assert len({r["sim_id"] for r in runs if r["group"] == "核心"}) == 48
    for row in resource:
        assert (
            row["F_used_epochs"] <= row["F_limit_epochs"]
            and row["E_selection"] <= row["attempts"] + 1 <= row["E_limit"]
        )
        if row["cfg_to_mem"] is not None:
            assert (
                row["cfg_to_mem"] + row["mem_to_cfg"]
                <= row["attempts"] - row["failed_attempts"]
            )
            for kind in ("cfg_to_mem", "mem_to_cfg"):
                assert row[f"deployed_{kind}"] <= row[f"retained_{kind}"] <= row[kind]
    tables = {
        "说明": [
            {
                "项目": "用途",
                "说明": "偏强且增益克制的合成情景。用于规划、复算和结论演练。",
            },
            {
                "项目": "结果身份",
                "说明": "全表 SIMULATED。已登记的原生实验状态保留为未运行。",
            },
            {
                "项目": "矩阵",
                "说明": "ACDC、ControlledFog，YOLOv8n/C2，8 个配对种子。核心48条，扩展后共80条。",
            },
            {
                "项目": "效应设定",
                "说明": "模拟输入事先指定小幅配对差，包含个别负差。汇总区间只描述这个情景。",
            },
            {
                "项目": "统计口径",
                "说明": "四客户端等权。95%为边际配对t区间。每项核心比较跨两数据集做Holm调整。",
            },
            {
                "项目": "指标单位",
                "说明": "SeenAP、CurrentAP及最弱客户端AP为AP50:95百分数。差值、Forget和BWT为百分点。",
            },
            {
                "项目": "资源口径",
                "说明": "资源为示例计数，使用配方中100逻辑epoch和16次尝试。V图像评价量为模拟值。",
            },
            {
                "项目": "试跑和设备",
                "说明": "两条开发试跑的耗时、设备W额度及Jetson计时/功耗保持待实测。",
            },
            {
                "项目": "扩展对照",
                "说明": "DeModCL为模拟占位，其原生检测器资格及资产以实际验证为准。",
            },
            {
                "项目": "四格",
                "说明": "空组合与固定0.35权重非空组合，同等8 epochs，在V上选配置，在相同非空组合的T上比较。",
            },
            {
                "项目": "原始矩阵SHA256",
                "说明": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        ],
        "比较汇总": titled(comparisons),
        "逐种子配对": titled(pairs),
        "模拟轨迹": titled(runs),
        "阶段历史域": titled(domains),
        "阶段资源与接续": titled(resource),
        "四格汇总": titled(diag_summary),
        "四格逐seed": titled(diagnostics),
        "四格原始值": titled(cells),
    }
    from supplement_sim import extend

    extend(tables, runs, resource, rng)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "data"
    target.mkdir(exist_ok=True)
    for name, rows in tables.items():
        with (target / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (output / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "counts": {name: len(value) for name, value in tables.items()},
                "effects": comparisons[:4],
                "mechanism": diag_summary,
            },
            ensure_ascii=True,
        )
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--core-matrix", required=True, type=Path)
    parser.add_argument("--extended-matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    run(args.output, args.core_matrix, args.extended_matrix)


if __name__ == "__main__":
    main()
