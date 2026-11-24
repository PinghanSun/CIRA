"""Extend the planning simulation with ordered feedback and deployment accounting."""

from collections import defaultdict
import math
from statistics import mean, stdev

import numpy as np
from scipy.stats import t


FAMILIES = ("configuration", "membership")
DIRECTIONS = ("cfg_to_mem", "mem_to_cfg")
METHODS = ("Full", "Strict No-cross", "Serial")
STATUS = "SIMULATED"


def stage_action_order(stage):
    crosses = [stage[k] or 0 for k in DIRECTIONS]
    retained = [stage["retained_" + k] or 0 for k in DIRECTIONS]
    deployed = [stage["deployed_" + k] or 0 for k in DIRECTIONS]
    # Seed both required families before attaching any cross-family child.
    plan = [
        (family, "root", None, False, True) for family, n in enumerate(crosses) if n
    ]
    left = deployed.copy()
    family = 0 if left[0] >= left[1] else 1
    while any(left):
        assert left[family] > 0
        plan.append((1 - family, "path", family, True, True))
        left[family] -= 1
        family = 1 - family
    for family, n in enumerate(crosses):
        for i in range(n - deployed[family]):
            plan.append(
                (
                    1 - family,
                    "branch",
                    family,
                    False,
                    i < retained[family] - deployed[family],
                )
            )
    return plan


def simulate_attempts(stage, gen):
    context = {
        k: stage[k] for k in ("sim_id", "dataset", "method", "seed", "client", "stage")
    }
    r = gen("actions", *context.values())
    prefix = f"{stage['sim_id']}-C{stage['client']}-S{stage['stage']}"
    entry = prefix + "-entry"
    available = [
        [(prefix + "-import-cfg", "imported_checkpoint")],
        [(prefix + "-import-mem", "imported_checkpoint")],
    ]
    roots = {}
    path_end = None
    success_plan = iter(stage_action_order(stage))
    failed = set(
        r.choice(stage["attempts"], stage["failed_attempts"], replace=False).tolist()
    )
    xtx = 0.55 * np.eye(4)
    xty = np.zeros(4)
    history = []
    rows = []
    for i in range(stage["attempts"]):
        success = i not in failed
        spec = next(success_plan, None) if success else None
        if spec is not None:
            child, link, source, on_path, retained = spec
        else:
            child, link, source, on_path, retained = (
                int(r.integers(0, 2)),
                "ordinary",
                None,
                False,
                None,
            )
        state_id = f"{prefix}-A{i + 1}"
        parent, parent_type, parent_family = entry, "stage_entry", None
        if link in ("path", "branch"):
            parent = (path_end or roots[source]) if link == "path" else roots[source]
            parent_type, parent_family = "stage_revision", source
        elif link == "ordinary" and r.random() < 0.68:
            eligible = available[child]
            parent, parent_type = eligible[int(r.integers(len(eligible)))]
            parent_family = child

        feature = np.array(
            [
                1,
                float(r.normal(0.25, 0.8)),
                float(r.uniform(-1, 1)),
                float(source is not None),
            ]
        )
        pred = round(float(feature @ np.linalg.solve(xtx, xty)), 6)
        noise = 0.17 if parent_type == "stage_entry" else 0.24
        gain = float(
            np.clip(
                feature @ np.array([0.10, 0.54, 0.24, 0.14]) + r.normal(0, noise),
                -1.9,
                2.0,
            )
        )
        observed = round(gain, 6) if success else None
        feedback = observed if success else 0.0
        previous = round(mean(history), 6) if history else 0.0
        if success:
            if retained is None:
                retained = bool(gain >= 0 or r.random() < 0.18)
            if retained:
                available[child].append((state_id, "stage_revision"))
            if link == "root":
                roots[child] = state_id
            if on_path:
                if path_end is not None:
                    assert parent == path_end
                path_end = state_id
        rows.append(
            context
            | dict(
                attempt=i + 1,
                proposal_id=state_id,
                parent_id=parent,
                parent_origin=parent_type,
                parent_family=FAMILIES[parent_family]
                if parent_family is not None
                else "entry",
                child_family=FAMILIES[child],
                direction=DIRECTIONS[source] if source is not None else None,
                child_retained=int(retained) if retained is not None else None,
                deployed_cross_path=int(on_path)
                if stage["cfg_to_mem"] is not None
                else None,
                attempt_status="complete" if success else "failed",
                pre_feature_compat=round(float(feature[1]), 6),
                pre_feature_fit=round(float(feature[2]), 6),
                ridge_pred_pp=pred,
                zero_pred_pp=0.0,
                history_pred_pp=previous,
                observed_gain_pp=observed,
                feedback_gain_pp=feedback,
                ridge_abs_pp=round(abs(pred - feedback), 6),
                zero_abs_pp=abs(feedback),
                history_abs_pp=round(abs(previous - feedback), 6),
            )
        )
        history.append(feedback)
        xtx += np.outer(feature, feature)
        xty += feature * feedback
    assert next(success_plan, None) is None
    assert (
        sum(x["attempt_status"] == "failed" for x in rows) == stage["failed_attempts"]
    )
    for k in DIRECTIONS:
        selected = [x for x in rows if x["direction"] == k]
        assert len(selected) == (stage[k] or 0)
        assert sum(x["child_retained"] for x in selected) == (
            stage["retained_" + k] or 0
        )
        assert sum(x["deployed_cross_path"] for x in selected) == (
            stage["deployed_" + k] or 0
        )
    return rows


def account_fitting(attempts, stages, runs, gen):
    """Roll up charged and completed optimizer steps within each simulated stage."""
    grouped = defaultdict(list)
    for row in attempts:
        grouped[row["sim_id"], row["client"], row["stage"]].append(row)
    run_totals = defaultdict(list)
    for stage in stages:
        key = stage["sim_id"], stage["client"], stage["stage"]
        domain_size = 76 if stage["dataset"] == "ACDC" else 439 + (stage["stage"] != 1)
        batches = math.ceil(domain_size / 8)
        stage["scenario_D_images"] = domain_size
        stage["steps_per_logical_epoch"] = batches
        ledger = grouped.get(key)
        if ledger is None:
            stage.update(
                fit_attempted_steps=None,
                fit_completed_steps=None,
                fit_failed_charged_steps=None,
                F_accounting="external_scenario_total",
            )
        else:
            noise = gen("fit_step_accounting", *key)
            remaining = stage["F_limit_epochs"] * batches
            charged_total = completed_total = failed_charge = 0
            for action in ledger:
                configuration = action["child_family"] == "configuration"
                epochs_cap = 8 if configuration else 4
                cap = epochs_cap * batches
                goal = batches * int(
                    noise.integers(4 if configuration else 1, epochs_cap + 1)
                )
                admitted = min(goal, remaining)
                complete = action["attempt_status"] == "complete"
                charged = (
                    admitted
                    if complete or not admitted
                    else int(noise.integers(0, admitted))
                )
                completed = max(0, charged - int(not complete and charged > 0))
                remaining -= charged
                charged_total += charged
                completed_total += completed
                failed_charge += charged if not complete else 0
                action.update(
                    fit_mode="local" if configuration else "mixing_only",
                    fit_epoch_cap=epochs_cap,
                    steps_per_logical_epoch=batches,
                    fit_step_cap=cap,
                    fit_steps_admitted=admitted,
                    fit_steps_attempted=charged,
                    fit_steps_completed=completed,
                    fit_epochs_charged=round(charged / batches, 9),
                )
            assert charged_total <= len(ledger) * 8 * batches
            stage.update(
                F_used_epochs=round(charged_total / batches, 9),
                fit_attempted_steps=charged_total,
                fit_completed_steps=completed_total,
                fit_failed_charged_steps=failed_charge,
                F_accounting="attemptwise_charged_steps",
            )
        run_totals[stage["sim_id"]].append(stage)
    for run in runs:
        group = run_totals[run["sim_id"]]
        run["F_used_epochs"] = round(sum(s["F_used_epochs"] for s in group), 9)
        for col in (
            "fit_attempted_steps",
            "fit_completed_steps",
            "fit_failed_charged_steps",
        ):
            run[col] = sum(s[col] for s in group) if group[0][col] is not None else None
        run["F_accounting"] = group[0]["F_accounting"]


def prediction_tables(attempts):
    keys = ("ridge", "zero", "history")
    seeds = defaultdict(list)
    for r in attempts:
        if r["method"] != "Full":
            continue
        for population in (
            ("all_executed", "completed_only")
            if r["attempt_status"] == "complete"
            else ("all_executed",)
        ):
            for origin in ("all", r["parent_origin"]):
                seeds[r["dataset"], r["seed"], population, origin].append(r)
    per_seed, summaries = [], []
    groups = defaultdict(list)
    for (d, seed, p, o), rows in sorted(seeds.items()):
        row = dict(
            dataset=d,
            seed=seed,
            population=p,
            parent_origin=o,
            attempts=len(rows),
            failed=sum(x["attempt_status"] == "failed" for x in rows),
        )
        row.update(
            {
                k + "_MAE_pp": round(mean(x[k + "_abs_pp"] for x in rows), 6)
                for k in keys
            }
        )
        per_seed.append(row)
        groups[d, p, o].append(row)
    for (d, p, o), rows in groups.items():
        gap = [x["history_MAE_pp"] - x["ridge_MAE_pp"] for x in rows]
        m = mean(gap)
        half = (
            t.ppf(0.975, len(gap) - 1) * stdev(gap) / math.sqrt(len(gap))
            if len(gap) > 1
            else None
        )
        summary = dict(
            dataset=d,
            population=p,
            parent_origin=o,
            n_seeds=len(rows),
            n_attempts=sum(x["attempts"] for x in rows),
            n_failed=sum(x["failed"] for x in rows),
        )
        summary.update(
            {
                k + "_MAE_pp": round(mean(x[k + "_MAE_pp"] for x in rows), 6)
                for k in keys
            }
        )
        summaries.append(
            summary
            | dict(
                history_minus_ridge_pp=round(m, 6),
                CI95_lo_pp=round(m - half, 6) if half is not None else None,
                CI95_hi_pp=round(m + half, 6) if half is not None else None,
            )
        )
    return per_seed, summaries


def resource_model(runs, stages, gen):
    costs = defaultdict(dict)
    stage_rows = []
    for r in stages:
        g = gen("cycle_time", r["dataset"], r["seed"], r["client"], r["stage"])
        epoch_speed = float(g.uniform(0.82, 1.12)) * (
            1.35 if r["dataset"] == "ACDC" else 4.4
        )
        fit = r["F_used_epochs"] * epoch_speed
        valid = r["E_selection"] * r["V_images_per_eval"] * 0.020
        transfer = r["search_payload_bytes"] / 1048576 / float(g.uniform(3.8, 5.5))
        audit_close = 1.9 + r["attempts"] * 0.26
        modeled_time = fit + valid + transfer + audit_close
        watts = float(g.uniform(16, 21))
        row = {
            k: r[k] for k in ("sim_id", "dataset", "method", "seed", "client", "stage")
        }
        row |= dict(
            F_used_epochs=r["F_used_epochs"],
            F_accounting=r["F_accounting"],
            model_fit_s=round(fit, 5),
            model_validation_s=round(valid, 5),
            model_transfer_s=round(transfer, 5),
            model_audit_close_s=round(audit_close, 5),
            model_adaptation_s=round(modeled_time, 5),
            scenario_mean_watts=round(watts, 5),
            model_active_energy_J=round(modeled_time * watts, 5),
            measured_start_time=None,
            measured_end_time=None,
            measured_adaptation_s=None,
            asset_sha256=None,
        )
        stage_rows.append(row)
        costs[r["sim_id"]][r["client"], r["stage"]] = row
    totals = []
    for r in runs:
        key = r["sim_id"]
        by_stage = costs[key]
        simulated_wall = sum(
            max(by_stage[c, s]["model_adaptation_s"] for c in range(1, 5))
            for s in range(1, 5)
        )
        device = gen("device_example", r["dataset"], r["seed"])
        method = {
            "Full": 1.0,
            "Strict No-cross": 0.97,
            "Serial": 0.99,
            "Strict No-gain": 1.0,
            "DeModCL": 1.03,
        }[r["method"]]
        repeats = device.lognormal(mean=np.log(12.4 * method), sigma=0.105, size=100)
        peak = float(device.normal(2020, 58)) * method
        energy = sum(row["model_active_energy_J"] for row in by_stage.values())
        context = {k: r[k] for k in ("sim_id", "dataset", "method", "seed")}
        totals.append(
            context
            | dict(
                model_adaptation_wall_s=round(simulated_wall, 4),
                model_device_seconds=round(
                    sum(x["model_adaptation_s"] for x in by_stage.values()), 4
                ),
                scenario_p95_ms=round(
                    float(np.quantile(repeats, 0.95, method="higher")), 4
                ),
                scenario_peak_process_MiB=round(peak, 2),
                model_active_energy_kJ=round(energy / 1000, 4),
                measured_adaptation_wall_s=None,
                measured_p95_ms=None,
                measured_peak_process_MiB=None,
                measured_energy_kJ=None,
                hardware_profile=None,
                checkpoint_sha256=None,
                measurement_status="待实测",
            )
        )
    return stage_rows, totals


def bdd_revisit(gen):
    stages, detail, summaries = [], [], []
    seed_list = range(1001, 1009)
    domain_order = ["clear", "overcast", "rainy", "foggy"]
    revisits = defaultdict(list)
    final = {}
    for seed in seed_list:
        g = gen("BDD_video_cycles", seed)
        offsets = g.normal(0, 0.38, 4).tolist()
        old_gain = g.normal(0.62, 0.29, 4).tolist()
        repeated_gain = g.normal(0.64, 0.39, 4).tolist()
        full_after = g.normal(0.60, 0.36, 4).tolist()
        order = np.roll(domain_order, seed % 4).tolist()
        bases = {"clear": 38, "overcast": 35, "rainy": 31, "foggy": 26}
        for method in ("Full", "Strict No-cross"):
            previous, first, peak, first_stage = {}, {}, {}, {}
            for step in range(8):
                local = step % 4
                domain = order[local]
                cycle = 1 + step // 4
                current = bases[domain] + offsets[local]
                if cycle == 2:
                    current += repeated_gain[local]
                if method == "Full":
                    current += old_gain[local] if cycle == 1 else full_after[local]
                once_seen = list(previous)
                anchor = gen("BDD_history", seed, step)
                policy = gen("BDD_history_policy", seed, step, method)
                scores = {}
                for known in once_seen:
                    if known == domain:
                        continue
                    # The deployed stage checkpoint changes scores on domains not trained in this stage.
                    stochastic = float(anchor.normal(0, 0.14)) + float(
                        policy.normal(0, 0.07)
                    )
                    retention = -0.15 if method == "Full" else -0.29
                    known_revisit_offset = (
                        float(anchor.normal(0.08, 0.07)) if cycle == 2 else 0.0
                    )
                    scores[known] = round(
                        previous[known] + retention + stochastic + known_revisit_offset,
                        4,
                    )
                scores[domain] = round(current, 4)
                checkpoint = f"SIM-BDD-{seed}-{method}-S{step + 1}"
                pool = once_seen + ([domain] if domain not in previous else [])
                row = dict(
                    dataset="BDD100K",
                    detector="YOLOv8n",
                    seed=seed,
                    method=method,
                    stage=step + 1,
                    cycle=cycle,
                    current_domain=domain,
                    video_group=f"SIM-cycle{cycle}-{seed}-{domain}",
                    is_revisit=int(cycle == 2),
                    checkpoint_id=checkpoint,
                    current_AP_pct=scores[domain],
                    seen_AP_pct=round(mean(scores.values()), 4),
                )
                hist = [k for k in once_seen if k != domain]
                row.update(
                    historical_domain_count=len(hist),
                    historical_AP_pct=round(mean(scores[k] for k in hist), 4)
                    if hist
                    else None,
                    Forget_pp=round(mean(max(0, peak[k] - scores[k]) for k in hist), 4)
                    if hist
                    else None,
                    BWT_pp=round(mean(scores[k] - first[k] for k in hist), 4)
                    if hist
                    else None,
                )
                for name in pool:
                    active = name == domain
                    historical = name in hist
                    detail.append(
                        dict(
                            dataset="BDD100K",
                            detector="YOLOv8n",
                            seed=seed,
                            method=method,
                            stage=step + 1,
                            checkpoint_id=checkpoint,
                            current_domain=domain,
                            test_domain=name,
                            is_current=int(active),
                            is_historical=int(historical),
                            first_acquisition_stage=first_stage.get(name, step + 1),
                            fixed_test_pool_id=f"SIM-T-{seed}-{name}",
                            AP50_95_pct=scores[name],
                            prior_peak_AP_pct=peak.get(name),
                            first_acquisition_AP_pct=first.get(name, scores[name]),
                        )
                    )
                    first.setdefault(name, scores[name])
                    first_stage.setdefault(name, step + 1)
                    peak[name] = max(peak.get(name, -math.inf), scores[name])
                previous = scores
                stages.append(row)
                if cycle == 2:
                    revisits[seed, method].append(scores[domain])
                if step == 7:
                    final[seed, method] = row
    contrasts = [("revisit CurrentAP", lambda s, m: mean(revisits[s, m]), True)]
    contrasts.extend(
        (label, lambda s, m, k=field: final[s, m][k], field != "Forget_pp")
        for label, field in [
            ("stage 8 SeenAP", "seen_AP_pct"),
            ("stage 8 historical AP", "historical_AP_pct"),
            ("stage 8 Forget", "Forget_pp"),
            ("stage 8 BWT", "BWT_pp"),
        ]
    )
    for label, read, higher in contrasts:
        full = [read(s, "Full") for s in seed_list]
        strict = [read(s, "Strict No-cross") for s in seed_list]
        diffs = [a - b for a, b in zip(full, strict)]
        m = mean(diffs)
        hw = t.ppf(0.975, 7) * stdev(diffs) / math.sqrt(8)
        summaries.append(
            dict(
                dataset="BDD100K",
                detector="YOLOv8n",
                metric=label,
                contrast="Full − Strict No-cross",
                n_pairs=8,
                Full_mean=round(mean(full), 4),
                Strict_mean=round(mean(strict), 4),
                mean_diff_pp=round(m, 4),
                CI95_lo_pp=round(m - hw, 4),
                CI95_hi_pp=round(m + hw, 4),
                favorable_seeds=sum((x if higher else -x) > 0 for x in diffs),
                protocol_status="八阶段模拟，正式矩阵待冻结",
            )
        )
    assert (
        len(stages) == 128
        and len(detail) == 416
        and all(n == 4 for n in map(len, revisits.values()))
    )
    return stages, detail, summaries


def extend(tables, runs, resources, gen):
    attempts = [
        a
        for stage in resources
        if stage["method"] != "DeModCL"
        for a in simulate_attempts(stage, gen)
    ]
    account_fitting(attempts, resources, runs, gen)
    seed_prediction, summary = prediction_tables(attempts)
    phase_cost, device = resource_model(runs, resources, gen)
    bdd, bdd_domains, bdd_summary = bdd_revisit(gen)
    tables["模拟轨迹"] = [dict(data_status=STATUS, **row) for row in runs]
    tables["阶段资源与接续"] = [dict(data_status=STATUS, **row) for row in resources]
    core_summary = []
    for ds in ("ACDC", "ControlledFog"):
        for method in METHODS:
            selected = [r for r in runs if r["dataset"] == ds and r["method"] == method]
            core_summary.append(
                dict(
                    dataset=ds,
                    method=method,
                    n=len(selected),
                    mean_SeenAP_pct=round(mean(r["SeenAP_pct"] for r in selected), 4),
                    sd_SeenAP_pp=round(stdev(r["SeenAP_pct"] for r in selected), 4),
                    CurrentAP_pct=round(mean(r["CurrentAP_pct"] for r in selected), 4),
                    Forget_pp=round(mean(r["Forget_pp"] for r in selected), 4),
                    BWT_pp=round(mean(r["BWT_pp"] for r in selected), 4),
                )
            )
    tables["说明"].extend(
        [
            {
                "项目": "预测误差",
                "说明": "模拟的4维在线岭回归。仅使用尝试前的特征与早先反馈。预测汇总限Full，失败按协议零反馈并另列成功子集。",
            },
            {
                "项目": "真实预测待补",
                "说明": "正式图14应读取实际原生25维Ridge和同次反馈，History仅使用同轨迹、客户端和阶段中的早先反馈。",
            },
            {
                "项目": "时间与设备",
                "说明": "模型时间=拟合+验证+传输+审核收尾。四节点同阶段取最大后按阶段求和。能耗估计只含设备活跃期。",
            },
            {
                "项目": "设备状态",
                "说明": "P95、峰值内存是示例场景。measured字段保留空值用于实际上机回填。",
            },
            {
                "项目": "BDD八阶段",
                "说明": "独立的2方法×8配对seed情景，每域在两轮中各出现一次，模拟视频组分开。",
            },
            {
                "项目": "拟合按动作结算",
                "说明": "配置单次最多8 epochs，成员单次最多4。每次分别记录尝试步和完成步，耗用含失败前已尝试的步。",
            },
            {
                "项目": "DeModCL工作量",
                "说明": "沿用扩展对照的场景总量。动作日志与逐步拟合约束适用内部方法。",
            },
            {
                "项目": "BDD同检测器",
                "说明": "同seed、方法和阶段使用一个阶段末模拟检测器，分别给所有已获得域评分。",
            },
            {
                "项目": "BDD历史保持",
                "说明": "SeenAP按已获得独立域等权。Forget使用该域此前峰值，BWT使用首次获得时AP，历史域均排除当前域。",
            },
            {
                "项目": "同工作量对照",
                "说明": "当前数据比较同阶段预算上限。实际工作量配对须收集实测轨迹后匹配。",
            },
        ]
    )
    additions = {
        "主结果": core_summary,
        "预测误差汇总": summary,
        "预测误差逐seed": seed_prediction,
        "模拟逐次预测": attempts,
        "阶段时间估算": phase_cost,
        "设备资源与待实测": device,
        "BDD八阶段汇总": bdd_summary,
        "BDD八阶段逐seed": bdd,
        "BDD八阶段域明细": bdd_domains,
    }
    tables.update(
        {
            sheet: [dict(data_status=STATUS, **row) for row in rows]
            for sheet, rows in additions.items()
        }
    )
