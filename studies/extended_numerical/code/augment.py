from pathlib import Path
from itertools import product
from collections import defaultdict, Counter
import csv, json, statistics as st, hashlib, shutil

R = Path(__file__).resolve().parents[1]
p = json.loads((R / "code/payload.json").read_bytes())
T = p["tables"]
sha = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()


def read(f):
    with Path(f).open(encoding="utf-8-sig") as s:
        return list(csv.DictReader(s))


def add(name, file, rows):
    h = list(rows[0])
    dest = R / "tables" / file
    with dest.open("w", encoding="utf-8-sig", newline="") as s:
        w = csv.DictWriter(s, fieldnames=h)
        w.writeheader()
        w.writerows(rows)
    T[name] = {"header": h, "rows": [[a.get(k) for k in h] for a in rows]}
    return rows


def from_table(name):
    return [dict(zip(T[name]["header"], r)) for r in T[name]["rows"]]


native = from_table("Study summary")
datasets = sorted({a["dataset"] for a in native})
profiles = sorted({a["profile"] for a in native})
seeds = range(1001, 1006)
cond = list(product(datasets, profiles))
basekey = lambda a: (a["dataset"], a["profile"], int(a["seed"]), a["method"])
canonical = []
for a in native:
    b = dict(a)
    b["block"] = "E2-main"
    b["metric_role"] = "final_released"
    canonical.append(b)
idx = {basekey(a): a["plan_id"] for a in canonical}
strict = from_table("Study summary")
for a in strict:
    method = "CIRA" if a["variant"] == "Full" else a["variant"]
    key = (a["dataset"], a["profile"], int(a["seed"]), method)
    if key not in idx:
        pid = "CORE-" + a["plan_id"]
        idx[key] = pid
        b = {k: None for k in native[0]}
        b.update(
            plan_id=pid,
            dataset=key[0],
            profile=key[1],
            seed=key[2],
            method=method,
            block="E1-strict",
            metric_role="final_released",
            status="Study summary",
        )
        canonical.append(b)
    a.update(
        canonical_plan_id=idx[key],
        reuse_is_conditional=method in ["CIRA", "Local", "Serial", "Round-robin"],
        execution_identity_verified=False,
    )
for i, (ds, pr, seed, method) in enumerate(
    product(
        datasets, profiles, seeds, ["NAS-only", "HPO-only", "No-Refine", "Discrete"]
    ),
    1,
):
    b = {k: None for k in native[0]}
    b.update(
        plan_id=f"ABL-{i:03d}",
        dataset=ds,
        profile=pr,
        method=method,
        seed=seed,
        block="E2-other-ablations",
        metric_role="final_released",
        status="Study summary",
    )
    canonical.append(b)
    idx[ds, pr, seed, method] = b["plan_id"]
assert len(canonical) == 600 and len(set(idx.values())) == 600
result_fields = [
    "SeenAP",
    "CurrentAP",
    "MinClientSeenAP",
    "Forget_pp",
    "BWT_pp",
    "AP50",
    "AP75",
    "fit_work",
    "control_work",
    "select_eval_work",
    "audit_work",
    "device_check_work",
    "failure_work",
    "transfer_MiB",
    "cross_family_non_entry_activation_rate",
    "failed_attempts",
    "rollback_count",
    "budget_violations",
]
for row in canonical:
    for field in result_fields:
        row[field] = None
    for field in [
        "released_checkpoint_SHA256",
        "best_V_checkpoint_SHA256",
        "audit_locked_checkpoint_SHA256",
        "release_mapping_path",
    ]:
        row[field] = None
for row in strict:
    for field in result_fields:
        row[field] = None
    row["result_owner"] = "Study summary"
all_methods = sorted({a["method"] for a in canonical})
locks = []
for i, (ds, pr, method) in enumerate(product(datasets, profiles, all_methods), 1):
    locks.append(
        dict(
            DEV_lock_id=f"LOCK-{i:03d}",
            dataset=ds,
            profile=pr,
            method=method,
            development_seeds="9001,9002,9003",
            selection_metric="Study summary",
            selected_config_SHA256=None,
            SOURCE_checkpoint_SHA256=None,
            split_manifest_SHA256=None,
            code_commit=None,
            event_manifest_SHA256=None,
            release_rule_SHA256=None,
            status="Study summary",
        )
    )
lockmap = {(a["dataset"], a["profile"], a["method"]): a["DEV_lock_id"] for a in locks}
for a in canonical:
    a["DEV_lock_id"] = lockmap[a["dataset"], a["profile"], a["method"]]
add("Study summary", "E1_E2_unique_600.csv", canonical)
add("Study summary", "strict_180.csv", strict)
add("Study summary", "dev_locks_120.csv", locks)
snapshots = []
for bi, (ds, pr, seed) in enumerate(product(datasets, profiles, seeds), 1):
    for client in range(1, 5):
        snapshots.append(
            dict(
                snapshot_id=f"SNAP-{bi:02d}-C{client}",
                seed_block=f"SEED-{bi:02d}",
                dataset=ds,
                profile=pr,
                seed=seed,
                client=client,
                stage=2,
                domain_id=None,
                entry_checkpoint_SHA256=None,
                D_manifest_SHA256=None,
                exposure_SHA256=None,
            )
        )
e3 = []
e4 = []
e5 = []
pairs = []
mapping = []
for a in snapshots:
    sid = a["snapshot_id"]
    for config, ctx in product([0, 1], [0, 1]):
        e3.append(
            dict(
                a,
                endpoint_id=f"{sid}/E3/Z{config}/C{ctx}",
                config_choice=config,
                context_choice=ctx,
                port_id=None,
                proposed_width=None,
                upstream_input_SHA256=None,
                frozen_donor_response_SHA256=None,
                effective_RMS=None,
                RMS_check_passed=None,
                AP_V=None,
                AP_T=None,
                V_context_choice=None,
                T_selection_gain=None,
                status="Study summary",
            )
        )
    for variant in ["Uniform", "Frozen-parent-logits", "D-only-from-same-parent"]:
        e4.append(
            dict(
                a,
                endpoint_id=f"{sid}/E4/{variant}",
                variant=variant,
                parent_logits_SHA256=None,
                local_parameters_SHA256=None,
                members_SHA256=None,
                admitted_optimizer_steps=None,
                executed_optimizer_steps=None,
                gradient_norm=None,
                logit_change_norm=None,
                prediction_change_norm=None,
                AP50_95_T=None,
                AP75_T=None,
                status="Study summary",
            )
        )
    for variant in ["Inherited", "Reset-local-only", "Reset-logits-only", "Reset-both"]:
        e5.append(
            dict(
                a,
                endpoint_id=f"{sid}/E5/{variant}",
                variant=variant,
                common_edit_SHA256=None,
                config_after_edit_SHA256=None,
                members_SHA256=None,
                phi_initial_SHA256=None,
                beta_initial_SHA256=None,
                entry_H_same=True,
                optimizer_moments_reset_all=True,
                actual_data_visits=None,
                actual_optimizer_steps=None,
                actual_work=None,
                work_unit=None,
                AP_T=None,
                status="Study summary",
            )
        )
    for variant in ["Reset-local-only", "Reset-logits-only", "Reset-both"]:
        pair_id = f"{sid}/Inherited-vs-{variant}"
        pairs.append(
            dict(
                a,
                pair_id=pair_id,
                inherited_endpoint=f"{sid}/E5/Inherited",
                other_endpoint=f"{sid}/E5/{variant}",
                same_stage=None,
                same_domain=None,
                same_exposure=None,
                same_edit=None,
                work_inherited=None,
                work_other=None,
                relative_work_gap=None,
                threshold=0.05,
                match_status="Study summary",
                exclusion_reason=None,
                AP_T_difference=None,
            )
        )
        if variant in ["Reset-logits-only", "Reset-both"]:
            old = "Weights-only" if variant == "Reset-logits-only" else "ColdLocal"
            mapping.append(
                dict(
                    old_pair_id=sid.replace("SNAP", "ANCHOR") + f"/Inherited-vs-{old}",
                    new_pair_id=pair_id,
                    old_registration_only=True,
                    old_result_reused=False,
                    note="Study summary",
                )
            )
add("Study summary", "E3_factorial_480.csv", e3)
add("Study summary", "E4_D_only_360.csv", e4)
add("Study summary", "E5_inheritance_480.csv", e5)
add("Study summary", "E5_pairs_360.csv", pairs)
add("Study summary", "legacy240_to_E5.csv", mapping)
robust = []
for i, (ds, pr, seed, method, stress) in enumerate(
    product(
        datasets,
        profiles,
        seeds,
        ["CIRA", "Serial", "pFedDIL", "DeModCL"],
        [
            "normal",
            "weak_donor",
            "empty_donor",
            "peer_visibility_half",
            "one_half_payload_truncation",
        ],
    ),
    1,
):
    robust.append(
        dict(
            plan_id=f"E7-{i:03d}",
            dataset=ds,
            profile=pr,
            seed=seed,
            method=method,
            stress=stress,
            normal_main_reference=idx[ds, pr, seed, method],
            normal_reference_reuse_verified=False,
            event_rule_SHA256=None,
            SeenAP=None,
            MinClientSeenAP=None,
            own_normal_difference=None,
            paired_difference_in_differences=None,
            transfer_MiB=None,
            rejections=None,
            rollbacks=None,
            status="Study summary",
        )
    )
add("Study summary", "E7_600.csv", robust)
for axis, values, name in [
    ("R", [300, 500, 1000], "Study summary"),
    ("F", [0.5, 1, 2], "Study summary"),
]:
    rows = []
    for i, (ds, pr, seed, method, level) in enumerate(
        product(datasets, profiles, seeds, ["CIRA", "Serial"], values), 1
    ):
        rows.append(
            dict(
                plan_id=f"E8-{axis}-{i:03d}",
                dataset=ds,
                profile=pr,
                seed=seed,
                method=method,
                axis=axis,
                level=level,
                rounds=level if axis == "R" else 1000,
                stages=4,
                normal_main_reference=idx[ds, pr, seed, method],
                normal_anchor=level == (1000 if axis == "R" else 1),
                reference_reuse_verified=False,
                SeenAP=None,
                CurrentAP=None,
                actual_work=None,
                status="Study summary",
            )
        )
    add(name, f"E8_{axis}_180.csv", rows)
param = []
for name, levels in [
    ("ridge_lambda", [0.1, 1, 10]),
    ("kappa", [0, 0.01, 0.1]),
    ("eta_beta", [0.003, 0.01, 0.03]),
    ("M", [32, 48, 64]),
]:
    for level in levels:
        param.append(
            dict(
                parameter=name,
                candidate_level=level,
                native_center_config_SHA256=None,
                other_parameters_fixed=None,
                DEV_lock_verified=False,
                old_Q0_Q2_not_relabelled=True,
                native_result=None,
                status="Study summary",
            )
        )
add("Study summary", "E8_parameter_lock.csv", param)
devices = []
for a in native:
    for node, repeat in product(range(1, 5), range(1, 4)):
        devices.append(
            dict(
                session_plan_id=f"E9/{a['plan_id']}/N{node}/REP{repeat}",
                canonical_plan_id=a["plan_id"],
                dataset=a["dataset"],
                profile=a["profile"],
                method=a["method"],
                model_seed=a["seed"],
                planned_node=node,
                repeat=repeat,
                warmup_requests=50,
                synchronized_measured_requests=200,
                actual_device_id=None,
                power_mode=None,
                precision=None,
                checkpoint_SHA256=None,
                latency_requests_path=None,
                power_series_path=None,
                P50_ms=None,
                P95_ms=None,
                energy_J=None,
                peak_RSS_MiB=None,
                peak_PSS_MiB=None,
                CUDA_allocated_MiB=None,
                update_time_s=None,
                status="Study summary",
            )
        )
assert len(devices) == 5040
add("Study summary", "E9_device5040.csv", devices)
panels = from_table("Study summary")
for a in panels:
    if a["method"] == "Local":
        a["method"] = "DeModCL"
        a["panel_id"] = a["panel_id"].replace("Local", "DeModCL")
    a["method_qualification_verified"] = False
    a["status"] = "Study summary"
add("Study summary", "prediction_panels_48.csv", panels)
figsrc = Path("data/figure_content_inventory.json")
paper = Path("data/paper")
figs = []
for a in json.loads(figsrc.read_bytes()):
    if a["figure"] in [9, 10, 11]:
        path = paper / a["file"]
        dst = R / "references/existing_figures" / path.name
        dst.parent.mkdir(exist_ok=True)
        if path.exists():
            shutil.copy2(path, dst)
        figs.append(
            dict(
                figure=a["figure"],
                existing_asset=dst.relative_to(R).as_posix() if dst.exists() else None,
                existing_asset_SHA256=sha(dst) if dst.exists() else None,
                scene_mapping=None,
                model_identity=None,
                stage_checkpoint=None,
                display_threshold=None,
                crop_registry=None,
                SoftCAS_CIRA_identity_verified=False,
                C_DFL_identity_verified=False,
                status="Study summary",
            )
        )
add("Study summary", "existing_figures_binding.csv", figs)
classes = []
for a in canonical:
    for ci, cname in enumerate(
        ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    ):
        classes.append(
            dict(
                endpoint_id=a["plan_id"],
                block=a["block"],
                dataset=a["dataset"],
                profile=a["profile"],
                seed=a["seed"],
                class_index=ci,
                class_name=cname,
                GT_support=None,
                AP50=None,
                AP75=None,
                AP50_95=None,
                evaluation_id=None,
                status="Study summary",
            )
        )
for a in e4:
    for ci, cname in enumerate(
        ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
    ):
        classes.append(
            dict(
                endpoint_id=a["endpoint_id"],
                block="E4",
                dataset=a["dataset"],
                profile=a["profile"],
                seed=a["seed"],
                class_index=ci,
                class_name=cname,
                GT_support=None,
                AP50=None,
                AP75=None,
                AP50_95=None,
                evaluation_id=None,
                status="Study summary",
            )
        )
add("Study summary", "class_AP_7680.csv", classes)

main = read(R / "references/desktop_package/data/endpoints.csv")
groups = defaultdict(list)
for a in main:
    groups[a["dataset"], a["profile"], a["method"]].append(a)
summary = []
for (ds, pr, method), block in sorted(groups.items()):
    for metric in [
        "AP50",
        "AP75",
        "Domain0_AP",
        "Domain1_AP",
        "Domain2_AP",
        "Domain3_AP",
    ]:
        x = [float(a[metric]) for a in block]
        summary.append(
            dict(
                dataset=ds,
                profile=pr,
                method=method,
                metric=metric,
                n=len(x),
                mean=st.mean(x),
                sample_SD=st.stdev(x),
                scope="Study summary",
            )
        )
assert len(summary) == 504
add("Study summary", "existing_504_mean_SD.csv", summary)
by = {}
for (ds, pr, method), block in groups.items():
    by[ds, pr, method] = {
        k: st.mean(float(a[k]) for a in block)
        for k in ["SeenAP", "Forget_pp", "BWT_pp", "transfer_MiB"]
    }
gaps = [
    by[ds, pr, "CIRA"]["SeenAP"] - by[ds, pr, "pFedDIL"]["SeenAP"] for ds, pr in cond
]
serial_saving = [
    100
    * (1 - by[ds, pr, "CIRA"]["transfer_MiB"] / by[ds, pr, "Serial"]["transfer_MiB"])
    for ds, pr in cond
]
demod_extra = [
    100
    * (by[ds, pr, "CIRA"]["transfer_MiB"] / by[ds, pr, "DeModCL"]["transfer_MiB"] - 1)
    for ds, pr in cond
]
conclusions = [
    dict(
        item="Study summary",
        lower=min(gaps),
        upper=max(gaps),
        unit="pp",
        scope="Study summary",
    ),
    dict(
        item="Study summary",
        lower=sum(
            by[ds, pr, "pFedDIL"]["Forget_pp"] < by[ds, pr, "CIRA"]["Forget_pp"]
            for ds, pr in cond
        ),
        upper=6,
        unit="conditions",
        scope="Study summary",
    ),
    dict(
        item="Study summary",
        lower=sum(
            by[ds, pr, "pFedDIL"]["BWT_pp"] > by[ds, pr, "CIRA"]["BWT_pp"]
            for ds, pr in cond
        ),
        upper=6,
        unit="conditions",
        scope="Study summary",
    ),
    dict(
        item="Study summary",
        lower=min(serial_saving),
        upper=max(serial_saving),
        unit="percent",
        scope="Study summary",
    ),
    dict(
        item="Study summary",
        lower=min(demod_extra),
        upper=max(demod_extra),
        unit="percent",
        scope="Study summary",
    ),
]
add("Study summary", "existing_claim_numeric_checks.csv", conclusions)
events = read(R / "references/desktop_package/data/events.csv")
eg = defaultdict(list)
for a in events:
    eg[a["run_id"]].append(a)
logs = []
for a in main:
    block = eg.get(a["run_id"], [])
    logs.append(
        dict(
            run_id=a["run_id"],
            dataset=a["dataset"],
            profile=a["profile"],
            method=a["method"],
            event_rows=len(block),
            event_types=";".join(sorted({e["event"] for e in block})),
            full_parent_child_identity_present=False,
            preexecution_gain_label_present=False,
            released_checkpoint_link_verified=False,
            status="Study summary",
        )
    )
add("Study summary", "E6_existing_event_inventory.csv", logs)
rules = from_table("Study summary")
rules = [a for a in rules if a["id"] not in ["P14", "P23"]]
rules += [
    dict(
        id="E5-RULE",
        item="Study summary",
        rule="Study summary.\n",
    ),
    dict(
        id="E9-RULE",
        item="Study summary",
        rule="Study summary.\n",
    ),
    dict(
        id="REUSE",
        item="Study summary",
        rule="Study summary.\n",
    ),
    dict(
        id="E7-RULE",
        item="Study summary",
        rule="Study summary.\n",
    ),
    dict(
        id="E8-RULE",
        item="Study summary",
        rule="Study summary.\n",
    ),
]
add("Study summary", "protocol_aligned.csv", rules)
status = [
    (
        "E0",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E1",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E2",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E3",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E4",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E5",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E6",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E7",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E8",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
    (
        "E9",
        "Study summary",
        "Study summary",
        "Study summary",
    ),
]
add(
    "Study summary",
    "E0_E9_status.csv",
    [
        dict(id=i, item=n, completed_material=c, remaining=r, native_complete=False)
        for i, n, c, r in status
    ],
)
add(
    "Study summary",
    "acceptance_aligned.csv",
    [
        dict(id=i, protocol_and_registry_ready=True, native_complete=False, required=r)
        for i, _, _, r in status
    ],
)
keep = [
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
    "Study summary",
]
index = []
for f in (R / "references").rglob("*"):
    if f.is_file():
        index.append(
            dict(
                path=f.relative_to(R).as_posix(),
                bytes=f.stat().st_size,
                SHA256=sha(f),
                scope="Study summary",
            )
        )
add("Study summary", "evidence_aligned.csv", index)
p["tables"] = {name: T[name] for name in keep}
p["order"] = ["Study summary"] + keep
p["cover"] = [
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
    [
        "Study summary",
        "Study summary.\n",
    ],
]
p["star_fields"] = list(
    set(
        result_fields
        + [
            "AP50_95",
            "AP50_95_T",
            "AP75_T",
            "AP_V",
            "AP_T",
            "AP_T_difference",
            "GT_support",
            "P50_ms",
            "P95_ms",
            "energy_J",
            "peak_RSS_MiB",
            "peak_PSS_MiB",
            "CUDA_allocated_MiB",
            "update_time_s",
            "effective_RMS",
            "T_selection_gain",
            "gradient_norm",
            "logit_change_norm",
            "prediction_change_norm",
            "actual_optimizer_steps",
            "actual_data_visits",
            "actual_work",
            "own_normal_difference",
            "paired_difference_in_differences",
            "rejections",
            "rollbacks",
            "scene_count",
            "image_count",
            "box_count",
            "images_with_class",
            "GT_boxes",
            "relative_work_gap",
            "native_result",
        ]
    )
)
(R / "code/payload.json").write_text(
    json.dumps(p, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    encoding="utf-8",
)

active = {
    f"E1_E2_unique_600.csv",
    "strict_180.csv",
    "dev_locks_120.csv",
    "E3_factorial_480.csv",
    "E4_D_only_360.csv",
    "E5_inheritance_480.csv",
    "E5_pairs_360.csv",
    "legacy240_to_E5.csv",
    "E7_600.csv",
    "E8_R_180.csv",
    "E8_F_180.csv",
    "E8_parameter_lock.csv",
    "E9_device5040.csv",
    "prediction_panels_48.csv",
    "existing_figures_binding.csv",
    "class_AP_7680.csv",
    "existing_504_mean_SD.csv",
    "E6_existing_event_inventory.csv",
    "protocol_aligned.csv",
    "E0_E9_status.csv",
    "acceptance_aligned.csv",
    "evidence_aligned.csv",
    "split_census_144.csv",
    "class_support_1152.csv",
    "existing_MSE_summary.csv",
}
legacy = R / "legacy_registrations"
legacy.mkdir(exist_ok=True)
for f in (R / "tables").glob("*.csv"):
    if f.name not in active:
        f.replace(legacy / f.name)
(R / "qa/aligned_counts.json").write_text(
    json.dumps(
        {k: len(v["rows"]) for k, v in p["tables"].items()},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "sheets": len(p["order"]),
            "counts": {k: len(v["rows"]) for k, v in p["tables"].items()},
        },
        ensure_ascii=True,
    )
)
