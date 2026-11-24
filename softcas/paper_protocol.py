"""Experiment settings and study matrix."""

from __future__ import annotations
import copy
import itertools
import json
from pathlib import Path
from .state import digest

LABELS = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
DETECTORS = {
    "YOLOv8n": {
        "modules": [4, 6, 9, 15, 18, 21],
        "channels": [64, 128, 256, 64, 128, 256],
    },
    "YOLO11n": {
        "modules": [4, 6, 10, 16, 19, 22],
        "channels": [128, 128, 256, 64, 128, 256],
    },
    "YOLO26n": {
        "modules": [4, 6, 10, 16, 19, 22],
        "channels": [128, 128, 256, 64, 128, 256],
    },
}
INTERNAL = (
    "local",
    "local_nas_hpo",
    "fixed_collaboration",
    "serial",
    "round_robin",
    "softcas",
)
EXTERNAL = (
    "DPFL",
    "pFedDIL",
    "DA-DPFL",
    "DeModCL",
    "DCFCL",
    "dFLMoE",
    "PerFedSDE-NAS",
    "DSpodFL",
)
METHODS = INTERNAL + EXTERNAL
SEEDS = (1001, 1002, 1003, 1004, 1005)
DEV_SEEDS = (9001, 9002, 9003)
ABLATIONS = ("nas_only", "hpo_only", "no_refine", "no_handoff", "discrete", "no_gain")


def settings():
    """Return a fresh JSON-compatible protocol. Unknown DEV values remain null."""
    return {
        "schema": "cira.paper_protocol.20261113",
        "revision": "20261114",
        "labels": list(LABELS),
        "clients": 4,
        "stages": 4,
        "seeds": list(SEEDS),
        "dev_seeds": list(DEV_SEEDS),
        "detectors": copy.deepcopy(DETECTORS),
        "methods": list(METHODS),
        "datasets": {
            "ACDC": {
                "domains": ["fog", "night", "rain", "snow"],
                "online_rgb": 1600,
                "roles": {"D": 1216, "V": 192, "A": 192},
                "reporting_rgb": 406,
                "per_client_stage": {"D": 76, "V": 12, "A": 12},
                "known_D_groups": 1208,
                "allocation_D_groups": 1206,
                "exposure": "single_exposure",
            },
            "ControlledFog": {
                "domains": ["clear", "fog_005", "fog_010", "fog_020"],
                "online_bases": 2375,
                "online_rgb": 9500,
                "roles": {"D": 7036, "V": 1232, "A": 1232},
                "reporting_bases": 500,
                "reporting_rgb": 2000,
                "exposure": "repeated_content",
            },
            "BDD100K": {
                "train_rgb": 70000,
                "validation_rgb": 10000,
                "test_rgb": 20000,
                "role_targets": {"D": 56000, "V": 7000, "A": 7000},
                "role_fractions": {"D": 0.8, "V": 0.1, "A": 0.1},
                "domain_priority": ["night", "adverse", "overcast", "clear"],
                "unknown_attributes": "reserve",
                "exposure": "single_exposure",
            },
        },
        "partition": {
            "alpha": 1.0,
            "clients": 4,
            "whole_allocation_groups": True,
            "grouping_basis": "known_identity",
            "role_and_client_shared_across_appearances": True,
            "scene_independence_verified": False,
            "dev_return": "D_only_after_lock_and_reset",
        },
        "initialization": {
            "source_bases": 600,
            "fit_bases": 540,
            "clear_validation_rgb": 60,
            "fog_validation_rgb": 180,
            "C0": {"labeled_rgb": 540, "unlabeled_rgb": 0},
            "C1": {"labeled_rgb": 2160, "unlabeled_rgb": 0},
            "C2": {"labeled_rgb": 2160, "unlabeled_rgb": 19997},
            "teacher_momentum": None,
            "pseudo_confidence": None,
            "unlabeled_weight": None,
            "updates": None,
            "lr": None,
            "batch_size": None,
            "checkpoint_selection": "SOURCE_clear_validation",
        },
        "augmentation": {
            "D_only": True,
            "ACDC_views": ["original", "hflip", "rotate_minus5", "rotate_plus5"],
            "views_per_original_per_epoch": 1,
            "originals": 1216,
            "view_recipes": 4864,
            "V_A_T": "deterministic",
            "shared_local_peer_image": True,
        },
        "development": {
            "stage_epoch_candidates": [50, 100, 200],
            "attempt_candidates": [8, 16, 32],
            "provisional_stage_epochs": 100,
            "provisional_attempts": 16,
            "locked": False,
        },
        "search": {
            "local_ports": 6,
            "active_local_ports": [1, 2],
            "ratios": [1 / 16, 1 / 8],
            "local_lrs": [1e-4, 3e-4, 1e-3],
            "residual_penalties": [1e-4, 1e-3],
            "structures": 42,
            "local_choices": 252,
            "features": 25,
            "proposals_per_type": 3,
            "archive_capacity": 8,
            "retained_checkpoints": 8,
            "external_paths_per_port": 2,
            "external_paths_total": 12,
            "local_block_epochs": 8,
            "logit_block_epochs": 4,
            "lambda": 1.0,
            "kappa": 0.01,
            "beta_lr": 0.01,
            "audit_margin": 0.0,
            "native_feature_scales": None,
            "native_F_ceiling": None,
            "native_W_ceiling": None,
        },
        "runtime": {
            "ultralytics": "8.4.0",
            "image_size": 640,
            "fit_batch": 8,
            "eval_batch": 1,
            "precision": "float32",
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "frozen_base": True,
            "frozen_peer_parameters": True,
            "input_gradients": True,
            "null_logit": 0.0,
            "reset_optimizer_per_block": True,
        },
        "resources": {
            "communication_bytes": 32 * 2**20,
            "cache_bytes": 128 * 2**20,
            "checkpoint_bytes": 512 * 2**20,
            "transient_bytes": 256 * 2**20,
            "deployed_bytes": 128 * 2**20,
            "RAM_bytes": 2 * 2**30,
            "CUDA_bytes": 2 * 2**30,
            "p95_ms": 100,
        },
        "device": {
            "online_nodes": 4,
            "online_model": "Jetson Orin Nano 8GB",
            "preparation_workstation": "RTX 5080",
            "warmup": 50,
            "timed_requests": 200,
            "sessions_per_checkpoint_node": 3,
            "measurement_values": None,
        },
        "statistics": {
            "unit": "complete_federation_trajectory",
            "paired": True,
            "confidence": 0.95,
            "sd_ddof": 1,
            "primary_holm_family": 12,
            "mechanism_holm_family": 36,
            "U_holm_family": 10,
            "BDD_holm_family": 4,
            "unsupported_class": "null",
            "AP_internal": "fraction",
            "AP_report": "percent",
            "AP_difference": "percentage_points",
        },
        "diagnostics": {
            "E3": {
                "operation": "configuration_x_composition",
                "native_fixed_peer_mass": True,
                "separate_linear_diagnostic_fixed_D_RMS": True,
                "cells": 4,
            },
            "E4": {
                "operation": "D_only_logit_fit",
                "controls": ["frozen_parent_logits", "uniform"],
            },
            "E5": {
                "operation": "inherited_state",
                "controls": ["reset_local", "reset_both", "reset_logits"],
            },
            "prediction_references": ["predictor", "zero", "past_feedback_mean"],
            "noise_sigma": [0, 0.01, 0.03],
            "F_multipliers": [0.5, 1, 2],
            "round_budgets": [300, 500, 1000],
            "alpha": [0.1, 0.3, 0.5, 0.7, 1.0],
            "equal_target": True,
            "native_alpha_values_must_be_executed": True,
            "peer_perturbations": [
                "half_visibility",
                "half_payload",
                "empty_donor",
                "weak_donor",
            ],
        },
        "legacy_numerical": {
            "reference": "R1000",
            "main_controller": "Q2",
            "lambda": 10.0,
            "native_local_block_epochs": 32,
            "native_logit_block_epochs": 16,
            "feature_model_is_separate": True,
            "existing_interpolated_alpha": [0.5, 0.7],
        },
    }


def build_matrix(protocol=None, blocks=None):
    """Compile all native study conditions and analysis views, deduplicating anchors."""
    p = protocol or settings()
    if p["schema"] != "cira.paper_protocol.20261113":
        raise ValueError("Unknown protocol")
    blocks = set(
        blocks
        or [
            "primary",
            "ablations",
            "U",
            "BDD",
            "F",
            "R",
            "alpha",
            "robustness",
            "dev",
            "BDD_dev",
            "single_exposure",
        ]
    )
    legal = {
        "primary",
        "ablations",
        "U",
        "BDD",
        "F",
        "R",
        "alpha",
        "robustness",
        "dev",
        "BDD_dev",
        "single_exposure",
    }
    if not blocks <= legal:
        raise ValueError("Unknown study blocks")
    jobs = {}
    views = []

    def add(block, dataset, detector, method, seed, **kw):
        recipe = dict(
            dataset=dataset,
            detector=detector,
            method=method,
            seed=seed,
            initialization="C2",
            stage_epochs=p["development"]["provisional_stage_epochs"],
            attempts=p["development"]["provisional_attempts"],
            ablation="none",
            F=1.0,
            alpha=1.0,
            exposure="single_exposure"
            if dataset != "ControlledFog"
            else "repeated_content",
            perturbation="normal",
            phase="formal",
        )
        recipe.update(kw)
        key = digest(recipe)
        if key not in jobs:
            jobs[key] = {
                "id": key,
                "recipe": recipe,
                "status": "not_executed",
                "first_block": block,
            }
        views.append(
            {"block": block, "job_id": key, "reused": jobs[key]["first_block"] != block}
        )

    cells = list(itertools.product(("ACDC", "ControlledFog"), DETECTORS))
    for block in sorted(blocks):
        if block == "primary":
            for (d, m), a, s in itertools.product(cells, METHODS, SEEDS):
                add(block, d, m, a, s)
        elif block == "ablations":
            for (d, m), a, s in itertools.product(cells, ("none",) + ABLATIONS, SEEDS):
                add(block, d, m, "softcas", s, ablation=a)
        elif block == "U":
            for d, a, s, c in itertools.product(
                ("ACDC", "ControlledFog"),
                ("softcas", "local", "serial", "fixed_collaboration", "DeModCL"),
                SEEDS,
                ("C0", "C1", "C2"),
            ):
                add(block, d, "YOLOv8n", a, s, initialization=c)
        elif block == "BDD":
            for a, s in itertools.product(
                ("softcas", "local", "serial", "fixed_collaboration", "DeModCL"), SEEDS
            ):
                add(block, "BDD100K", "YOLOv8n", a, s)
        elif block == "BDD_dev":
            for a, s, e, k in itertools.product(
                ("softcas", "local"), DEV_SEEDS, (50, 100, 200), (8, 16, 32)
            ):
                add(
                    block,
                    "BDD100K",
                    "YOLOv8n",
                    a,
                    s,
                    stage_epochs=e,
                    attempts=k,
                    phase="development",
                )
        elif block == "single_exposure":
            for m, a, s in itertools.product(DETECTORS, ("softcas", "serial"), SEEDS):
                add(block, "ControlledFog", m, a, s, exposure="single_exposure")
        elif block == "F":
            for (d, m), a, s, f in itertools.product(
                cells, ("softcas", "serial"), SEEDS, (0.5, 1.0, 2.0)
            ):
                add(block, d, m, a, s, F=f)
        elif block == "R":
            for (d, m), a, s, r in itertools.product(
                cells, ("softcas", "serial"), SEEDS, (300, 500, 1000)
            ):
                add(
                    block,
                    d,
                    m,
                    a,
                    s,
                    rounds=r,
                    stage_epochs=r * 2 // 4,
                    attempts=64,
                    phase="archived_round_protocol",
                )
        elif block == "alpha":
            for (d, m), a, s, v in itertools.product(
                cells,
                ("softcas", "serial", "fixed_collaboration", "DeModCL", "DCFCL"),
                SEEDS,
                (0.1, 0.3, 0.5, 0.7, 1.0, "equal"),
            ):
                add(block, d, m, a, s, alpha=v)
        elif block == "robustness":
            for (d, m), a, s, v in itertools.product(
                cells,
                ("softcas", "serial", "DeModCL", "DCFCL"),
                SEEDS,
                ("normal", *p["diagnostics"]["peer_perturbations"]),
            ):
                add(block, d, m, a, s, perturbation=v)
        elif block == "dev":
            for (d, m), a, s, e, k in itertools.product(
                cells, ("softcas", "local"), DEV_SEEDS, (50, 100, 200), (8, 16, 32)
            ):
                add(block, d, m, a, s, stage_epochs=e, attempts=k, phase="development")
    return {
        "schema": "cira.paper_matrix.20261113",
        "protocol_sha256": digest(p),
        "jobs": list(jobs.values()),
        "views": views,
        "unique_jobs": len(jobs),
        "view_counts": {b: sum(v["block"] == b for v in views) for b in sorted(blocks)},
        "execution_started": False,
    }


def build_receiver_study(scope, seeds=None):
    """Register paired YOLOv8n receiver experiments using the native protocol."""
    if scope not in ("pilot", "core", "extended"):
        raise ValueError("Unknown receiver study scope")
    if scope == "pilot" and seeds is not None:
        raise ValueError("The ACDC pilot uses the registered DEV seed 9001")
    seeds = (
        (DEV_SEEDS[0],)
        if scope == "pilot"
        else tuple(range(1001, 1009) if seeds is None else seeds)
    )
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(
            type(seed) is not int
            or seed < 0
            or (scope != "pilot" and seed in DEV_SEEDS)
            for seed in seeds
        )
    ):
        raise ValueError(
            "Use distinct nonnegative reporting seeds, separate from DEV seeds"
        )
    arms = {("softcas", "none"), ("softcas", "no_handoff"), ("serial", "none")}
    if scope == "extended":
        arms.update({("softcas", "no_gain"), ("DeModCL", "none")})
    template = build_matrix(blocks=["primary", "ablations"])
    jobs = []
    for job in template["jobs"]:
        recipe = job["recipe"]
        arm = recipe["method"], recipe["ablation"]
        if (
            recipe["seed"] != SEEDS[0]
            or recipe["detector"] != "YOLOv8n"
            or arm not in arms
        ):
            continue
        if scope == "pilot" and (
            recipe["dataset"] != "ACDC" or recipe["method"] != "softcas"
        ):
            continue
        for seed in seeds:
            paired_recipe = dict(recipe, seed=seed)
            if scope == "pilot":
                paired_recipe["phase"] = "development"
            jobs.append(
                {
                    "id": digest(paired_recipe),
                    "recipe": paired_recipe,
                    "status": "not_executed",
                    "first_block": "receiver_study",
                }
            )
    return {
        **template,
        "receiver_study": scope,
        "selected_seeds": list(seeds),
        "jobs": jobs,
        "unique_jobs": len(jobs),
        "views": [
            {"block": "receiver_study", "job_id": job["id"], "reused": False}
            for job in jobs
        ],
        "view_counts": {"receiver_study": len(jobs)},
    }


def native_config(
    config_root, stage_epochs=100, attempts=16, *, seed=1001, method="softcas"
):
    from .config import _read_config, validate_config

    if stage_epochs not in (50, 100, 200) or attempts not in (8, 16, 32):
        raise ValueError("Use the registered native DEV grid")
    if method not in INTERNAL:
        raise ValueError("External methods use their independent baseline runner")
    p = _read_config(Path(config_root) / "native_r1000.json")
    p.update(
        seed=seed,
        method=method,
        round_budget=None,
        purpose="DEVELOPMENT",
        attempts=attempts,
    )
    p["fitting"]["stage_epochs"] = stage_epochs
    p["budgets"]["search"]["E"] = 1 + attempts
    p["fitting"].update(local_epochs=8, receiving_epochs=4)
    p["paper_protocol"] = "cira.paper_protocol.20261113"
    return validate_config(p)


def bdd_domain(attributes):
    weather = str(attributes.get("weather", "")).lower()
    day = str(attributes.get("timeofday", "")).lower()
    if day == "night":
        return "night"
    if weather in ("rainy", "snowy", "foggy"):
        return "adverse"
    if weather == "overcast":
        return "overcast"
    if weather in ("clear", "partly cloudy") and day in ("daytime", "dawn/dusk"):
        return "clear"
    return None


def export(directory, config_root):
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Export directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (root / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    save("protocol.json", settings())
    save("matrix.json", build_matrix())
    for e, k in itertools.product((50, 100, 200), (8, 16, 32)):
        save(f"native_E{e}_M{k}.json", native_config(config_root, e, k))
    for d, v in DETECTORS.items():
        save(
            d + ".json",
            dict(
                ports=v["modules"],
                labels=list(LABELS),
                image_size=640,
                preprocessing="rgb_letterbox114_bilinear_float32_div255",
                cached_packages=[],
            ),
        )
    for c in ("C0", "C1", "C2"):
        save(
            c + "_initialization.example.json",
            {
                "condition": c,
                "detector": "YOLOv8n",
                "seed": 1001,
                "device": "cuda:0",
                "labeled_manifest": None,
                "validation_manifest": None,
                "unlabeled_manifest": None,
                "initial_checkpoint": None,
                "initial_checkpoint_type": "coco_compatible",
                "allow_random_initialization": False,
                "output": None,
                "image_size": 640,
                "batch_size": None,
                "expected_counts": {
                    "labeled_rgb": 540 if c == "C0" else 2160,
                    "validation_rgb": 60,
                    "unlabeled_rgb": 19997 if c == "C2" else 0,
                    "labeled_bases": 540,
                },
                "recipe": {
                    "updates": None,
                    "lr": None,
                    "validation_interval": None,
                    "teacher_momentum": None,
                    "pseudo_confidence": None,
                    "unlabeled_weight": None,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "epsilon": 1e-8,
                },
            },
        )
    for name, kind, extra in [
        ("E3", "configuration_composition", {}),
        ("E4", "receiving", {"include_frozen_parent_logits": True}),
        ("E5", "continuation", {"include_reset_local_only": True}),
    ]:
        template = json.loads(
            (Path(config_root) / "mechanism.example.json").read_text(encoding="utf-8")
        )
        template["kind"] = kind
        template["diagnostic"].update(extra)
        save(name + "_native_snapshot.example.json", template)
    save(
        "registry.example.json",
        {
            "source_records": None,
            "output": None,
            "seed": 1001,
            "clients": 4,
            "domains": ["fog", "night", "rain", "snow"],
            "role_fractions": {"D": 0.76, "V": 0.12, "A": 0.12},
            "alpha": 1.0,
            "grouping_basis": "known_identity",
            "exposure_policy": "single_exposure",
            "augment_acdc": True,
            "formal": False,
            "preserve_allocation": True,
            "require_full_support": True,
            "phase": "formal",
        },
    )
    save(
        "single_exposure.example.json",
        {
            "registry": None,
            "output": None,
            "seed": 1001,
            "domains": ["clear", "fog_005", "fog_010", "fog_020"],
            "formal": False,
        },
    )
    save(
        "dev_lock.example.json",
        {
            "matrix": {"path": None, "sha256": None},
            "records": [],
            "cells": [
                "ACDC/YOLOv8n",
                "ACDC/YOLO11n",
                "ACDC/YOLO26n",
                "ControlledFog/YOLOv8n",
                "ControlledFog/YOLO11n",
                "ControlledFog/YOLO26n",
                "BDD100K/YOLOv8n",
            ],
        },
    )
    save(
        "allocation_import.example.json",
        {
            "allocations": [],
            "catalogs": [],
            "annotations": [],
            "output_directory": None,
            "seed": 1001,
            "known_identities_checked": False,
            "exposure_policy": "single_exposure",
            "augment_acdc": True,
            "phase": "formal",
        },
    )
    save(
        "BDD100K_conversion.example.json",
        {"annotations": None, "identity_map": None, "image_root": None, "output": None},
    )
    save(
        "bindings.example.json",
        {
            "formal": True,
            "cells": {},
            "conditions": {},
            "external_adapters": {},
            "runs_root": None,
            "development_lock": None,
            "archived_development_lock": None,
            "execution_topology": "physical_nodes",
            "clients": ["0", "1", "2", "3"],
            "physical_nodes": {
                str(i): {"endpoint": {"host": None, "port": None}, "overrides": {}}
                for i in range(4)
            },
            "epoch_start_unix": None,
            "trajectory_spacing_seconds": None,
            "cell_key_format": "dataset/detector/initialization/seed",
            "cell_template": {
                "checkpoint": None,
                "registry": None,
                "development_registry": None,
                "detector_config": None,
                "contact_trace": None,
                "overrides": {},
                "development_lock": None,
            },
        },
    )
    return {"directory": str(root.resolve()), "files": len(list(root.glob("*.json")))}
