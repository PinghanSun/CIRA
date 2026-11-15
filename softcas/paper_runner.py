"""Experiment binding and execution."""

from __future__ import annotations
import copy
import json
import hashlib
from pathlib import Path
from .paper_protocol import INTERNAL, settings
from .state import digest


def bind_file(binding):
    if (
        not isinstance(binding, dict)
        or not binding.get("path")
        or not binding.get("sha256")
    ):
        raise ValueError("Unbound file: path and SHA-256 required")
    p = Path(binding["path"]).resolve()
    if (
        not p.is_file()
        or hashlib.sha256(p.read_bytes()).hexdigest() != binding["sha256"]
    ):
        raise ValueError("Input absent or changed: " + str(p))
    return str(p)


def condition_key(recipe):
    return digest(
        {
            k: recipe[k]
            for k in (
                "dataset",
                "detector",
                "seed",
                "alpha",
                "perturbation",
                "exposure",
            )
        }
    )


def compile_matrix(matrix, bindings, config_root, output):
    from .config import _read_config, merge_dict, validate_config
    from .data import validate_registry, registry_options

    if matrix.get("schema") != "cira.paper_matrix.20261113":
        raise ValueError("Current paper matrix required")
    if matrix["protocol_sha256"] != digest(settings()):
        raise ValueError("Protocol changed after matrix construction")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("New compile directory required")
    formal = bindings.get("formal", True)
    if type(formal) is not bool:
        raise ValueError("Explicit formal boolean required")
    output.mkdir(parents=True, exist_ok=True)
    ready = []
    blocked = []
    for row in matrix["jobs"]:
        try:
            extra_inputs = []
            r = row["recipe"]
            key = "/".join(
                map(str, (r["dataset"], r["detector"], r["initialization"], r["seed"]))
            )
            cell = bindings.get("cells", {}).get(key)
            if cell is None:
                raise ValueError("Missing cell binding " + key)
            files = {
                k: bind_file(cell[k])
                for k in ("checkpoint", "registry", "detector_config", "contact_trace")
            }
            if r["phase"] == "development":
                final_path = files["registry"]
                files["registry"] = bind_file(cell["development_registry"])
                extra_inputs.append(final_path)
            nondefault = (
                r["alpha"] != 1.0
                or r["perturbation"] != "normal"
                or (
                    r["dataset"] == "ControlledFog"
                    and r["exposure"] == "single_exposure"
                )
            )
            if nondefault:
                cond = bindings.get("conditions", {}).get(condition_key(r))
                if cond is None:
                    raise ValueError(
                        "Bind the actual transformed condition; reference data cannot substitute"
                    )
                files.update(
                    {k: bind_file(cond[k]) for k in ("registry", "contact_trace")}
                )
            registry = json.loads(Path(files["registry"]).read_text(encoding="utf-8"))
            if not registry.get("records"):
                raise ValueError("Empty native registry")
            if r["phase"] == "development":
                if registry.get("phase") not in ("DEV", "development"):
                    raise ValueError("Explicit independent DEV registry phase required")
                final = json.loads(Path(final_path).read_text(encoding="utf-8-sig"))
                roles = {
                    row.get("allocation_group", row["scene_id"]): row["role"]
                    for row in final["records"]
                }
                if any(
                    roles.get(row.get("allocation_group", row["scene_id"])) != "D"
                    for row in registry["records"]
                ):
                    raise ValueError(
                        "Every DEV group must belong only to final D, never final V/A/T"
                    )
            validate_registry(registry["records"], formal, **registry_options(registry))
            if registry.get("exposure_policy", "single_exposure") != r["exposure"]:
                raise ValueError("Stream exposure policy differs from recipe")
            cfg = _read_config(Path(config_root) / "native_r1000.json")
            cfg = merge_dict(cfg, cell.get("overrides", {}))
            cfg.update(
                seed=r["seed"],
                method=r["method"] if r["method"] in INTERNAL else "softcas",
                ablation=r["ablation"],
                purpose="FORMAL" if formal else "DEVELOPMENT",
                attempts=r["attempts"],
                round_budget=None,
                paper_protocol=matrix["protocol_sha256"],
            )
            cfg["fitting"]["stage_epochs"] = r["stage_epochs"]
            cfg["budgets"]["search"]["E"] = 1 + r["attempts"]
            if r["phase"] != "archived_round_protocol":
                cfg["fitting"].update(local_epochs=8, receiving_epochs=4)
            if "rounds" in r:
                cfg["round_budget"] = {
                    "per_client": r["rounds"],
                    "stages": 4,
                    "epochs_per_round": 2,
                }
            cfg["budgets"]["search"]["F"] *= r["F"]
            if formal and r["phase"] != "development":
                binding = (
                    (
                        cell.get("archived_development_lock")
                        or bindings.get("archived_development_lock")
                    )
                    if r["phase"] == "archived_round_protocol"
                    else (
                        cell.get("development_lock") or bindings.get("development_lock")
                    )
                )
                lock_path = bind_file(binding)
                lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
                extra_inputs.append(lock_path)
                if r["phase"] == "archived_round_protocol":
                    if lock.get("role") != "DEV" or lock.get("locked") is not True:
                        raise ValueError("Locked archived DEV selection required")
                    selected = lock
                else:
                    from .paper_development import selected_budget

                    selected = selected_budget(lock, r["dataset"] + "/" + r["detector"])
                if (
                    selected.get("stage_epochs") != r["stage_epochs"]
                    or selected.get("attempts") != r["attempts"]
                ):
                    raise ValueError(
                        "DEV lock and actual fitting/attempt settings differ; freeze the matrix first"
                    )
            validate_config(cfg, formal)
            dest = output / (row["id"] + ".config.json")
            dest.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            job = {
                **{
                    k: r[k]
                    for k in ("dataset", "detector", "method", "seed", "ablation")
                },
                **files,
                "config": str(dest),
                "formal": formal,
                "name": row["id"],
                "registry_id": digest(registry),
                "clients": ["0", "1", "2", "3"],
                "output": str(Path(bindings["runs_root"]).resolve() / row["id"]),
                "paper_recipe": r,
                "paper_job_id": row["id"],
                "executor": "internal",
                "input_manifest": {},
            }
            job["overrides"] = {}
            if r["method"] not in INTERNAL:
                from .external import validate_adapter

                adapter = bindings.get("external_adapters", {}).get(r["method"])
                if not adapter:
                    raise ValueError(
                        "Bind the independent external runner for " + r["method"]
                    )
                check = validate_adapter(adapter, job)
                job.update(
                    executor="external",
                    adapter=copy.deepcopy(adapter),
                    expected_adapter_sha256=check["adapter_sha256"],
                )
                job["input_manifest"].update(check["input_manifest"])
            if formal:
                from .study_execution import validate_native_metadata

                validate_native_metadata(job, cfg)
            if cfg.get("cost_profile"):
                extra_inputs.append(str(Path(cfg["cost_profile"]).resolve()))
            for p in [*files.values(), str(dest), *extra_inputs]:
                job["input_manifest"][p] = hashlib.sha256(
                    Path(p).read_bytes()
                ).hexdigest()
            ready.append(job)
        except (KeyError, ValueError, FileNotFoundError, TypeError) as e:
            blocked.append({"id": row["id"], "recipe": row["recipe"], "reason": str(e)})
    groups = []
    if ready and bindings.get("execution_topology") == "physical_nodes":
        from .experiment import _physical_plan

        ready, groups = _physical_plan(
            ready, {**bindings, "clients": ["0", "1", "2", "3"]}
        )
        for job in ready:
            path = Path(job["config"]).resolve()
            job["input_manifest"][str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    result = {
        "schema": "cira.paper_compiled.20261113",
        "jobs": ready,
        "blocked": blocked,
        "trajectory_groups": groups,
        "ready": len(ready),
        "blocked_count": len(blocked),
        "training_started": False,
        "matrix_sha256": digest(matrix),
        "execution_topology": bindings.get(
            "execution_topology", "single_host_verifier"
        ),
    }
    (output / "compiled.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def execute_job(job, resume=False):
    """Explicit execution. Refuse silently treating one host as four physical nodes."""
    if job.get("formal") and job.get("node_id") is None:
        raise ValueError(
            "Formal four-device jobs must be scheduled through production physical run-plan with a node_id"
        )
    for p, h in job["input_manifest"].items():
        bind_file({"path": p, "sha256": h})
    if job["executor"] == "external":
        from .external import execute_external

        return execute_external(job, resume)
    from .experiment import execute_trajectory

    return execute_trajectory(job, resume)
