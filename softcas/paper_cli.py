"""Experiment commands."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(prog="python -m softcas.paper_cli")
    s = p.add_subparsers(dest="command", required=True)
    x = s.add_parser("export")
    x.add_argument("--output", required=True)
    x = s.add_parser("matrix")
    group = x.add_mutually_exclusive_group()
    group.add_argument("--blocks", nargs="+")
    group.add_argument("--receiver-study", choices=("pilot", "core", "extended"))
    x.add_argument("--receiver-seeds", nargs="+", type=int)
    x.add_argument("--output", required=True)
    x = s.add_parser("compile")
    x.add_argument("--matrix", required=True)
    x.add_argument("--bindings", required=True)
    x.add_argument("--output", required=True)
    x = s.add_parser("run-job")
    x.add_argument("--plan", required=True)
    x.add_argument("--id", required=True)
    x.add_argument("--node-id")
    x.add_argument("--resume", action="store_true")
    x = s.add_parser("initialize")
    x.add_argument("--spec", required=True)
    x = s.add_parser("convert-bdd")
    x.add_argument("--spec", required=True)
    x = s.add_parser("registry")
    x.add_argument("--spec", required=True)
    x = s.add_parser("import-allocation")
    x.add_argument("--spec", required=True)
    x = s.add_parser("single-exposure")
    x.add_argument("--spec", required=True)
    x = s.add_parser("dev-lock")
    x.add_argument("--spec", required=True)
    x.add_argument("--output", required=True)
    x = s.add_parser("freeze-matrix")
    x.add_argument("--matrix", required=True)
    x.add_argument("--lock", required=True)
    x.add_argument("--output", required=True)
    x = s.add_parser("preflight")
    x.add_argument("--bindings", required=True)
    x.add_argument("--configs", required=True)
    x.add_argument("--output", required=True)
    x = s.add_parser("analyze-actions")
    x.add_argument("--input", required=True)
    x.add_argument("--unit", required=True)
    x.add_argument("--output", required=True)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
    from .paper_protocol import export, build_matrix, build_receiver_study

    from .paths import configuration_root

    config_root = configuration_root()
    if a.command == "export":
        r = export(a.output, config_root)
    elif a.command == "matrix":
        if a.receiver_seeds and not a.receiver_study:
            p.error("--receiver-seeds requires --receiver-study")
        try:
            r = (
                build_receiver_study(a.receiver_study, a.receiver_seeds)
                if a.receiver_study
                else build_matrix(blocks=a.blocks)
            )
        except ValueError as error:
            p.error(str(error))
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif a.command == "compile":
        from .paper_runner import compile_matrix

        r = compile_matrix(read(a.matrix), read(a.bindings), config_root, a.output)
    elif a.command == "run-job":
        from .paper_runner import execute_job

        plan = read(a.plan)
        jobs = [
            j
            for j in plan["jobs"]
            if j["paper_job_id"] == a.id
            and (a.node_id is None or j.get("node_id") == a.node_id)
        ]
        if len(jobs) != 1:
            raise ValueError("Select one bound job ID")
        r = execute_job(jobs[0], a.resume)
    elif a.command == "initialize":
        from .initialization import run_yolo_initialization

        r = run_yolo_initialization(read(a.spec))
    elif a.command == "convert-bdd":
        from .paper_data import convert_bdd

        r = convert_bdd(read(a.spec))
    elif a.command == "import-allocation":
        from .paper_data import import_allocation

        r = import_allocation(read(a.spec))
    elif a.command == "single-exposure":
        from .paper_data import single_exposure

        r = single_exposure(read(a.spec))
    elif a.command == "preflight":
        from .readiness import check_readiness

        r = check_readiness(a.configs, read(a.bindings))
        target = Path(a.output)
        if target.exists():
            raise FileExistsError("Preserve prior preflight report")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    elif a.command in ("dev-lock", "freeze-matrix"):
        from .paper_development import lock_stage_budget, freeze_matrix

        r = (
            lock_stage_budget(read(a.spec))
            if a.command == "dev-lock"
            else freeze_matrix(read(a.matrix), read(a.lock))
        )
        target = Path(a.output)
        if target.exists():
            raise FileExistsError("Preserve prior locks and frozen matrices")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    elif a.command == "analyze-actions":
        from .paper_analysis import prediction_references, action_summary

        rows = read(a.input)
        r = {
            "prediction": prediction_references(rows, a.unit),
            "actions": action_summary(rows),
        }
        target = Path(a.output)
        if target.exists():
            raise FileExistsError("Preserve existing analyses")
        target.write_text(
            json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    else:
        from .paper_data import make_registry

        r = make_registry(read(a.spec))
    summary = {
        k: v
        for k, v in r.items()
        if k not in ("jobs", "views", "blocked", "rows", "records")
    }
    if a.command == "analyze-actions":
        summary = {
            "output": a.output,
            "action_runs": len(r["actions"]),
            "prediction_observations": len(r["prediction"]["observations"]),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
