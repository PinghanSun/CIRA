"""Supplementary study commands."""

import argparse, copy, hashlib, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = ROOT.parent / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Auxiliary source changed: " + row["path"])
    return manifest


def run(kind, output, case_ids=(), all_cases=False):
    manifest = verify_sources()
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Write auxiliary results outside frozen study sources")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve prior study results")
    if kind in ("feature", "fixed"):
        sub = "independent_feature" if kind == "feature" else "fixed_reference"
        cases = json.loads(
            (ROOT / sub / "inputs/cases.json").read_text(encoding="utf-8-sig")
        )
        selected = (
            cases if all_cases else [c for c in cases if c["case_id"] in set(case_ids)]
        )
        if not selected or (
            not all_cases and {c["case_id"] for c in selected} != set(case_ids)
        ):
            raise ValueError(
                "Select existing --case-id values or explicitly request --all"
            )
        output.mkdir(parents=True, exist_ok=True)
        records = []
        if kind == "feature":
            engine = load("cira_frozen_feature_engine", ROOT / sub / "src/engine.py")
            profiles = json.loads(
                (ROOT / sub / "inputs/profiles.json").read_text(encoding="utf-8-sig")
            )
            grouped = defaultdict(list)
            for c in selected:
                grouped[(c["profile"], c["round_budget"])].append(c)
            for (profile, _), rows in sorted(grouped.items()):
                for i in range(0, len(rows), 8):
                    records += engine.run_batch(
                        rows[i : i + 8], profiles[profile], output, capture_stages=True
                    )
            scope = "independent_feature_model_AP_surrogate_not_native_detector"
        else:
            engine = load(
                "cira_frozen_numerical_generator", ROOT / sub / "generate_scenario.py"
            )
            engine.DATA = output
            engine.SOURCE = ROOT / sub / "inputs/cases.json"
            grouped = defaultdict(list)
            for c in selected:
                grouped[c["round_budget"]].append(c)
            batch = 0
            for _, rows in sorted(grouped.items()):
                for i in range(0, len(rows), 32):
                    result = engine.generate_batch(
                        rows[i : i + 32], batch, write_files=True
                    )
                    records += result[0]
                    batch += 1
            scope = "fixed_reference_numerical_scenario_not_native_detector"
        (output / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = {
            "study": kind,
            "runs": len(records),
            "scope": scope,
            "native_detector_training": False,
        }
    elif kind == "fixed-summary":
        output.mkdir(parents=True, exist_ok=True)
        source = ROOT / "fixed_reference/frozen_results/results.json"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = json.loads(source.read_text(encoding="utf-8-sig"))
        bindings = []
        data = output / "endpoints"
        data.mkdir()
        for record in rows:
            setting = record["setting"]
            if setting["experiment"] != "N01" or setting["seed"] not in range(
                1001, 1006
            ):
                continue
            raw = record["metrics"][0]
            if not isinstance(raw, (float, int)) or not 0 <= raw <= 100:
                raise ValueError("Recorded numerical AP is in percent")
            dataset = setting["dataset"].replace(
                "ControlledNATIVE_RUN", "ControlledFog"
            )
            method = setting["method"].split("（")[0].strip()
            method = (
                "softcas"
                if method == "CIRA"
                else "serial"
                if method == "Serial"
                else method
            )
            row = {
                "study_kind": "numerical_record",
                "complete": True,
                "cell": dataset + "/" + setting["profile"],
                "method": method,
                "seed": setting["seed"],
                "seen": raw / 100,
                "recorded_units": "percent",
                "output_units": "fraction",
                "source_run_id": record["run_id"],
                "source_sha256": source_hash,
            }
            path = data / (record["run_id"] + ".json")
            path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            bindings.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        if len(bindings) != 420:
            raise ValueError("Frozen N01 comparison requires 420 records")
        renderer = load(
            "cira_frozen_record_renderer", ROOT.parent / "analysis/render_results.py"
        )
        spec = {
            "evidence_scope": "numerical_record",
            "trajectory_files": bindings,
            "seeds": list(range(1001, 1006)),
            "reference": "softcas",
            "comparators": ["serial", "DeModCL"],
        }
        renderer.render(spec, output / "summary")
        result = {
            "study": kind,
            "scope": "fixed_reference_N01_numerical_records_not_native_detector",
            "records": 420,
            "native_detector_training": False,
        }
    elif kind == "fixed-analysis":
        output.mkdir(parents=True, exist_ok=True)
        path = ROOT / "fixed_reference/audit"
        sys.path.insert(0, str(path))
        common = load("dev_selection_closure", path / "dev_selection_closure.py")
        common.DATA = ROOT / "fixed_reference/frozen_results"
        common.OUT = output
        availability = load(
            "d05_quality_availability", path / "d05_quality_availability.py"
        )
        availability.main()
        report = load(
            "cira_frozen_fixed_quality_report",
            path / "fixed_reference_quality_outcomes.py",
        )
        report.main()
        result = {
            "study": kind,
            "scope": "retrospective_fixed_reference_records_not_native_detector",
            "native_detector_training": False,
        }
    elif kind in ("D-only", "continuation"):
        output.mkdir(parents=True, exist_ok=True)
        path = ROOT / "numerical/mechanisms"

        for source in path.glob("*.py"):
            (output / source.name).write_bytes(source.read_bytes())
        sys.path.insert(0, str(path))
        base = load(
            "run_d_only_logit_diagnostics", path / "run_d_only_logit_diagnostics.py"
        )
        base.HERE = output
        base.OLD = ROOT / "numerical/frozen_fixture"
        base.REGISTRY = ROOT / "numerical/references/results.json"
        base.METHOD = ROOT.parent / "softcas/model.py"
        base.main()
        if kind == "continuation":
            reverse = load("cira_frozen_d_only_reverse", path / "run_d_only_reverse.py")
            reverse.HERE = output
            reverse.main()
        result = {
            "study": kind,
            "scope": "linear_MSE_arithmetic_not_native_AP",
            "native_detector_training": False,
        }
    else:
        raise ValueError("Unknown auxiliary study")
    result.update(
        source_manifest_sha256=hashlib.sha256(
            (ROOT / "SOURCE_MANIFEST.json").read_bytes()
        ).hexdigest(),
        output=str(output),
    )
    (output / "execution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "study",
        choices=(
            "feature",
            "fixed",
            "fixed-summary",
            "fixed-analysis",
            "D-only",
            "continuation",
            "verify",
            "list",
        ),
    )
    p.add_argument("--output")
    p.add_argument("--case-id", action="append", default=[])
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if a.study == "verify":
        r = {"verified_files": len(verify_sources()["files"])}
    elif a.study == "list":
        r = {
            name: [
                c["case_id"]
                for c in json.loads((ROOT / path / "inputs/cases.json").read_text())[
                    :10
                ]
            ]
            for name, path in [
                ("feature", "independent_feature"),
                ("fixed", "fixed_reference"),
            ]
        }
    else:
        if not a.output:
            p.error("--output is required")
        r = run(a.study, a.output, a.case_id, a.all)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
