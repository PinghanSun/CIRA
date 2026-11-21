"""Supplementary study commands."""

import argparse, hashlib, importlib.util, json, sys
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
    verify_sources()
    output = Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Write auxiliary results outside frozen study sources")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve prior study results")
    if kind != "feature":
        raise ValueError("Unknown study")
    study = ROOT / "independent_feature"
    cases = json.loads((study / "inputs/cases.json").read_text(encoding="utf-8"))
    requested = set(case_ids)
    selected = cases if all_cases else [c for c in cases if c["case_id"] in requested]
    if not selected or (
        not all_cases and {c["case_id"] for c in selected} != requested
    ):
        raise ValueError("Select existing --case-id values or explicitly request --all")
    output.mkdir(parents=True, exist_ok=True)
    engine = load("cira_feature_engine", study / "src/engine.py")
    profiles = json.loads((study / "inputs/profiles.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for case in selected:
        grouped[(case["profile"], case["round_budget"])].append(case)
    records = []
    for (profile, _), rows in sorted(grouped.items()):
        for i in range(0, len(rows), 8):
            records.extend(
                engine.run_batch(
                    rows[i : i + 8], profiles[profile], output, capture_stages=True
                )
            )
    (output / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "study": kind,
        "runs": len(records),
        "scope": "independent_feature_model_AP_surrogate_not_native_detector",
        "native_detector_training": False,
    }
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
    p.add_argument("study", choices=("feature", "verify", "list"))
    p.add_argument("--output")
    p.add_argument("--case-id", action="append", default=[])
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if a.study == "verify":
        r = {"verified_files": len(verify_sources()["files"])}
    elif a.study == "list":
        case_file = ROOT / "independent_feature/inputs/cases.json"
        cases = json.loads(case_file.read_text(encoding="utf-8"))
        r = {"feature": [case["case_id"] for case in cases[:10]]}
    else:
        if not a.output:
            p.error("--output is required")
        r = run(a.study, a.output, a.case_id, a.all)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
