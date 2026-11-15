"""Derive tail-time admission reserves from recorded independent DEV runs."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from .state import canonical, digest
from .storage import atomic_write


def calibrate_tail(stage_directories, output, safety):
    if not math.isfinite(safety) or safety <= 1:
        raise ValueError("Use an explicit safety multiplier greater than one")
    directories = [Path(p).resolve() for p in stage_directories]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Provide distinct independent development stage directories")
    identity = None
    feedback, closing, inputs = [], [], []
    for directory in directories:
        stage_path, ledger_path = directory / "stage.json", directory / "ledger.jsonl"
        stage = json.loads(stage_path.read_bytes())
        config = stage["config"]
        if config.get("purpose") != "DEVELOPMENT" or config.get("seed") not in (
            9001,
            9002,
            9003,
        ):
            raise ValueError(
                "Tail calibration accepts independent DEV seeds 9001--9003 only"
            )
        if stage.get("stop") in (
            "initialization_failed",
            "proposal_preparation_failed",
        ):
            raise ValueError(
                "Failed initialization is not a completed tail-calibration stage"
            )
        if any(stage.get("violations", {}).values()):
            raise ValueError(
                "Calibrate from feasible DEV records, preserving failed records separately"
            )
        runtime = {
            k: v for k, v in stage.get("runtime_identity", {}).items() if k != "seed"
        }
        if not runtime:
            raise ValueError(
                "Target runtime identity is required; no CPU-to-Jetson substitution"
            )
        context = {
            "runtime": runtime,
            "base": stage["base"],
            "execution": stage["execution"],
            "features": config["features"],
            "archive_capacity": config["archive_capacity"],
            "storage": config.get("storage"),
            "journal": config.get("journal"),
        }
        if identity is not None and identity != context:
            raise ValueError(
                "Do not pool different runtimes, encoders, storage or archive configurations"
            )
        identity = context
        stage_closing = 0.0
        for row in [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]:
            if row["scope"] != "search":
                continue
            if row["reason"] in (
                "predictor_feedback",
                "stage_search_bookkeeping",
                "wall_clock_unbilled",
            ):
                value = row["cost"]["W"]
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid measured tail duration")
                if row["reason"] == "predictor_feedback":
                    feedback.append(value)
                else:
                    stage_closing += value
        closing.append(stage_closing)
        inputs.append(
            {
                "directory": str(directory),
                "stage_sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
                "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            }
        )
    if len(feedback) < 3 or not closing or max(feedback) <= 0 or max(closing) <= 0:
        raise ValueError(
            "Need at least three measured feedback events and positive closing durations"
        )
    result = {
        "schema": "cira.tail_reserve.v1",
        "status": "unreviewed_development_envelope",
        "context": identity,
        "context_sha256": digest(identity),
        "inputs": inputs,
        "safety_multiplier": safety,
        "feedback_events": len(feedback),
        "closing_events": len(closing),
        "estimates": {
            "feedback_W": max(feedback) * safety,
            "search_close_W": max(closing) * safety,
        },
        "hard_upper_bound": False,
        "review": "Validate the envelope on held-out DEV traces; actual overruns remain charged and invalidate feasibility.",
    }
    atomic_write(Path(output), canonical(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--safety", required=True, type=float)
    args = parser.parse_args()
    calibrate_tail(args.stages, args.output, args.safety)
