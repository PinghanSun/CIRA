"""Select stage epoch/attempt settings from complete, bound DEV trajectories."""

from pathlib import Path
import hashlib, json, math
from collections import defaultdict
from .state import digest


def lock_stage_budget(spec):
    from .paper_runner import bind_file

    matrix_path = bind_file(spec["matrix"])
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8-sig"))
    jobs = {
        r["id"]: r["recipe"]
        for r in matrix["jobs"]
        if r["recipe"]["phase"] == "development"
    }
    cells = set(
        spec.get("cells") or [r["dataset"] + "/" + r["detector"] for r in jobs.values()]
    )
    expected = {
        k: v for k, v in jobs.items() if v["dataset"] + "/" + v["detector"] in cells
    }
    if not expected:
        raise ValueError("No registered DEV candidates for selected cells")
    records = {}
    inputs = []
    for binding in spec["records"]:
        path = bind_file(binding)
        row = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        key = row.get("paper_job_id")
        recipe = expected.get(key)
        if recipe is None or row.get("paper_recipe") != recipe or key in records:
            raise ValueError("Unregistered, changed or duplicate DEV trajectory")
        if (
            row.get("role") != "DEV"
            or row.get("complete") is not True
            or row.get("metric") != "AP50:95"
            or row.get("selection_metric") != "final_macro_seen"
            or set(map(str, row.get("clients", []))) != {"0", "1", "2", "3"}
            or row.get("stages") != 4
        ):
            raise ValueError(
                "Complete four-client/four-stage DEV federation endpoint required"
            )
        for name in ("seen", "F_total", "network_tx_bytes"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("DEV selection requires actual finite " + name)
        if row["seen"] > 1:
            raise ValueError("DEV SeenAP uses fractions")
        if row.get("study_kind") != "native_detector":
            raise ValueError(
                "Feature or supplied numerical results cannot lock native DEV"
            )
        records[key] = row
        inputs.append(
            {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
        )
    if set(records) != set(expected):
        raise ValueError(
            "Missing or failed DEV candidates; do not choose from survivors"
        )
    grouped = defaultdict(list)
    for key, recipe in expected.items():
        grouped[
            (
                recipe["dataset"] + "/" + recipe["detector"],
                recipe["stage_epochs"],
                recipe["attempts"],
            )
        ].append(records[key])
    summaries = []
    by_cell = defaultdict(list)
    for (cell, epochs, attempts), rows in sorted(grouped.items()):
        mean = lambda key: sum(r[key] for r in rows) / len(rows)
        item = {
            "cell": cell,
            "stage_epochs": epochs,
            "attempts": attempts,
            "mean_seen": mean("seen"),
            "mean_F": mean("F_total"),
            "mean_B": mean("network_tx_bytes"),
            "trajectories": len(rows),
        }
        summaries.append(item)
        by_cell[cell].append(item)
    if set(by_cell) != cells:
        raise ValueError("Missing requested DEV cell")
    winners = {
        c: min(
            rows,
            key=lambda r: (
                -r["mean_seen"],
                r["mean_F"],
                r["mean_B"],
                r["stage_epochs"],
                r["attempts"],
            ),
        )
        for c, rows in by_cell.items()
    }
    result = {
        "schema": "cira.native_budget_lock.20261114",
        "role": "DEV",
        "locked": True,
        "cells": winners,
        "matrix_sha256": hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest(),
        "inputs": inputs,
        "candidate_summaries": summaries,
        "seeds": [9001, 9002, 9003],
        "local_epochs": 8,
        "receiving_epochs": 4,
        "selection_metric": "final_macro_seen",
        "tie_rule": "higher macro SeenAP, lower actual F, lower network Tx, fixed epoch/attempt ID",
    }
    result["evidence_sha256"] = digest(
        {"matrix": spec["matrix"], "inputs": inputs, "cells": winners}
    )
    return result


def selected_budget(lock, cell):
    if (
        lock.get("schema") != "cira.native_budget_lock.20261114"
        or lock.get("role") != "DEV"
        or lock.get("locked") is not True
    ):
        raise ValueError("Bound native stage budget lock required")
    result = lock.get("cells", {}).get(cell)
    if result is None:
        raise ValueError("DEV lock lacks cell " + cell)
    if (
        result.get("stage_epochs") not in (50, 100, 200)
        or result.get("attempts") not in (8, 16, 32)
        or lock.get("local_epochs") != 8
        or lock.get("receiving_epochs") != 4
    ):
        raise ValueError("DEV lock uses a different fitting protocol")
    return result


def freeze_matrix(matrix, lock):
    """New IDs reflect actual DEV winners; archived round protocols stay intact."""
    import copy

    jobs = {}
    old_to_new = {}
    for job in matrix["jobs"]:
        recipe = copy.deepcopy(job["recipe"])
        if recipe["phase"] == "formal":
            selected = selected_budget(
                lock, recipe["dataset"] + "/" + recipe["detector"]
            )
            recipe.update(
                stage_epochs=selected["stage_epochs"], attempts=selected["attempts"]
            )
        key = digest(recipe)
        old_to_new[job["id"]] = key
        jobs.setdefault(
            key, {**job, "id": key, "recipe": recipe, "status": "not_executed"}
        )
    result = {
        **matrix,
        "jobs": list(jobs.values()),
        "unique_jobs": len(jobs),
        "views": [{**v, "job_id": old_to_new[v["job_id"]]} for v in matrix["views"]],
        "development_evidence_sha256": lock["evidence_sha256"],
        "execution_started": False,
    }
    return result
