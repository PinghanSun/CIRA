"""Static preparation report. Never probes GPUs or starts training/network jobs."""

import json
from pathlib import Path
from .paper_runner import bind_file
from .initialization import validate_recipe


def check_readiness(config_directory, bindings):
    root = Path(config_directory)
    checks = []

    def add(item, fn):
        try:
            fn()
            checks.append({"item": item, "ready": True})
        except (ValueError, KeyError, TypeError, OSError) as e:
            checks.append({"item": item, "ready": False, "reason": str(e)})

    for condition in ("C0", "C1", "C2"):
        source = root / (condition + "_initialization.example.json")
        if source.exists():
            recipe = bindings.get("initialization", {}).get(
                condition, json.loads(source.read_text(encoding="utf-8"))
            )
            add(
                condition + " preparation recipe",
                lambda recipe=recipe, condition=condition: validate_recipe(
                    recipe["recipe"], condition
                ),
            )
    for name, cell in sorted(bindings.get("cells", {}).items()):
        for field in (
            "checkpoint",
            "registry",
            "detector_config",
            "contact_trace",
            "development_registry",
            "development_lock",
        ):
            add(
                name + "/" + field,
                lambda cell=cell, field=field: bind_file(cell.get(field)),
            )

        def resources(cell=cell):
            override = cell.get("overrides", {})
            for scope in ("search", "release"):
                for r in ("F", "W"):
                    if not override.get("budgets", {}).get(scope, {}).get(r, 0) > 0:
                        raise ValueError("Bind calibrated positive " + scope + "/" + r)
            p = Path(override["cost_profile"])
            if not p.is_file():
                raise ValueError("Cost profile absent")

        add(name + "/native resource calibration", resources)
    if not bindings.get("cells"):
        checks.append(
            {
                "item": "native input cells",
                "ready": False,
                "reason": "No actual data/checkpoint/config cells bound",
            }
        )

    def nodes():
        physical = bindings.get("physical_nodes", {})
        if set(physical) != {"0", "1", "2", "3"}:
            raise ValueError("Four actual physical endpoints required")
        endpoints = []
        for node in physical.values():
            endpoint = node["endpoint"]
            host, port = endpoint.get("host"), endpoint.get("port")
            if not host or type(port) is not int or not 0 < port < 65536:
                raise ValueError("Bind actual endpoint host/port")
            endpoints.append((host, port))
        if len(set(endpoints)) != 4:
            raise ValueError("Physical endpoints must be distinct")

    add("four-node endpoints", nodes)
    return {
        "schema": "cira.readiness.20261114",
        "checks": checks,
        "ready": all(x["ready"] for x in checks),
        "training_started": False,
        "device_or_network_probe_performed": False,
        "interpretation": "Use the checks to bind local data, development results and calibrated device settings.",
    }
