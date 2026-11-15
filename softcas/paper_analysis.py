"""Action summaries and prediction references from actual ordered feedback."""

from collections import defaultdict, Counter
import math
import numpy as np
from .reporting import retention, paired_effects


def prediction_references(rows, unit):
    if unit not in ("fractional_AP_gain", "MSE_derived_score", "recorded_gain"):
        raise ValueError("Declare gain units explicitly")
    grouped = defaultdict(list)
    for row in rows:
        if row.get("executed", True) is not True:
            continue
        key = tuple(row[k] for k in ("dataset", "detector", "seed", "client", "stage"))
        grouped[key].append(row)
    result = []
    per_seed = defaultdict(lambda: defaultdict(list))
    for key, items in sorted(grouped.items()):
        attempts = [r["attempt"] for r in items]
        if len(set(attempts)) != len(attempts):
            raise ValueError("Duplicate feedback attempt in one context")
        history = []
        for r in sorted(items, key=lambda x: x["attempt"]):
            gain = float(r["gain"])
            pred = float(r["prediction"])
            if not math.isfinite(gain) or not math.isfinite(pred):
                raise ValueError("Missing/nonfinite feedback is not zero imputed")
            previous = float(np.mean(history)) if history else 0.0
            values = {"predictor": pred, "zero": 0.0, "history": previous}
            errors = {k: abs(v - gain) for k, v in values.items()}
            result.append(
                {
                    **{
                        k: r[k]
                        for k in (
                            "dataset",
                            "detector",
                            "seed",
                            "client",
                            "stage",
                            "attempt",
                        )
                    },
                    "gain": gain,
                    "predictions": values,
                    "absolute_error": errors,
                }
            )
            for name, value in errors.items():
                per_seed[key[:3]][name].append(value)
            history.append(gain)
    summaries = [
        {
            "dataset": d,
            "detector": m,
            "seed": s,
            "MAE": {k: float(np.mean(v)) for k, v in values.items()},
            "observations": len(values["predictor"]),
        }
        for (d, m, s), values in sorted(per_seed.items())
    ]
    return {
        "unit": unit,
        "observations": result,
        "seed_summaries": summaries,
        "history_scope": "earlier executed feedback in the same run/client/stage; initial mean zero",
    }


def action_summary(rows):
    mapping = {
        "refine": "Refine",
        "fresh": "Fresh",
        "add": "Collaborate",
        "replace": "Collaborate",
        "remove": "Remove",
        "import": "Import",
    }
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "detector", "seed"))
        if row["kind"] not in mapping:
            raise ValueError("Unknown recorded action kind")
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items()):
        counts = Counter(mapping[r["kind"]] for r in items)
        retained = [r["retained"] for r in items if r.get("retained") is not None]
        if any(type(v) is not bool for v in retained):
            raise ValueError("Retained state must be an actual boolean")
        out.append(
            {
                "dataset": key[0],
                "detector": key[1],
                "seed": key[2],
                "actions": len(items),
                "share_percent": {
                    k: 100 * counts[k] / len(items) for k in mapping.values()
                },
                "retained_percent": 100 * sum(retained) / len(retained)
                if retained
                else None,
                "retention_observations": len(retained),
            }
        )
    return out
