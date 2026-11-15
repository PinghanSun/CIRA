"""Endpoint plots and tables."""

import argparse, csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from softcas.paper_runner import bind_file
from softcas.reporting import paired_effects


def render(spec, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    identities = set()
    inputs = []
    failed = []
    scope = spec["evidence_scope"]
    if scope not in (
        "native_detector",
        "independent_feature",
        "numerical_record",
        "software_fixture",
    ):
        raise ValueError("Declare result evidence scope")
    for binding in spec["trajectory_files"]:
        path = Path(bind_file(binding))
        r = json.loads(path.read_text(encoding="utf-8-sig"))
        inputs.append(binding)
        if r.get("complete") is not True:
            failed.append({"path": str(path), "status": r.get("status", "incomplete")})
            continue
        if scope != "software_fixture" and r.get("study_kind") != scope:
            raise ValueError("Do not mix experiment systems")
        recipe = r.get("paper_recipe", r)
        cell = r.get("cell") or recipe["dataset"] + "/" + recipe["detector"]
        key = (cell, r["method"], r["seed"])
        if key in identities:
            raise ValueError("Reused endpoint cannot increase seed count")
        identities.add(key)
        score = r["seen"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("This endpoint interface uses fractional SeenAP")
        rows.append(
            {"cell": cell, "method": r["method"], "seed": r["seed"], "seen": score}
        )
    if not rows:
        raise ValueError("No complete endpoints")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve existing analysis")
    out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["cell"], row["method"]].append(row["seen"])
    summary = []
    for (cell, method), values in sorted(groups.items()):
        summary.append(
            {
                "cell": cell,
                "method": method,
                "n_seed": len(values),
                "SeenAP_percent": 100 * float(np.mean(values)),
                "SD_pp": 100 * float(np.std(values, ddof=1))
                if len(values) > 1
                else None,
            }
        )
    with (out / "quality.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    escape = (
        lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    )
    tex = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Cell & Method & $n$ & SeenAP (\%) & SD (pp) \\",
        r"\hline",
    ]
    for r in summary:
        sd = "NA" if r["SD_pp"] is None else f"{r['SD_pp']:.2f}"
        tex.append(
            f"{escape(r['cell'])} & {escape(r['method'])} & {r['n_seed']} & {r['SeenAP_percent']:.2f} & {sd}"
            + r" \\"
        )
    tex.extend([r"\hline", r"\end{tabular}"])
    (out / "quality.tex").write_text("\n".join(tex), encoding="utf-8")
    contrasts = []
    reference = spec.get("reference", "softcas")
    comparators = spec.get("comparators", [])
    cells = sorted({r["cell"] for r in rows})
    full = {(r["cell"], r["seed"]): r["seen"] for r in rows if r["method"] == reference}
    declared = [(c, m) for c in cells for m in comparators]
    left = {
        (str((c, m)), s): full.get((c, s)) for c, m in declared for s in spec["seeds"]
    }
    right = {
        (str((r["cell"], r["method"])), r["seed"]): r["seen"]
        for r in rows
        if r["method"] in comparators
    }
    if declared:
        contrasts = paired_effects(
            left, right, [str(x) for x in declared], tuple(spec["seeds"])
        )
    with plt.rc_context(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            len(cells), 1, figsize=(7, max(2.4, 2.1 * len(cells))), squeeze=False
        )
        for ax, cell in zip(axes[:, 0], cells):
            subset = [r for r in summary if r["cell"] == cell]
            for i, r in enumerate(subset):
                ax.errorbar(
                    i,
                    r["SeenAP_percent"],
                    yerr=r["SD_pp"],
                    fmt="o",
                    capsize=3,
                    color="#B52A27" if r["method"] == reference else "#36758E",
                )
            ax.set_xticks(
                range(len(subset)),
                [r["method"] for r in subset],
                rotation=25,
                ha="right",
            )
            ax.set_ylabel("SeenAP (%)")
            ax.set_title(cell, loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "quality.pdf")
        fig.savefig(out / "quality.svg")
        plt.close(fig)
    result = {
        "evidence_scope": scope,
        "inputs": inputs,
        "failed_or_incomplete": failed,
        "summary": summary,
        "paired_contrasts": contrasts,
        "seed_unit": "complete federation trajectory",
        "uncertainty_in_figure": "sample SD across seeds; single-seed SD unavailable",
        "new_training_started": False,
    }
    (out / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    r = render(json.loads(Path(a.spec).read_text(encoding="utf-8-sig")), a.output)
    print(json.dumps({"groups": len(r["summary"]), "output": a.output}))
