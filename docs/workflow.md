# Workflow

## Data

```bash
python -m softcas.paper_cli import-allocation --spec configs/allocation.private.json
python -m softcas.paper_cli registry --spec configs/registry.private.json
python -m softcas.paper_cli convert-bdd --spec configs/bdd.private.json
python -m softcas.paper_cli single-exposure --spec configs/exposure.private.json
```

## Initialization and development

```bash
python -m softcas.paper_cli initialize --spec configs/initialization.private.json
python -m softcas.paper_cli matrix --blocks dev BDD_dev --output runs/dev.json
python -m softcas.paper_cli dev-lock --spec configs/dev.private.json --output runs/dev-lock.json
python -m softcas.paper_cli freeze-matrix --matrix runs/matrix.json --lock runs/dev-lock.json --output runs/locked.json
```

## Diagnostics and analysis

```bash
python -m softcas.production mechanism --spec configs/mechanism.private.json
python -m softcas.paper_cli analyze-actions --input runs/actions.json --unit fractional_AP_gain --output runs/prediction.json
python analysis/render_results.py --spec configs/analysis.private.json --output runs/analysis
```

Runtime configurations use D for fitting, V for selection, A for release audit, and T for reporting. Feature-model and numerical studies use their own runners and metrics.
