# CIRA

Composition-Informed Receiver Adaptation.

PyTorch code for receiver-side adapter composition, budgeted search, and continual object detection.

## Installation

```bash
python -m pip install -e ".[yolo,plots]"
```

## Usage

```bash
python -m softcas.paper_cli export --output runs/configs
python -m softcas.paper_cli matrix --output runs/matrix.json
python -m softcas.paper_cli compile --matrix runs/matrix.json --bindings configs/bindings.private.json --output runs/compiled
python -m softcas.paper_cli run-job --plan runs/compiled/compiled.json --id JOB_ID --node-id 0
```

Use `configs/paper/` as the experiment template. Store local paths and device settings in a `*.private.json` file.

## Supplementary studies

```bash
python studies/run.py list
python studies/run.py feature --case-id CL-HYP-00001 --output runs/feature
python studies/run.py fixed-summary --output runs/summary
python studies/run.py continuation --output runs/continuation
```

## Tests

```bash
python -m unittest discover -s tests -q
```

## Structure

```text
softcas/      Method, runtime, and baseline implementations
configs/      Experiment and device templates
studies/      Feature-model and numerical studies
analysis/     Plotting and table export
examples/     Small runnable examples
tests/        Unit and integration tests
```

See [the workflow](docs/workflow.md) for data preparation and development commands.
