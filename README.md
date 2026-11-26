# Run CIRA

Use Python 3.12. Open a terminal in the repository root and install the package.

```sh
python -m pip install -e .
```

## Run an example

```sh
python studies/run.py feature --case-id CL-HYP-00001 --output runs/feature
```

This example generates its feature inputs and saves the results in `runs/feature/results.json`. Use an empty output directory on subsequent runs.

## Run detector jobs

```sh
python -m pip install -e ".[yolo]"
python -m softcas.paper_cli export --output runs/configs
python -m softcas.paper_cli matrix --blocks primary BDD --output runs/matrix.json
```

Follow the [detector workflow](docs/workflow.md) to supply your local datasets, checkpoints, and preprocessed assets and compile the execution plan. Replace `JOB_ID` and `NODE_ID` with the values in that plan.

```sh
python -m softcas.paper_cli run-job --plan runs/compiled/compiled.json --id JOB_ID --node-id NODE_ID
```

For numerical runs, follow the [scenario commands](docs/experiments.md).

## Run the tests

```sh
python -m pip install -e ".[yolo,plots]"
python -m unittest discover -s tests -q
```
