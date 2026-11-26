# Running CIRA

## Install

Use Python 3.12 in a fresh environment. Run the commands from the root of the extracted repository.

```bash
python -m pip install -e ".[plots]"
```

For detector jobs, install the YOLO dependencies as well.

```bash
python -m pip install -e ".[yolo,plots]"
```

## Run a reduced feature-model example

```bash
python studies/run.py feature --case-id CL-HYP-00001 --output runs/feature
```

The bundled case generates features and writes `runs/feature/results.json`. Use a new or empty output directory for each run.

## Run numerical scenarios

```bash
python -m softcas.paper_cli matrix --receiver-study core --output runs/receiver/core.json
python -m softcas.paper_cli matrix --receiver-study extended --output runs/receiver/extended.json
python studies/continuation/generate.py --core-matrix runs/receiver/core.json --extended-matrix runs/receiver/extended.json --output runs/continuation
python studies/time_and_revisits/generate.py --inputs runs/continuation/tables.json --output runs/time_revisit_tables.json
```

The two matrices configure the continuation scenario. Its output feeds the time-budget and BDD100K revisit scenarios.

## Recompute summaries and supplementary plots

If you have the companion-data archive, extract it so that `runs/companion/` contains `data/`, `data_catalog.csv`, and `SHA256SUMS.txt`.

```bash
python analysis/recompute.py --data-root runs/companion > runs/companion-summary.json
python analysis/plot_continuation.py --data-root runs/companion --output runs/continuation-figures
```

These commands recompute the tabulated summaries and rebuild Supplementary Figures S2 and S3 from the supplied records.

## Run detector jobs

```bash
python -m softcas.paper_cli export --output runs/configs
python -m softcas.paper_cli matrix --blocks primary BDD --output runs/matrix.json
```

Follow the [detector workflow](docs/workflow.md) to prepare local images and annotations, bind detector checkpoints and device settings, and compile the execution plan. On each assigned device, replace `JOB_ID` and `NODE_ID` with the values in the compiled plan.

```bash
python -m softcas.paper_cli run-job --plan runs/compiled/compiled.json --id JOB_ID --node-id NODE_ID
```

## Run the tests

```bash
python -m unittest discover -s tests -q
```

Install the `yolo` and `plots` extras to run the detector and plotting tests.
