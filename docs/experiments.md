# Experiment reproduction

Install the project from the repository root with `python -m pip install -e ".[yolo,plots]"`. The commands below use separate directories for generated files.

## Receiver study on ACDC and ControlledFog

Generate the native YOLOv8n experiment matrices. The pilot uses the development seed 9001. The core matrix pairs Full, Strict No-cross and Serial within each of eight seeds per dataset. The extended matrix adds Strict No-gain and DeModCL.

```bash
python -m softcas.paper_cli matrix --receiver-study pilot --output runs/receiver/pilot.json
python -m softcas.paper_cli matrix --receiver-study core --output runs/receiver/core.json
python -m softcas.paper_cli matrix --receiver-study extended --output runs/receiver/extended.json
```

Apply the development lock and bind the registered detector inputs as described in the [detector workflow](workflow.md). Each native trajectory uses local data, a checkpoint, a contact trace, and resource and device settings. The execution plan lists the jobs that have valid bindings.

The action records include `parent_is_stage_entry`, `parent_from_stage_revision` and `child_retained`. The last field refers to archive membership immediately after that attempt. Read `stage.json` for the deployed state after the release audit.

## Regenerate the scenarios

The continuation scenario uses the default eight seeds and the core and extended matrices above. Its comparison includes preset endpoint offsets and simulated action records.

```bash
python studies/continuation/generate.py --core-matrix runs/receiver/core.json --extended-matrix runs/receiver/extended.json --output runs/continuation
```

The time-budget model and BDD100K revisit extension use the stage-domain reference rows, revisit scores and four-cell contrasts generated in the preceding step.

```bash
python studies/time_and_revisits/generate.py --inputs runs/continuation/tables.json --output runs/time_revisit_tables.json
```

Run the [detector workflow](workflow.md) with separately bound hardware and checkpoints to collect device measurements.
