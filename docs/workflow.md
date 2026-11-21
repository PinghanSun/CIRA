# Detector workflow

The CPU example in the [README](../README.md) runs with generated inputs. The detector workflow uses local image and annotation files, detector checkpoints, resource calibration and node settings.

## Prepare the data and initialization

Export the templates with `python -m softcas.paper_cli export --output runs/configs`. Copy the applicable templates to local `*.private.json` files and fill the paths and output fields. Use the same class order as [the detector protocol](../configs/paper/protocol.json).

| Task | Template | Command |
| --- | --- | --- |
| Import a stage allocation | [Allocation](../configs/paper/allocation_import.example.json) | `python -m softcas.paper_cli import-allocation --spec runs/allocation.private.json` |
| Create an image registry | [Registry](../configs/paper/registry.example.json) | `python -m softcas.paper_cli registry --spec runs/registry.private.json` |
| Convert BDD100K annotations | [BDD100K conversion](../configs/paper/BDD100K_conversion.example.json) | `python -m softcas.paper_cli convert-bdd --spec runs/bdd.private.json` |
| Build the single-exposure allocation | [Exposure](../configs/paper/single_exposure.example.json) | `python -m softcas.paper_cli single-exposure --spec runs/exposure.private.json` |
| Prepare a base detector | [C0](../configs/paper/C0_initialization.example.json), [C1](../configs/paper/C1_initialization.example.json), [C2](../configs/paper/C2_initialization.example.json) | `python -m softcas.paper_cli initialize --spec runs/initialization.private.json` |

The data roles are D for fitting, V for candidate selection, A for release audit and T for reporting. Peer exchange uses versioned adapter packages.

## Select and bind a configuration

Generate the development matrix and bind its applicable cells using the [bindings template](../configs/paper/bindings.example.json). Bind each checkpoint, registry, detector configuration and contact trace by path and SHA-256. Provide the resource profile and limits in the cell overrides. For a physical run, provide four node endpoints and a common start time.

```bash
python -m softcas.paper_cli matrix --blocks dev BDD_dev --output runs/dev.json
python -m softcas.paper_cli compile --matrix runs/dev.json --bindings runs/bindings.private.json --output runs/dev-plan
```

Use the development results to populate the [development lock](../configs/paper/dev_lock.example.json). Apply the lock to the experiment matrix, then bind the locked cells to create the execution plan.

```bash
python -m softcas.paper_cli dev-lock --spec runs/dev-lock.private.json --output runs/dev-lock.json
python -m softcas.paper_cli freeze-matrix --matrix runs/matrix.json --lock runs/dev-lock.json --output runs/locked.json
python -m softcas.paper_cli preflight --bindings runs/bindings.private.json --configs runs/configs --output runs/preflight.json
python -m softcas.paper_cli compile --matrix runs/locked.json --bindings runs/bindings.private.json --output runs/compiled
```

## Run a bound job

Choose the job ID and node ID from the `jobs` list in `runs/compiled/compiled.json`. Run each node's entry on its assigned device.

```bash
python -m softcas.paper_cli run-job --plan runs/compiled/compiled.json --id JOB_ID --node-id NODE_ID
```

Use `python -m softcas.production --help` for runtime, device and reporting commands. The [result renderer](../analysis/render_results.py) reads bound trajectory files. The [reduced feature model](../studies/run.py) has a separate entry point.
