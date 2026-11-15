"""Matched continuation diagnostics."""

from pathlib import Path
from collections import defaultdict
import csv, gzip, json, math, statistics
import numpy as np
import run_d_only_logit_diagnostics as base

HERE = Path(__file__).resolve().parent
PATHS = [
    "configuration_then_D_logit_fit",
    "D_logit_fit_then_configuration_keep",
    "D_logit_fit_then_configuration_zero_logits",
]


def csv_write(name, rows):
    header = list(dict.fromkeys(k for r in rows for k in r))
    opener = gzip.open if name.endswith(".gz") else open
    kw = {"encoding": "utf-8", "newline": ""}
    with opener(HERE / name, "wt", **kw) as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def state_sha(local, beta, U):
    return base.array_sha(
        np.r_[
            np.asarray(local).ravel(), np.asarray(beta).ravel(), np.asarray(U).ravel()
        ]
    )


def config_delta(x, y, local, ridge, step):
    return step * np.linalg.solve(
        x.T @ x / len(y) + ridge * np.eye(len(local)), x.T @ (y - x @ local) / len(y)
    )


def main():
    old = base.OLD
    params0 = json.loads((old / "generation_parameters.json").read_bytes())
    stage = {
        k: v for k, v in np.load(old / "fixture_stage2.npz", allow_pickle=False).items()
    }
    meta = [
        r
        for r in base.read_csv(old / "fixture_inputs_index.csv")
        if r["target_stage"] == "2"
    ]
    assert len(meta) == 120
    before = {
        str(p): base.sha(p)
        for p in [
            old / "fixture_stage2.npz",
            old / "fixture_inputs_index.csv",
            old / "generation_parameters.json",
            HERE / "run_d_only_logit_diagnostics.py",
            base.REGISTRY,
        ]
    }
    parameters = dict(
        revision="D_ONLY_REVERSE_NUMERICAL_PATHS_20261107_v1",
        metric="toy_linear_MSE",
        paths=PATHS,
        source_fixture=params0["fixture_version"],
        source_files_sha256=before,
        donor_amplitude=params0["donor_amplitude"],
        configuration_ridge=params0["configuration_ridge"],
        configuration_step=params0["configuration_step"],
        delta_scope="same unmixed-reference D ridge delta on all paths; recomputed once per path and checked bitwise, not native context-conditioned configuration training",
        D_samples=64,
        batch_size=8,
        epochs=16,
        updates=128,
        optimizer=dict(
            name="numerical Adam",
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            decay=0.0,
        ),
        fitting_access=["D"],
        endpoint="fixed final step128; V/A/T reporting only",
        fixed_null_logit=0.0,
        initial_external_logits=[0.0, 0.0, 0.0],
        cleared_state="external logits reset to 0, preserving members; null and three external coefficients become 1/4 each",
        work_counts_per_path={
            "D_ridge_solve": 1,
            "configuration_vector_add": 1,
            "Adam_updates": 128,
            "full_D_epochs": 16,
            "batches": 128,
            "D_fitting_rows_consumed": 1024,
            "final_logit_vector_write": 1,
        },
        scope="explicit additive parameter edit on a simplified three-external-column linear residual fixture; not native NAS, original controller or physical FLOP evidence",
        original_main_AP_modified=False,
        prior_V_only_paths_replaced=False,
    )
    frozen = HERE / "d_reverse_parameters.json"
    if frozen.exists() and json.loads(frozen.read_bytes()) != parameters:
        raise RuntimeError("conflicting frozen reverse-path parameters")
    frozen.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    a = float(params0["donor_amplitude"])
    ridge = float(params0["configuration_ridge"])
    cstep = float(params0["configuration_step"])
    inputs = {
        k: stage[k].copy()
        for k in [
            "w_local",
            "U",
            "delta",
            "X_D",
            "y_D",
            "X_V",
            "y_V",
            "X_A",
            "y_A",
            "X_T",
            "y_T",
            "snapshot_id",
            "run_id",
            "client",
            "seed",
        ]
    }
    permutations = np.stack(
        [
            np.stack(
                [
                    base.rng([str(s), epoch, "D-only-reverse-batch-order"]).permutation(
                        64
                    )
                    for epoch in range(1, 17)
                ]
            )
            for s in inputs["snapshot_id"]
        ]
    )
    inputs["D_epoch_permutations"] = permutations
    np.savez_compressed(HERE / "d_reverse_inputs.npz", **inputs)
    input_sha = base.sha(HERE / "d_reverse_inputs.npz")
    endpoints = []
    steps = []
    epochs = []
    pairs = []
    indexes = []
    work_audit = []
    access = set()
    for item in meta:
        i = int(item["array_index"])
        local = inputs["w_local"][i]
        U = inputs["U"][i]
        X = inputs["X_D"][i]
        y = inputs["y_D"][i]
        before_local = base.array_sha(local)
        before_U = base.array_sha(U)
        shared = {
            k: item[k]
            for k in ["snapshot_id", "run_id", "dataset", "profile", "seed", "client"]
        }
        initial = state_sha(local, np.zeros(3), U)
        results = {}
        lasts = {}
        range_start = len(steps)
        for path in PATHS:
            step_start = len(steps)
            zero = np.zeros(3)
            if path == "configuration_then_D_logit_fit":
                delta = config_delta(X, y, local, ridge, cstep)
                edited = local + delta
                fit_local = edited
                cfg_parent = initial
                cfg_child = state_sha(edited, zero, U)
                beta_parent = cfg_child
            else:
                fit_local = local
            beta = np.zeros(3)
            m = np.zeros(3)
            v = np.zeros(3)
            nonzero_updates = 0
            t = 0
            for epoch in range(1, 17):
                order = permutations[i, epoch - 1]
                for offset in range(0, 64, 8):
                    selected = order[offset : offset + 8]
                    old_beta = beta.copy()
                    loss, gradient, alpha = base.loss_gradient(
                        old_beta, X[selected], y[selected], fit_local, U, a
                    )
                    access.add("D")
                    t += 1
                    m = 0.9 * m + 0.1 * gradient
                    v = 0.999 * v + 0.001 * gradient**2
                    change = (
                        0.01 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
                    )
                    beta = old_beta - change
                    nonzero_updates += int(np.any(beta != old_beta))
                    steps.append(
                        dict(
                            shared,
                            path=path,
                            epoch=epoch,
                            step=t,
                            D_rows=base.canonical(selected.tolist()),
                            fitting_role="D",
                            batch_MSE_before=loss,
                            **{
                                f"logit_before_{j + 1}": float(x)
                                for j, x in enumerate(old_beta)
                            },
                            **{
                                f"gradient_{j + 1}": float(x)
                                for j, x in enumerate(gradient)
                            },
                            **{f"moment_{j + 1}": float(x) for j, x in enumerate(m)},
                            **{
                                f"second_moment_{j + 1}": float(x)
                                for j, x in enumerate(v)
                            },
                            **{
                                f"logit_after_{j + 1}": float(x)
                                for j, x in enumerate(beta)
                            },
                        )
                    )
                loss, _, alpha = base.loss_gradient(beta, X, y, fit_local, U, a)
                epochs.append(
                    dict(
                        shared,
                        path=path,
                        epoch=epoch,
                        cumulative_steps=t,
                        full_D_MSE=loss,
                        readout_role="D",
                        **{f"logit_{j + 1}": float(x) for j, x in enumerate(beta)},
                        **{f"alpha_{j}": float(x) for j, x in enumerate(alpha)},
                    )
                )
            assert t == 128 and nonzero_updates == 128
            fitted = beta.copy()
            if path != "configuration_then_D_logit_fit":
                beta_parent = initial
                beta_child = state_sha(local, fitted, U)
                delta = config_delta(X, y, local, ridge, cstep)
                edited = local + delta
                cfg_parent = beta_child
                cfg_child = state_sha(edited, fitted, U)
            else:
                beta_child = state_sha(edited, fitted, U)
            assert np.array_equal(delta, inputs["delta"][i])
            final_beta = zero.copy() if path.endswith("_zero_logits") else fitted.copy()
            final_state = state_sha(edited, final_beta, U)
            result = {}
            alpha = base.weights(final_beta)
            for role in ("D", "V", "A", "T"):
                xr = inputs["X_" + role][i]
                yr = inputs["y_" + role][i]
                residual = a * xr @ U @ alpha[1:]
                error = xr @ edited + residual - yr
                result["MSE_" + role] = float(error @ error / len(yr))
                result["external_" + role + "_RMS"] = float(
                    np.sqrt(np.mean(residual**2))
                )
            results[path] = result
            lasts[path] = fitted
            endpoints.append(
                dict(
                    shared,
                    path=path,
                    initial_parent_sha256=initial,
                    configuration_parent_sha256=cfg_parent,
                    configuration_child_sha256=cfg_child,
                    logit_fit_parent_sha256=beta_parent,
                    logit_fit_child_sha256=beta_child,
                    terminal_state_sha256=final_state,
                    source_local_sha256=before_local,
                    final_local_sha256=base.array_sha(edited),
                    member_matrix_sha256=before_U,
                    configuration_delta_sha256=base.array_sha(delta),
                    **{f"fitted_logit_{j + 1}": float(x) for j, x in enumerate(fitted)},
                    **{
                        f"final_logit_{j + 1}": float(x)
                        for j, x in enumerate(final_beta)
                    },
                    **{f"alpha_{j}": float(x) for j, x in enumerate(alpha)},
                    D_ridge_solve_count=1,
                    configuration_add_count=1,
                    executed_Adam_update_count=t,
                    positive_update_count=nonzero_updates,
                    full_D_epochs=16,
                    batch_size=8,
                    D_rows_consumed=1024,
                    final_logit_vector_write_count=1,
                    final_logit_write_mode="zero"
                    if path.endswith("_zero_logits")
                    else "keep",
                    local_dimension=8,
                    logit_dimension=3,
                    work_unit="counted numerical operations, not physical FLOP",
                    step_log="d_reverse_step_log.csv.gz",
                    step_row_start=step_start,
                    step_row_end=len(steps) - 1,
                    **result,
                )
            )
            assert (
                base.array_sha(local) == before_local and base.array_sha(U) == before_U
            )
        assert np.array_equal(lasts[PATHS[1]], lasts[PATHS[2]])
        for first, second in [
            (PATHS[0], PATHS[1]),
            (PATHS[1], PATHS[2]),
            (PATHS[0], PATHS[2]),
        ]:
            pairs.append(
                dict(
                    shared,
                    contrast=first + "_minus_" + second,
                    gain_definition="MSE(second)-MSE(first), positive favors first",
                    **{
                        "gain_MSE_" + role: results[second]["MSE_" + role]
                        - results[first]["MSE_" + role]
                        for role in ("D", "V", "A", "T")
                    },
                )
            )
        work_audit.append(
            dict(
                shared,
                paths=3,
                all_positive_update_count_128=True,
                all_D_ridge_solve_count_1=True,
                all_configuration_add_count_1=True,
                same_configuration_delta=True,
                same_D_batch_order=True,
                keep_and_zero_paths_have_identical_fitted_logits=True,
                local_and_members_unchanged=True,
                frozen_endpoint_step=128,
                V_A_T_used_for_fitting_or_selection=False,
            )
        )
        indexes.append(
            dict(
                shared,
                input_file="d_reverse_inputs.npz",
                input_file_sha256=input_sha,
                array_index=i,
                source_snapshot_sha256=item["snapshot_sha256"],
                source_local_sha256=before_local,
                member_matrix_sha256=before_U,
                step_log="d_reverse_step_log.csv.gz",
                step_row_start=range_start,
                step_row_end=len(steps) - 1,
                step_rows=384,
            )
        )
    assert access == {"D"}
    grouped = defaultdict(list)
    for row in pairs:
        grouped[
            row["dataset"], row["profile"], row["seed"], row["run_id"], row["contrast"]
        ].append(row)
    seed_pairs = []
    for key, rows in sorted(grouped.items()):
        assert len(rows) == 4
        seed_pairs.append(
            dict(
                dataset=key[0],
                profile=key[1],
                seed=key[2],
                run_id=key[3],
                contrast=key[4],
                clients=4,
                **{
                    "gain_MSE_" + role: statistics.mean(
                        r["gain_MSE_" + role] for r in rows
                    )
                    for role in ("D", "V", "A", "T")
                },
            )
        )
    summary = []
    for ds, profile, contrast in sorted(
        {(r["dataset"], r["profile"], r["contrast"]) for r in seed_pairs}
    ):
        block = [
            r
            for r in seed_pairs
            if (r["dataset"], r["profile"], r["contrast"]) == (ds, profile, contrast)
        ]
        assert len(block) == 5
        for role in ("V", "A", "T"):
            values = [r["gain_MSE_" + role] for r in block]
            mu = statistics.mean(values)
            sd = statistics.stdev(values)
            half = 2.7764451051977987 * sd / math.sqrt(5)
            summary.append(
                dict(
                    dataset=ds,
                    profile=profile,
                    contrast=contrast,
                    role=role,
                    n_source_seed_blocks=5,
                    mean_gain_MSE=mu,
                    sample_SD=sd,
                    CI95_lo=mu - half,
                    CI95_hi=mu + half,
                    negative_seed_blocks=sum(v < -1e-6 for v in values),
                    scope="descriptive extended fixture, not original controller or native verification",
                )
            )
    products = {
        "d_reverse_inputs_index.csv": indexes,
        "d_reverse_endpoints.csv": endpoints,
        "d_reverse_pairs.csv": pairs,
        "d_reverse_work_audit.csv": work_audit,
        "d_reverse_epoch_log.csv": epochs,
        "d_reverse_step_log.csv.gz": steps,
        "d_reverse_seed_pairs.csv": seed_pairs,
        "d_reverse_condition_summary.csv": summary,
    }
    for name, rows in products.items():
        csv_write(name, rows)
    assert (len(endpoints), len(pairs), len(steps), len(epochs)) == (
        360,
        360,
        46080,
        5760,
    )
    assert all(base.sha(Path(p)) == expected for p, expected in before.items())
    qa = dict(
        status="PASS_D_ONLY_REVERSE_PATH_ARITHMETIC",
        snapshots=120,
        paths=360,
        D_closed_form_solves=360,
        configuration_adds=360,
        numerical_Adam_updates=46080,
        full_D_epochs=5760,
        all_positive_update_count_equals128=True,
        D_only_access=sorted(access),
        identical_per_path_counted_work=True,
        local_and_donor_invariance=True,
        same_delta_and_batch_order=True,
        source_files_unchanged=True,
        main_AP_unchanged=True,
        prior_mechanism_files_unchanged=True,
        original_controller_or_native_validation=False,
    )
    (HERE / "d_reverse_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa))


if __name__ == "__main__":
    main()
