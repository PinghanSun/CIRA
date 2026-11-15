"""D-only mixing diagnostics."""

from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, math, statistics
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path("data/mechanisms")
REGISTRY = Path("data/results.json")
METHOD = Path("data/04_softcas_method.tex")
P = {
    "revision": "D_ONLY_LOGIT_ARITHMETIC_20261107_v1",
    "metric": "linear_fixture_MSE",
    "source_fixture": "EXTENDED_LINEAR_RESIDUAL_FIXTURE_20261107_v1",
    "scope": "three-column simplified residual block, not native detector ports or original controller verification",
    "initial_external_logits": [0.0, 0.0, 0.0],
    "fixed_null_logit": 0.0,
    "epochs": 16,
    "batch_size": 8,
    "steps_per_epoch": 8,
    "total_steps_per_snapshot": 128,
    "optimizer": "numerical Adam",
    "learning_rate": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "weight_decay": 0.0,
    "loss": "mean((X_D@w_local + a*X_D@U@softmax([0,beta])[1:] - y_D)^2)",
    "fitting_access": ["D"],
    "endpoint_rule": "fixed final update 128; no V/A/T checkpoint selection",
    "role_sizes": {"D": 64, "V": 128, "A": 128, "T": 256},
    "finite_difference_step": 1e-6,
    "finite_difference_absolute_tolerance": 1e-7,
    "method_reference": {
        "weights": "Eq weights: alpha_e=exp(beta_e)/(1+sum(exp(beta))); alpha_null=1/(1+sum(exp(beta)))",
        "fit_split": "Fresh/Refine optimize phi,beta on D; Collaborate freezes phi and fits beta on D; V scores candidates; A audits a locked candidate; T reports",
        "native_cap_note": "native at most two selected paths per port; this inherited numerical block has three external columns and remains a simplified fixture",
        "normalization_note": "native softmax includes a null component and intentionally permits external amplitude changes; no additional fixed-RMS response normalization is inserted",
    },
    "prior_fixture_retained": True,
    "main_AP_modified": False,
    "randomization": "SHA256(snapshot_id, epoch, D-only-batch-order) PCG64; no outcome-dependent resampling",
}


def canonical(obj):
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda: s.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(arr):
    arr = np.ascontiguousarray(arr, dtype="<f8")
    return hashlib.sha256(
        canonical({"shape": arr.shape, "dtype": "<f8"}).encode() + arr.tobytes()
    ).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    header = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def rng(key):
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(canonical(key).encode()).digest()[:8], "little")
    )


def weights(beta):
    z = np.r_[0.0, np.asarray(beta)]
    e = np.exp(z - z.max())
    return e / e.sum()


def loss_gradient(beta, x, y, local, U, a):
    alpha = weights(beta)
    Z = a * x @ U
    r = Z @ alpha[1:]
    error = x @ local + r - y
    loss = float(np.dot(error, error) / len(y))
    gradient = (2 / len(y)) * alpha[1:] * (Z.T @ error - float(np.dot(r, error)))
    return loss, gradient, alpha


def report(beta, data, i, a):
    alpha = weights(beta)
    local = data["w_local"][i]
    U = data["U"][i]
    result = {}
    for role in ("D", "V", "A", "T"):
        x = data["X_" + role][i]
        y = data["y_" + role][i]
        r = a * x @ U @ alpha[1:]
        error = x @ local + r - y
        result["MSE_" + role] = float(np.dot(error, error) / len(y))
        result["external_" + role + "_RMS"] = float(np.sqrt(np.mean(r * r)))
    result["effective_parameter_residual_L2"] = float(np.linalg.norm(a * U @ alpha[1:]))
    result.update({f"logit_{j + 1}": float(v) for j, v in enumerate(beta)})
    result.update({f"alpha_{j}": float(v) for j, v in enumerate(alpha)})
    return result


def factorial_audit(a):
    data = {
        k: v for k, v in np.load(OLD / "fixture_stage2.npz", allow_pickle=False).items()
    }
    index = [
        r
        for r in read_csv(OLD / "fixture_inputs_index.csv")
        if r["target_stage"] == "2"
    ]
    cells = read_csv(OLD / "stage2_factorial_cells.csv")
    by = defaultdict(list)
    for row in cells:
        if row["mix_on"] == "1":
            by[row["snapshot_id"]].append(row)
    rows = []
    for meta in index:
        i = int(meta["array_index"])
        pair = by[meta["snapshot_id"]]
        assert len(pair) == 2
        pair = sorted(pair, key=lambda r: int(r["config_on"]))
        for key in ("donor_sha256", "beta", "donor_amplitude"):
            assert pair[0][key] == pair[1][key]
        beta0 = np.array(json.loads(pair[0]["beta"]))
        beta1 = np.array(json.loads(pair[1]["beta"]))
        r0 = a * data["X_D"][i] @ data["U"][i] @ beta0
        r1 = a * data["X_D"][i] @ data["U"][i] @ beta1
        rms0 = float(np.sqrt(np.mean(r0 * r0)))
        rms1 = float(np.sqrt(np.mean(r1 * r1)))
        rows.append(
            dict(
                snapshot_id=meta["snapshot_id"],
                run_id=meta["run_id"],
                dataset=meta["dataset"],
                profile=meta["profile"],
                seed=meta["seed"],
                client=meta["client"],
                config0_D_RMS=rms0,
                config1_D_RMS=rms1,
                D_RMS_absolute_difference=abs(rms0 - rms1),
                maximum_elementwise_D_difference=float(np.max(np.abs(r0 - r1))),
                exact_elementwise_equal=bool(np.array_equal(r0, r1)),
                same_D_array=True,
                same_donor_matrix=True,
                same_fixed_coefficients=True,
                X_D_sha256=array_sha(data["X_D"][i]),
                U_sha256=array_sha(data["U"][i]),
                config0_external_D_sha256=array_sha(r0),
                config1_external_D_sha256=array_sha(r1),
                amplitude_scope="within-snapshot mixed configuration pair; between-snapshot RMS need not be equal",
                finding="existing extended fixture already controls effective donor response; no replacement cells generated",
            )
        )
    assert len(rows) == 120 and all(r["exact_elementwise_equal"] for r in rows)
    write_csv("factorial_effective_amplitude_audit.csv", rows)
    return rows


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    source_paths = [
        OLD / "fixture_stage2.npz",
        OLD / "fixture_stage4.npz",
        OLD / "fixture_inputs_index.csv",
        OLD / "stage2_factorial_cells.csv",
        OLD / "generation_parameters.json",
        REGISTRY,
        METHOD,
    ]
    before = {str(p): sha(p) for p in source_paths}
    prior = json.loads((OLD / "generation_parameters.json").read_bytes())
    a = float(prior["donor_amplitude"])
    params = dict(P, donor_amplitude=a, source_files_sha256=before)
    config_path = HERE / "d_only_parameters.json"
    if config_path.exists() and json.loads(config_path.read_bytes()) != params:
        raise RuntimeError("conflicting frozen D-only parameters")
    config_path.write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = factorial_audit(a)
    source = {
        k: v for k, v in np.load(OLD / "fixture_stage4.npz", allow_pickle=False).items()
    }
    metas = [
        m
        for m in read_csv(OLD / "fixture_inputs_index.csv")
        if m["target_stage"] == "4"
    ]
    assert len(metas) == 120
    inputs = {
        k: source[k].copy()
        for k in [
            "w_local",
            "U",
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
    permutation = np.empty((120, 16, 64), int)
    for meta in metas:
        i = int(meta["array_index"])
        for epoch in range(16):
            permutation[i, epoch] = rng(
                [meta["snapshot_id"], epoch + 1, "D-only-batch-order"]
            ).permutation(64)
    inputs["D_epoch_permutations"] = permutation
    np.savez_compressed(HERE / "d_only_inputs.npz", **inputs)
    inputs_sha = sha(HERE / "d_only_inputs.npz")
    step_rows = []
    epoch_rows = []
    readouts = []
    pairs = []
    checks = []
    input_index = []
    access = set()
    max_softmax_error = 0.0
    min_alpha = 1.0
    all_invariance = True
    for meta in metas:
        i = int(meta["array_index"])
        local = inputs["w_local"][i]
        U = inputs["U"][i]
        x = inputs["X_D"][i]
        y = inputs["y_D"][i]
        step_row_start = len(step_rows)
        pre_local = array_sha(local)
        pre_members = array_sha(U)
        beta = np.zeros(3)
        moment = np.zeros(3)
        second = np.zeros(3)
        base = {
            k: meta[k]
            for k in ["snapshot_id", "run_id", "dataset", "profile", "seed", "client"]
        }
        uniform = report(beta, inputs, i, a)
        readouts.append(
            dict(base, variant="uniform_zero_logits", fixed_endpoint_step=0, **uniform)
        )
        t = 0
        for epoch in range(1, 17):
            order = permutation[i, epoch - 1]
            for offset in range(0, 64, 8):
                batch = order[offset : offset + 8]
                old = beta.copy()
                xb = x[batch]
                yb = y[batch]
                loss, gradient, alpha = loss_gradient(old, xb, yb, local, U, a)
                access.add("D")
                t += 1
                moment = 0.9 * moment + 0.1 * gradient
                second = 0.999 * second + 0.001 * gradient**2
                update = (
                    0.01
                    * (moment / (1 - 0.9**t))
                    / (np.sqrt(second / (1 - 0.999**t)) + 1e-8)
                )
                beta = old - update
                loss_after, _, after_alpha = loss_gradient(beta, xb, yb, local, U, a)
                max_softmax_error = max(
                    max_softmax_error, abs(float(after_alpha.sum()) - 1)
                )
                min_alpha = min(min_alpha, float(after_alpha.min()))
                step_rows.append(
                    dict(
                        base,
                        epoch=epoch,
                        step=t,
                        batch_in_epoch=offset // 8 + 1,
                        D_rows=canonical(batch.tolist()),
                        fitting_role="D",
                        loss_kind="toy_MSE",
                        batch_MSE_before=loss,
                        batch_MSE_after=loss_after,
                        **{
                            f"logit_before_{j + 1}": float(v) for j, v in enumerate(old)
                        },
                        **{
                            f"gradient_{j + 1}": float(v)
                            for j, v in enumerate(gradient)
                        },
                        **{f"moment_{j + 1}": float(v) for j, v in enumerate(moment)},
                        **{
                            f"second_moment_{j + 1}": float(v)
                            for j, v in enumerate(second)
                        },
                        **{
                            f"logit_after_{j + 1}": float(v) for j, v in enumerate(beta)
                        },
                        **{
                            f"alpha_after_{j}": float(v)
                            for j, v in enumerate(after_alpha)
                        },
                        null_logit=0.0,
                    )
                )
            dl, _, alpha = loss_gradient(beta, x, y, local, U, a)
            epoch_rows.append(
                dict(
                    base,
                    epoch=epoch,
                    cumulative_steps=t,
                    full_D_MSE=dl,
                    readout_role="D",
                    **{f"logit_{j + 1}": float(v) for j, v in enumerate(beta)},
                    **{f"alpha_{j}": float(v) for j, v in enumerate(alpha)},
                    endpoint_selected_by="fixed epoch16, not held-out outcome",
                )
            )
        assert t == 128
        fitted = report(beta, inputs, i, a)
        readouts.append(
            dict(base, variant="D_only_logit_fit", fixed_endpoint_step=128, **fitted)
        )
        pairs.append(
            dict(
                base,
                **{
                    "gain_MSE_" + role: uniform["MSE_" + role] - fitted["MSE_" + role]
                    for role in ("D", "V", "A", "T")
                },
                uniform_D_RMS=uniform["external_D_RMS"],
                fitted_D_RMS=fitted["external_D_RMS"],
                gain_definition="MSE(uniform)-MSE(D-fitted); positive favors fitted",
                fit_epochs=16,
                fit_steps=128,
                local_sha256=pre_local,
                member_matrix_sha256=pre_members,
                scope="D-only numerical logit MSE diagnostic, not native detector training",
            )
        )
        for label, point in [("initial", np.zeros(3)), ("fixed_endpoint", beta)]:
            _, analytic, _ = loss_gradient(point, x, y, local, U, a)
            for j in range(3):
                delta = np.zeros(3)
                delta[j] = 1e-6
                plus = loss_gradient(point + delta, x, y, local, U, a)[0]
                minus = loss_gradient(point - delta, x, y, local, U, a)[0]
                numerical = (plus - minus) / 2e-6
                checks.append(
                    dict(
                        base,
                        point=label,
                        coordinate=j + 1,
                        analytic_gradient=float(analytic[j]),
                        central_difference=float(numerical),
                        absolute_error=abs(float(analytic[j]) - numerical),
                        check_role="D",
                    )
                )
        all_invariance &= pre_local == array_sha(local) and pre_members == array_sha(U)
        input_index.append(
            dict(
                base,
                input_file="d_only_inputs.npz",
                input_file_sha256=inputs_sha,
                array_index=i,
                local_sha256=pre_local,
                member_matrix_sha256=pre_members,
                **{
                    role + "_arrays_sha256": hashlib.sha256(
                        (
                            array_sha(inputs["X_" + role][i])
                            + array_sha(inputs["y_" + role][i])
                        ).encode()
                    ).hexdigest()
                    for role in ("D", "V", "A", "T")
                },
                D_permutation_sha256=array_sha(permutation[i]),
                step_log_file="d_only_step_log.csv",
                step_rows=128,
                step_row_start=step_row_start,
                step_row_end=len(step_rows) - 1,
                sample_kind="independent numerical arrays copied without change from prior fixture",
            )
        )
    assert access == {"D"} and all_invariance
    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["dataset"], row["profile"], row["seed"], row["run_id"]].append(row)
    seed_pairs = []
    for key, rows in sorted(grouped.items()):
        assert len(rows) == 4
        seed_pairs.append(
            dict(
                dataset=key[0],
                profile=key[1],
                seed=key[2],
                run_id=key[3],
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
    for ds, profile in sorted({(r["dataset"], r["profile"]) for r in seed_pairs}):
        block = [r for r in seed_pairs if (r["dataset"], r["profile"]) == (ds, profile)]
        assert len(block) == 5
        for role in ("V", "A", "T"):
            vals = [r["gain_MSE_" + role] for r in block]
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals)
            half = 2.7764451051977987 * sd / math.sqrt(5)
            summary.append(
                dict(
                    dataset=ds,
                    profile=profile,
                    role=role,
                    n_source_seed_blocks=5,
                    mean_gain_MSE=mu,
                    sample_SD=sd,
                    CI95_lo=mu - half,
                    CI95_hi=mu + half,
                    negative_seed_blocks=sum(v < -1e-6 for v in vals),
                    interpretation="descriptive numerical fixture interval, not empirical native superiority",
                )
            )
    products = {
        "d_only_inputs_index.csv": input_index,
        "d_only_readouts.csv": readouts,
        "d_only_pairs.csv": pairs,
        "d_only_epoch_log.csv": epoch_rows,
        "d_only_step_log.csv": step_rows,
        "d_only_gradient_checks.csv": checks,
        "d_only_seed_pairs.csv": seed_pairs,
        "d_only_condition_summary.csv": summary,
    }
    for name, rows in products.items():
        write_csv(name, rows)
    assert (
        len(readouts),
        len(pairs),
        len(epoch_rows),
        len(step_rows),
        len(checks),
    ) == (240, 120, 1920, 15360, 720)
    max_grad = max(r["absolute_error"] for r in checks)
    assert max_grad < P["finite_difference_absolute_tolerance"]
    assert all(sha(Path(p)) == expected for p, expected in before.items())
    qa = dict(
        status="PASS_D_ONLY_LOGIT_NUMERICAL_DIAGNOSTIC",
        snapshot_pairs=120,
        readouts=240,
        epochs=1920,
        numerical_Adam_updates=15360,
        finite_difference_checks=720,
        maximum_gradient_error=max_grad,
        maximum_softmax_sum_error=max_softmax_error,
        minimum_coefficient=min_alpha,
        fixed_null_logit=0.0,
        D_only_fit_access=sorted(access),
        endpoint_fixed_at_step128=True,
        local_and_member_invariance=all_invariance,
        factorial_effective_amplitude_pairs=120,
        maximum_factorial_D_RMS_difference=max(
            r["D_RMS_absolute_difference"] for r in audit
        ),
        maximum_factorial_elementwise_difference=max(
            r["maximum_elementwise_D_difference"] for r in audit
        ),
        source_files_unchanged=True,
        main_AP_unchanged=True,
        **{
            "negative_gain_" + role + "_snapshot_count": sum(
                r["gain_MSE_" + role] < -1e-6 for r in pairs
            )
            for role in ("D", "V", "A", "T")
        },
        original_controller_validated=False,
        native_training_executed=False,
        physical_measurements=False,
        caveat="D-only fitting uses a toy MSE and a simplified three-external-column block, not the native detector loss/port implementation; beta adaptation intentionally may change effective amplitude",
    )
    (HERE / "d_only_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=True))


if __name__ == "__main__":
    main()
