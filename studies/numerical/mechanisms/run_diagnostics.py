"""Linear mechanism diagnostics."""

from pathlib import Path
from collections import defaultdict
import csv, hashlib, itertools, json, math, statistics
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path("data/CIRA______20261106")
REGISTRY = SOURCE_ROOT / "classroom/data/results.json"
GENERATOR = SOURCE_ROOT / "classroom/generate_scenario.py"
META = SOURCE_ROOT / "classroom/data/metadata.json"
CONFIG = {
    "fixture_version": "EXTENDED_LINEAR_RESIDUAL_FIXTURE_20261107_v1",
    "execution_kind": "closed_form_classroom_arithmetic",
    "original_controller_validation": False,
    "native_detector_validation": False,
    "dimension": 8,
    "donors": 3,
    "donor_column_L2": 1.0,
    "donor_amplitude": 0.28,
    "donor_direction_noise_SD": 0.025,
    "target_shift_SD": 0.14,
    "observation_noise_SD": 0.18,
    "samples": {"D": 64, "V": 128, "A": 128, "T": 256},
    "design_coordinate_variances": [float(v) for v in np.linspace(0.65, 1.35, 8)],
    "configuration_ridge": 0.5,
    "configuration_step": 0.65,
    "beta_ridge": 0.002,
    "beta_center": [1 / 3] * 3,
    "beta_constraint": "beta>=0; sum(beta)=1; seven nonempty active faces",
    "zero_beta_meaning": "mixture disabled; not a feasible fitted simplex vector",
    "inverse_q_bracket": [0.02, 0.97],
    "inverse_iterations": 80,
    "inverse_AP_tolerance": 1e-7,
    "KKT_tolerance": 1e-8,
    "preference_zero_tolerance_MSE": 1e-6,
    "snapshot_sources": {"stage2_entry": 0, "stage4_entry": 2},
    "randomization": "SHA256(dataset,profile,seed,stage,role) with named substreams; no desired ranks or outcome filtering",
    "target_rule": "embed(mean of four recovered client q vectors)+independent fixed Gaussian shift; shared across clients at a source trajectory/stage",
    "local_embedding": "[q-0.5, 0.5*(q-roll(q,-1))]",
    "donor_rule": "normalize(embed(q_peer)-w_local+independent direction noise) for every other client; no configuration or beta dependence",
    "configuration_rule": "delta=configuration_step*solve(X_D.T X_D/n+lambda_C I, X_D.T(y_D-X_D w_local)/n)",
    "response_rule": "prediction=X_role@(w_local+config_on*delta+a*U@beta)",
    "beta_rule": "argmin simplex mean((y_V-X_V w_local-a X_V U beta)^2)+lambda_beta*||beta-uniform||^2",
    "counting_rule": "one path=one 8-dimensional D ridge solve, one configuration vector addition, one beta solver enumerating seven KKT faces; operation counts are arithmetic bookkeeping, not device FLOPs",
    "source_scope": "v2 history_AP is a deterministic surrogate readout; inversion recovers its latent numerical state only",
    "existing_online_evidence": "full main logs already contain 360 collaboration-parent->local Refine and 480 local-parent->collaboration transitions with positive updates; all 360 reverse actions are Refine, not Fresh/configuration edits",
}


def canonical(obj):
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(value, dtype="<f8")
        h.update(canonical({"shape": arr.shape, "dtype": "<f8"}).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def named_rng(identity, part):
    key = canonical([identity, part]).encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    )


def write_csv(name, rows):
    assert rows
    header = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def ap_link(q):
    q = np.asarray(q, dtype=float)
    detection = 1 / (1 + np.exp(-6.4 * (q - 0.30)))
    location = 0.67 + 0.245 * q
    threshold = np.linspace(0.50, 0.95, 10)
    return (
        100
        * detection[..., None]
        / (1 + np.exp((threshold - location[..., None]) / 0.047))
    ).mean(axis=-1)


def inverse_ap(ap):
    ap = np.asarray(ap, dtype=float)
    lo = np.full_like(ap, CONFIG["inverse_q_bracket"][0])
    hi = np.full_like(ap, CONFIG["inverse_q_bracket"][1])
    if np.any(ap < ap_link(lo) - 1e-9) or np.any(ap > ap_link(hi) + 1e-9):
        raise ValueError("history AP outside the declared surrogate-state bracket")
    for _ in range(CONFIG["inverse_iterations"]):
        mid = (lo + hi) / 2
        below = ap_link(mid) < ap
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    q = (lo + hi) / 2
    error = float(np.max(np.abs(ap_link(q) - ap)))
    if error > CONFIG["inverse_AP_tolerance"]:
        raise ValueError("AP inversion residual exceeds tolerance")
    return q, error


def embed(q):
    q = np.asarray(q)
    return np.concatenate((q - 0.5, 0.5 * (q - np.roll(q, -1))))


def d_solve(fixture):
    x, y, w = fixture["X_D"], fixture["y_D"], fixture["w_local"]
    n = len(y)
    delta = CONFIG["configuration_step"] * np.linalg.solve(
        x.T @ x / n + CONFIG["configuration_ridge"] * np.eye(CONFIG["dimension"]),
        x.T @ (y - x @ w) / n,
    )
    return delta


def beta_solve(fixture, local):
    x, y = fixture["X_V"], fixture["y_V"]
    design = CONFIG["donor_amplitude"] * x @ fixture["U"]
    center = np.array(CONFIG["beta_center"])
    ridge = CONFIG["beta_ridge"]
    A = design.T @ design / len(y) + ridge * np.eye(3)
    rhs = design.T @ (y - x @ local) / len(y) + ridge * center
    candidates = []
    for mask in range(1, 8):
        ix = [j for j in range(3) if mask & (1 << j)]
        sub = A[np.ix_(ix, ix)]
        kkt = np.block(
            [[sub, np.ones((len(ix), 1))], [np.ones((1, len(ix))), np.zeros((1, 1))]]
        )
        sol = np.linalg.solve(kkt, np.r_[rhs[ix], 1.0])
        if np.any(sol[:-1] < -1e-10):
            continue
        beta = np.zeros(3)
        beta[ix] = np.maximum(sol[:-1], 0)
        beta /= beta.sum()
        objective = float(
            np.mean((y - x @ local - design @ beta) ** 2)
            + ridge * np.sum((beta - center) ** 2)
        )
        candidates.append((objective, mask, beta))
    if not candidates:
        raise ValueError("no feasible simplex face")
    objective, face, beta = min(candidates, key=lambda v: (v[0], v[1]))
    gradient = 2 * (A @ beta - rhs)
    active = beta > 1e-9
    active_level = float(np.mean(gradient[active]))
    residual = max(
        float(np.max(np.abs(gradient[active] - active_level))),
        max(0.0, float(np.max(active_level - gradient[~active])))
        if np.any(~active)
        else 0.0,
        abs(float(beta.sum()) - 1),
        max(0.0, -float(beta.min())),
    )
    if residual > CONFIG["KKT_tolerance"]:
        raise ValueError("simplex ridge KKT violation")
    uniform_obj = float(np.mean((y - x @ local - design @ center) ** 2))
    if objective > uniform_obj + 1e-10:
        raise ValueError(
            "uniform simplex baseline not dominated in the regularized V objective"
        )
    return beta, {
        "beta_objective_V": objective,
        "selected_face": face,
        "KKT_residual": residual,
        "feasible_faces": len(candidates),
        "KKT_systems_solved": 7,
    }


def evaluate(fixture, local, beta):
    effective = local + CONFIG["donor_amplitude"] * fixture["U"] @ beta
    result = {
        "MSE_" + role: float(
            np.mean((fixture["y_" + role] - fixture["X_" + role] @ effective) ** 2)
        )
        for role in ("D", "V", "A", "T")
    }
    result["effective_residual_L2"] = float(
        np.linalg.norm(CONFIG["donor_amplitude"] * fixture["U"] @ beta)
    )
    return result


def make_fixture(record, target_stage, client):
    setting = record["setting"]
    source_index = target_stage - 2
    ap = np.array(record["history_AP"][source_index], float)
    q_all, inverse_error = inverse_ap(ap)
    identity = [
        setting["dataset"],
        setting["profile"],
        int(setting["seed"]),
        target_stage,
    ]
    w_local = embed(q_all[client])
    w_truth = embed(q_all.mean(axis=0)) + named_rng(
        identity, "shared-target-shift"
    ).normal(0, CONFIG["target_shift_SD"], 8)
    fixture = {
        "q_all": q_all,
        "q_local": q_all[client].copy(),
        "w_local": w_local,
        "w_truth": w_truth,
    }
    peer_ids = [i for i in range(4) if i != client]
    columns = []
    for peer in peer_ids:
        raw = (
            embed(q_all[peer])
            - w_local
            + named_rng(identity, ["donor-direction", client, peer]).normal(
                0, CONFIG["donor_direction_noise_SD"], 8
            )
        )
        norm = np.linalg.norm(raw)
        if norm < 1e-10:
            raise ValueError(
                "degenerate donor direction; no outcome-dependent replacement allowed"
            )
        columns.append(raw / norm)
    fixture["U"] = np.stack(columns, axis=1)
    variance = np.array(CONFIG["design_coordinate_variances"])
    for role, n in CONFIG["samples"].items():
        rng = named_rng(identity, ["independent-role", role])
        x = rng.normal(0, 1, (n, 8)) * np.sqrt(variance)
        fixture["X_" + role] = x
        fixture["y_" + role] = x @ w_truth + rng.normal(
            0, CONFIG["observation_noise_SD"], n
        )
    fixture["delta"] = d_solve(fixture)
    meta = dict(
        snapshot_id=f"{record['run_id']}:stage{target_stage}:client{client + 1}",
        run_id=record["run_id"],
        dataset=setting["dataset"].replace("ControlledNATIVE_RUN", "ControlledFog"),
        profile=setting["profile"],
        seed=int(setting["seed"]),
        client=client + 1,
        target_stage=target_stage,
        source_release_stage=target_stage - 1,
        source_round=(int(setting["round_budget"]) // 4) * (target_stage - 1),
        snapshot_sha256=array_sha(q_all),
        local_sha256=array_sha(w_local),
        donor_sha256=array_sha(fixture["U"]),
        configuration_delta_sha256=array_sha(fixture["delta"]),
        inverse_AP_error=inverse_error,
        donor_peer_ids=canonical([i + 1 for i in peer_ids]),
        fixture_kind="extended_linear_residual_fixture",
        measured_detector_AP=False,
    )
    return fixture, meta


def build_npz(name, fixtures, metas):
    values = {key: np.stack([x[key] for x in fixtures]) for key in fixtures[0]}
    values.update(
        snapshot_id=np.array([m["snapshot_id"] for m in metas]),
        run_id=np.array([m["run_id"] for m in metas]),
        client=np.array([m["client"] for m in metas]),
        seed=np.array([m["seed"] for m in metas]),
    )
    np.savez_compressed(HERE / name, **values)
    return {
        "path": name,
        "sha256": file_sha(HERE / name),
        "snapshots": len(metas),
        "arrays": {k: list(v.shape) for k, v in values.items()},
    }


def seed_aggregate(comparisons):
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[
            (
                row["diagnostic"],
                row["contrast"],
                row["dataset"],
                row["profile"],
                row["seed"],
                row["run_id"],
            )
        ].append(row)
    aggregated = []
    for (diag, contrast, ds, profile, seed, run_id), rows in sorted(grouped.items()):
        if len(rows) != 4:
            raise ValueError("seed block must contain all four clients")
        output = dict(
            diagnostic=diag,
            contrast=contrast,
            dataset=ds,
            profile=profile,
            seed=seed,
            run_id=run_id,
            clients=4,
        )
        for role in ("D", "V", "A", "T"):
            output["gain_MSE_" + role] = statistics.mean(
                x["gain_MSE_" + role] for x in rows
            )
        aggregated.append(output)
    by_condition = defaultdict(list)
    for row in aggregated:
        by_condition[
            (row["diagnostic"], row["contrast"], row["dataset"], row["profile"])
        ].append(row)
    summary = []
    for key, rows in sorted(by_condition.items()):
        if len(rows) != 5 or len({r["seed"] for r in rows}) != 5:
            raise ValueError("condition requires five independent source seed labels")
        for role in ("V", "A", "T"):
            sample = [x["gain_MSE_" + role] for x in rows]
            mu, sd = statistics.mean(sample), statistics.stdev(sample)
            half = 2.7764451051977987 * sd / math.sqrt(5)
            summary.append(
                dict(
                    diagnostic=key[0],
                    contrast=key[1],
                    dataset=key[2],
                    profile=key[3],
                    role=role,
                    n_seed_blocks=5,
                    mean_gain_MSE=mu,
                    sample_SD=sd,
                    CI95_lo=mu - half,
                    CI95_hi=mu + half,
                    negative_seed_blocks=sum(x < -1e-6 for x in sample),
                    interpretation="descriptive classroom fixture interval, not native empirical superiority",
                )
            )
    return aggregated, summary


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    before = {str(p): file_sha(p) for p in (REGISTRY, GENERATOR, META)}
    records = json.loads(REGISTRY.read_bytes())
    main_records = sorted(
        [
            r
            for r in records
            if r["setting"]["experiment"] == "N01" and r["setting"]["method"] == "CIRA"
        ],
        key=lambda r: (
            r["setting"]["dataset"],
            r["setting"]["profile"],
            r["setting"]["seed"],
        ),
    )
    assert len(main_records) == 30 and all(
        r["scenario_revision"] == "CLASSROOM_LATENT_STATE_20261106_v2"
        for r in main_records
    )
    frozen = dict(
        CONFIG,
        source_files_sha256=before,
        main_source_trajectories=30,
        independent_seed_labels_per_condition=5,
        clients_per_source_trajectory=4,
    )
    frozen_path = HERE / "generation_parameters.json"
    if frozen_path.exists() and json.loads(frozen_path.read_bytes()) != frozen:
        raise RuntimeError("refusing a conflicting fixture configuration")
    frozen_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stage_data = {}
    indexes = []
    for stage in (2, 4):
        fixtures, metas = [], []
        for record in main_records:
            for client in range(4):
                fixture, meta = make_fixture(record, stage, client)
                fixtures.append(fixture)
                metas.append(meta)
        stage_data[stage] = (fixtures, metas)
        name = f"fixture_stage{stage}.npz"
        descriptor = build_npz(name, fixtures, metas)
        for position, meta in enumerate(metas):
            indexes.append(
                dict(
                    meta,
                    array_file=name,
                    array_index=position,
                    array_file_sha256=descriptor["sha256"],
                )
            )

    cells, interaction, paths, path_pairs, beta_reads, beta_pairs, comparisons = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    kkt_residuals, invariants = [], []
    uniform = np.array(CONFIG["beta_center"])
    zero = np.zeros(3)
    for fixture, meta in zip(*stage_data[2]):
        local_hash = array_sha(fixture["w_local"])
        donor_hash = array_sha(fixture["U"])
        grid = {}
        for config_on, mix_on in itertools.product((0, 1), repeat=2):
            beta = uniform if mix_on else zero
            result = evaluate(
                fixture, fixture["w_local"] + config_on * fixture["delta"], beta
            )
            grid[config_on, mix_on] = result
            cells.append(
                dict(
                    meta,
                    config_on=config_on,
                    mix_on=mix_on,
                    beta=canonical(beta.tolist()),
                    donor_amplitude=CONFIG["donor_amplitude"],
                    local_parameter_L2=float(np.linalg.norm(fixture["w_local"])),
                    configuration_delta_L2=float(np.linalg.norm(fixture["delta"])),
                    config_factor_count=1,
                    fixed_beta_readout_count=1,
                    **result,
                )
            )
        row = dict(meta)
        for role in ("D", "V", "A", "T"):
            none = grid[0, 0]["MSE_" + role] - grid[1, 0]["MSE_" + role]
            mixed = grid[0, 1]["MSE_" + role] - grid[1, 1]["MSE_" + role]
            row["config_gain_none_" + role] = none
            row["config_gain_uniform_" + role] = mixed
            row["gain_MSE_" + role] = mixed - none
            row["strict_preference_reversal_" + role] = bool(
                none * mixed < 0 and min(abs(none), abs(mixed)) > 1e-6
            )
        interaction.append(row)
        comparisons.append(
            dict(
                row,
                diagnostic="configuration_x_mixture",
                contrast="configuration_gain_uniform_minus_none",
            )
        )

        terminal = {}
        for path in (
            "configuration_then_beta_fit",
            "beta_fit_then_configuration_keep",
            "beta_fit_then_configuration_clear",
        ):
            delta = d_solve(fixture)
            assert np.array_equal(delta, fixture["delta"])
            fitting_local = (
                fixture["w_local"] + delta
                if path == "configuration_then_beta_fit"
                else fixture["w_local"]
            )
            fitted, audit = beta_solve(fixture, fitting_local)
            effective_beta = zero if path.endswith("_clear") else fitted
            result = evaluate(fixture, fixture["w_local"] + delta, effective_beta)
            terminal[path] = result
            kkt_residuals.append(audit["KKT_residual"])
            paths.append(
                dict(
                    meta,
                    path=path,
                    beta_fitted=canonical(fitted.tolist()),
                    beta_effective=canonical(effective_beta.tolist()),
                    beta_fit_local_sha256=array_sha(fitting_local),
                    final_local_sha256=array_sha(fixture["w_local"] + delta),
                    D_ridge_solve_count=1,
                    configuration_add_count=1,
                    beta_solver_count=1,
                    configuration_dimension=8,
                    beta_dimension=3,
                    work_unit="fixed counted analytic operations; not measured GFLOP or optimizer steps",
                    **audit,
                    **result,
                )
            )
        pairs = [
            ("configuration_then_beta_fit", "beta_fit_then_configuration_keep"),
            ("beta_fit_then_configuration_keep", "beta_fit_then_configuration_clear"),
            ("configuration_then_beta_fit", "beta_fit_then_configuration_clear"),
        ]
        for first, second in pairs:
            row = dict(
                meta,
                diagnostic="reverse_continuation",
                contrast=first + "_minus_" + second,
                gain_definition="MSE(second)-MSE(first); positive favors first",
                **{
                    "gain_MSE_" + role: terminal[second]["MSE_" + role]
                    - terminal[first]["MSE_" + role]
                    for role in ("D", "V", "A", "T")
                },
            )
            path_pairs.append(row)
            comparisons.append(row)
        invariants.append(
            local_hash == array_sha(fixture["w_local"])
            and donor_hash == array_sha(fixture["U"])
        )

    for fixture, meta in zip(*stage_data[4]):
        local_hash, donor_hash = array_sha(fixture["w_local"]), array_sha(fixture["U"])
        fitted, audit = beta_solve(fixture, fixture["w_local"])
        results = {}
        for name, beta in (
            ("uniform_beta", uniform),
            ("V_only_simplex_ridge_beta", fitted),
        ):
            result = evaluate(fixture, fixture["w_local"], beta)
            results[name] = result
            beta_reads.append(
                dict(
                    meta,
                    variant=name,
                    beta=canonical(beta.tolist()),
                    donor_amplitude=CONFIG["donor_amplitude"],
                    fixed_local_sha256=local_hash,
                    beta_solver_count=int(name != "uniform_beta"),
                    **(audit if name != "uniform_beta" else {}),
                    **result,
                )
            )
        row = dict(
            meta,
            diagnostic="beta_only_fixed_local",
            contrast="fitted_minus_uniform",
            gain_definition="MSE(uniform)-MSE(fitted); positive favors fitted",
            **{
                "gain_MSE_" + role: results["uniform_beta"]["MSE_" + role]
                - results["V_only_simplex_ridge_beta"]["MSE_" + role]
                for role in ("D", "V", "A", "T")
            },
        )
        beta_pairs.append(row)
        comparisons.append(row)
        kkt_residuals.append(audit["KKT_residual"])
        invariants.append(
            local_hash == array_sha(fixture["w_local"])
            and donor_hash == array_sha(fixture["U"])
        )

    aggregated, summary = seed_aggregate(comparisons)
    products = {
        "fixture_inputs_index.csv": indexes,
        "stage2_factorial_cells.csv": cells,
        "stage2_factorial_interactions.csv": interaction,
        "reverse_continuation_paths.csv": paths,
        "reverse_continuation_pairs.csv": path_pairs,
        "stage4_beta_reads.csv": beta_reads,
        "stage4_beta_pairs.csv": beta_pairs,
        "seed_block_pairs.csv": aggregated,
        "condition_summary.csv": summary,
    }
    for name, rows in products.items():
        write_csv(name, rows)
    all_numeric = [
        v
        for rows in products.values()
        for row in rows
        for v in row.values()
        if isinstance(v, (float, np.floating))
    ]
    assert np.isfinite(all_numeric).all() and all(invariants)
    after = {path: file_sha(Path(path)) for path in before}
    assert before == after, "source files changed during diagnostics"
    negative_counts = {
        f"beta_gain_{role}_negative": sum(
            row["gain_MSE_" + role] < -1e-6 for row in beta_pairs
        )
        for role in ("V", "A", "T")
    }
    qa = dict(
        status="PASS_CLOSED_FORM_FIXTURE_INTEGRITY",
        fixture_version=CONFIG["fixture_version"],
        source_files_unchanged=True,
        source_main_AP_unchanged=True,
        source_trajectories=30,
        stage2_snapshots=120,
        stage4_snapshots=120,
        factorial_cells=len(cells),
        reverse_path_endpoints=len(paths),
        beta_readouts=len(beta_reads),
        beta_solvers=480,
        KKT_systems_solved=3360,
        maximum_KKT_residual=max(kkt_residuals),
        maximum_inverse_AP_error=max(m["inverse_AP_error"] for m in indexes),
        local_and_donor_invariance=True,
        donor_unit_L2_max_error=max(
            float(np.max(np.abs(np.linalg.norm(f["U"], axis=0) - 1)))
            for fs, _ in stage_data.values()
            for f in fs
        ),
        uniform_baseline_feasible=True,
        outcome_filtering=False,
        strict_preference_reversals_T=sum(
            row["strict_preference_reversal_T"] for row in interaction
        ),
        **negative_counts,
        source_files_sha256=before,
        original_controller_validated=False,
        native_training_executed=False,
        physical_measurements=False,
        caveat="Numerical consistency and classroom controls only. Not original CIRA/YOLO verification or native AP evidence.",
    )
    (HERE / "QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths_to_hash = [
        p for p in HERE.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"
    ]
    manifest = {p.name: file_sha(p) for p in sorted(paths_to_hash)}
    (HERE / "SHA256SUMS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=True))


if __name__ == "__main__":
    main()
