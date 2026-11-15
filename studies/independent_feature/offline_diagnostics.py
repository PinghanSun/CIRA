from pathlib import Path
import argparse, json, csv, sys, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from engine import fixture, AP, loss_grad, masks, sigmoid, digest, SCOPE


def write(p, rows):
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pick(data, client, domain):
    return tuple(x[client, domain] for x in data)


def evaluate(data, W):
    return AP(*data, W)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "output")
    a = p.parse_args()
    out = a.out
    records = json.loads((out / "results.json").read_bytes())
    profiles = json.loads((ROOT / "inputs/profiles.json").read_bytes())
    config = []
    recv = []
    interactions = []
    stage_rows = []
    audits = []
    workfits = []
    for rec in [x for x in records if x["setting"]["experiment"] == "N01"]:
        c = rec["setting"]
        d = profiles[c["profile"]]["width"]
        ff = fixture(
            c["dataset"],
            d,
            c["seed"],
            str(c["alpha"]),
            profiles[c["profile"]]["projection_noise"],
        )
        for s in range(1, 5):
            p = out / "runs" / rec["run_id"] / f"stage_{s}.npz"
            if not p.exists():
                continue
            with np.load(p) as f:
                theta = (
                    f["frozen_base"]
                    + f["adapter"]
                    + f["receiving_weight"][:, None, None]
                    * f["external"]
                    * f["coupling"][:, None, None]
                )
                ap = evaluate(ff["T"], theta[:, None])
                assert np.allclose(ap, f["T_AP"], atol=1e-8, rtol=0)
                for cl in range(4):
                    audits.append(
                        dict(
                            run_id=rec["run_id"],
                            stage=s,
                            client=cl + 1,
                            A_entry=float(f["A_entry"][cl]),
                            A_candidate=float(f["A_candidate"][cl]),
                            A_accepted=bool(f["A_accepted"][cl]),
                            checkpoint=p.relative_to(out).as_posix(),
                            checkpoint_sha256=hashlib.sha256(
                                p.read_bytes()
                            ).hexdigest(),
                        )
                    )
                    for dom in f["order"][cl, :s]:
                        stage_rows.append(
                            dict(
                                run_id=rec["run_id"],
                                stage=s,
                                client=cl + 1,
                                domain=int(dom),
                                AP50_95=float(ap[cl, dom].mean()),
                                AP50=float(ap[cl, dom, 0]),
                                AP75=float(ap[cl, dom, 5]),
                                current=bool(dom == f["order"][cl, s - 1]),
                                checkpoint=p.relative_to(out).as_posix(),
                            )
                        )
        if c["method"] != "CIRA":
            continue
        for completed, target in [(1, 2), (3, 4)]:
            with np.load(out / "runs" / rec["run_id"] / f"stage_{completed}.npz") as f:
                base = f["frozen_base"]
                WW = f["adapter"]
                order = f["order"]
                zz = f["z"]
                hh = f["h"]
                mom = f["momentum"]
            domains = order[:, target - 1]
            old_domain = order[:, completed - 1]
            proto = np.stack([ff["D"][0][k, domains[k]].mean(axis=0) for k in range(4)])
            oldproto = np.stack(
                [ff["D"][0][k, old_domain[k]].mean(axis=0) for k in range(4)]
            )
            sim = (
                proto
                @ oldproto.T
                / np.maximum(
                    np.linalg.norm(proto, axis=1)[:, None]
                    * np.linalg.norm(oldproto, axis=1)[None],
                    1e-8,
                )
            )
            sim[np.arange(4), np.arange(4)] = -1e9
            donors = sim.argmax(axis=1)
            for cl in range(4):
                E = WW[donors[cl]]
                Ddata = pick(ff["D"], cl, domains[cl])
                Vdata = pick(ff["V"], cl, domains[cl])
                Tdata = pick(ff["T"], cl, domains[cl])
                common = dict(
                    run_id=rec["run_id"],
                    dataset=c["dataset"],
                    profile=c["profile"],
                    seed=c["seed"],
                    client=cl + 1,
                    target_stage=target,
                    parent_checkpoint=f"runs/{rec['run_id']}/stage_{completed}.npz",
                    member_version=digest(E),
                    member_available_stage=completed,
                )
                if target == 2:
                    vals = {}
                    for label, z in [
                        ("z0", int(zz[cl])),
                        ("z1", (int(zz[cl]) + 1) % 42),
                    ]:
                        m = masks(np.array([[z]]), d)[0, 0]
                        lr = np.array([0.02, 0.035, 0.05, 0.065, 0.085, 0.11])[
                            int(hh[cl]) % 6
                        ]
                        for context, beta in [("empty", 0.0), ("fixed_peer", 0.35)]:
                            W = np.zeros_like(base)
                            v = np.zeros_like(base)
                            coupling = c["gamma_int"] * (0.75 + 0.5 * m.mean())
                            for _ in range(32):
                                _, g = loss_grad(*Ddata, base + W + beta * E * coupling)
                                v = 0.85 * v + g
                                W -= lr * v * m[:, None]
                            theta = base + W + beta * E * coupling
                            V = evaluate(Vdata, theta)
                            T = evaluate(Tdata, theta)
                            vals[label, context] = float(T.mean())
                            path = (
                                out
                                / "offline_states"
                                / f"{rec['run_id']}-C{cl + 1}-{label}-{context}.npz"
                            )
                            path.parent.mkdir(exist_ok=True)
                            np.savez_compressed(
                                path,
                                base=base,
                                adapter=W,
                                momentum=v,
                                external=E,
                                beta=np.array(beta),
                                coupling=np.array(coupling),
                                scope=np.array(SCOPE),
                            )
                            config.append(
                                dict(
                                    **common,
                                    configuration=label,
                                    z=z,
                                    context=context,
                                    phi_epochs=32,
                                    beta_epochs=0,
                                    local_initialization="same zero adapter and zero momentum",
                                    V_AP=float(V.mean()),
                                    T_current_AP=float(T.mean()),
                                    T_AP50=float(T[0]),
                                    T_AP75=float(T[5]),
                                    dense_fit_ops=32 * 6 * 16 * 13 * int(m.sum()),
                                    state=path.relative_to(out).as_posix(),
                                    state_sha256=hashlib.sha256(
                                        path.read_bytes()
                                    ).hexdigest(),
                                )
                            )
                    de = vals["z1", "empty"] - vals["z0", "empty"]
                    dp = vals["z1", "fixed_peer"] - vals["z0", "fixed_peer"]
                    interactions.append(
                        dict(
                            **common,
                            delta_empty=de,
                            delta_peer=dp,
                            interaction=dp - de,
                            strict_preference_flip=bool(de * dp < 0),
                        )
                    )

                    m = masks(np.array([[zz[cl]]]), d)[0, 0]
                    lr = np.array([0.02, 0.035, 0.05, 0.065, 0.085, 0.11])[
                        int(hh[cl]) % 6
                    ]
                    coupling = c["gamma_int"] * (0.75 + 0.5 * m.mean())
                    for variant in ["Inherited", "WeightsOnly", "ColdLocal"]:
                        W = (
                            WW[cl].copy()
                            if variant != "ColdLocal"
                            else np.zeros_like(base)
                        )
                        v = (
                            mom[cl].copy()
                            if variant == "Inherited"
                            else np.zeros_like(base)
                        )
                        for _ in range(32):
                            _, g = loss_grad(*Ddata, base + W + 0.35 * E * coupling)
                            v = 0.85 * v + g
                            W -= lr * v * m[:, None]
                        V = evaluate(Vdata, base + W + 0.35 * E * coupling)
                        T = evaluate(Tdata, base + W + 0.35 * E * coupling)
                        path = (
                            out
                            / "offline_states"
                            / f"{rec['run_id']}-C{cl + 1}-work-{variant}.npz"
                        )
                        np.savez_compressed(
                            path,
                            base=base,
                            adapter=W,
                            momentum=v,
                            external=E,
                            beta=np.array(0.35),
                            coupling=np.array(coupling),
                            scope=np.array(SCOPE),
                        )
                        workfits.append(
                            dict(
                                **common,
                                variant=variant,
                                z=int(zz[cl]),
                                h=int(hh[cl]),
                                phi_epochs=32,
                                beta_epochs=0,
                                dense_fit_ops=32 * 6 * 16 * 13 * int(m.sum()),
                                V_AP=float(V.mean()),
                                T_current_AP=float(T.mean()),
                                T_AP50=float(T[0]),
                                T_AP75=float(T[5]),
                                state=path.relative_to(out).as_posix(),
                                state_sha256=hashlib.sha256(
                                    path.read_bytes()
                                ).hexdigest(),
                                comparison_scope="fixed-work snapshot intervention, not full-trajectory superiority",
                            )
                        )
                else:
                    W = WW[cl].copy()
                    W_hash = digest(W)
                    m = masks(np.array([[zz[cl]]]), d)[0, 0]
                    coupling = c["gamma_int"] * (0.75 + 0.5 * m.mean())
                    logit = 0.0
                    beforeV = evaluate(Vdata, base + W + 0.5 * E * coupling)
                    beforeT = evaluate(Tdata, base + W + 0.5 * E * coupling)
                    for _ in range(16):
                        beta = float(sigmoid(logit))
                        _, g = loss_grad(*Vdata, base + W + beta * E * coupling)
                        derivative = float(np.sum(g * E)) * coupling * beta * (1 - beta)
                        logit -= c["eta_beta"] * derivative
                    beta = float(sigmoid(logit))
                    assert digest(W) == W_hash
                    afterV = evaluate(Vdata, base + W + beta * E * coupling)
                    afterT = evaluate(Tdata, base + W + beta * E * coupling)
                    for label, b, V, T in [
                        ("Uniform", 0.5, beforeV, beforeT),
                        ("Fitted", beta, afterV, afterT),
                    ]:
                        path = (
                            out
                            / "offline_states"
                            / f"{rec['run_id']}-C{cl + 1}-receive-{label}.npz"
                        )
                        np.savez_compressed(
                            path,
                            base=base,
                            adapter=W,
                            momentum=mom[cl],
                            external=E,
                            beta=np.array(b),
                            coupling=np.array(coupling),
                            effective_weights=base + W + b * E * coupling,
                            local_optimizer_frozen=np.array(True),
                            beta_epochs=np.array(0 if label == "Uniform" else 16),
                            scope=np.array(SCOPE),
                        )
                        recv.append(
                            dict(
                                **common,
                                branch=label,
                                beta=b,
                                null_weight=1 - b,
                                phi_frozen=True,
                                phi_hash=W_hash,
                                beta_epochs=0 if label == "Uniform" else 16,
                                V_AP=float(V.mean()),
                                T_current_AP=float(T.mean()),
                                T_AP50=float(T[0]),
                                T_AP75=float(T[5]),
                                optimizer="scalar softmax-logit gradient; independent diagnostic",
                                state=path.relative_to(out).as_posix(),
                                state_sha256=hashlib.sha256(
                                    path.read_bytes()
                                ).hexdigest(),
                            )
                        )
    write(out / "fixed_work_inheritance.csv", workfits)
    write(out / "offline_configuration.csv", config)
    write(out / "offline_receiving.csv", recv)
    write(out / "offline_interactions.csv", interactions)
    write(out / "main_historical_T_three_APs.csv", stage_rows)
    write(out / "main_release_audits.csv", audits)
    proof = dict(
        status="PASS_DIRECT_SNAPSHOT_DIAGNOSTICS",
        fixed_work_snapshot_fits=len(workfits),
        fixed_work_snapshots=len(workfits) // 3,
        config_fits=len(config),
        snapshot_interactions=len(interactions),
        strict_preference_flips=sum(x["strict_preference_flip"] for x in interactions),
        receiving_evaluations=len(recv),
        receiver_fits=len(recv) // 2,
        local_state_unchanged_in_receiver_fit=True,
        main_historical_seen_domain_rows=len(stage_rows),
        main_client_stage_audits=len(audits),
        primary_adopted_results_changed=False,
        native_YOLO=False,
    )
    (out / "offline_verification.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8"
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
