"""Generate a disclosed classroom scenario, never detector-training evidence."""

from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, gzip, hashlib, json, math, time, os, concurrent.futures

for option in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(option, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "classroom" / "data"
SOURCE = Path("data/cases.json")
REV = "CLASSROOM_LATENT_STATE_20261106_v2"
FIELDS = [
    "SeenAP_final",
    "CurrentAP_final",
    "MinClientAP_final",
    "Forget_pp",
    "BWT_pp",
    "AP50_final",
    "AP75_final",
    "Domain0_AP",
    "Domain1_AP",
    "Domain2_AP",
    "Domain3_AP",
    "energy_J",
    "P95_ms",
    "peak_memory_MiB",
    "model_size_MiB",
    "online_GFLOP",
    "selection_evaluations",
    "received_component_MiB",
    "online_time_s",
    "actions",
]
METHODS = {
    "Local": (0.014, 0.00080, 0.0, 0.000, 0.92),
    "Serial": (0.014, 0.00048, 0.45, 0.004, 1.00),
    "Fixed-collab": (0.014, 0.00055, 0.70, 0.004, 1.04),
    "Round-robin": (0.013, 0.00045, 0.55, 0.004, 1.02),
    "Local-NAS-HPO": (0.012, 0.00065, 0.0, 0.009, 0.98),
    "DPFL": (0.0135, 0.00038, 0.65, 0.005, 1.02),
    "DSpodFL": (0.017, 0.00055, 0.40, 0.003, 0.72),
    "DA-DPFL": (0.0135, 0.00050, 0.48, 0.003, 0.75),
    "DCFCL": (0.0145, 0.00032, 0.72, 0.007, 1.08),
    "pFedDIL": (0.015, 0.00013, 0.60, 0.008, 1.14),
    "dFLMoE": (0.0145, 0.00030, 0.75, 0.007, 1.17),
    "PerFedSDE-NAS": (0.013, 0.00042, 0.35, 0.010, 1.13),
    "DeModCL": (0.015, 0.00022, 0.72, 0.009, 1.08),
    "CIRA": (0.0145, 0.00019, 0.70, 0.011, 1.09),
}
ABLATIONS = {
    "No-Refine",
    "No-cross-family-parent",
    "No-gain",
    "Discrete",
    "NAS-only",
    "HPO-only",
}


def sid(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "little")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def norm(x):
    return str(x).split("（")[0].split("(")[0].strip()


def rng_for(c, part, method_specific=False):
    key = [c["dataset"], c["profile"], c["seed"], part]
    if method_specific:
        method = norm(c["method"])
        key.append("CIRA" if method in ABLATIONS else method)
    return np.random.default_rng(sid(json.dumps(key, sort_keys=True)))


def write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def quantized(values):
    return [float(format(float(x), ".10g")) for x in values]


def ap_links(q):
    """One shared bounded monotone readout at ten IoU thresholds."""
    detection = 1 / (1 + np.exp(-6.4 * (q - 0.30)))
    location = 0.67 + 0.245 * q
    threshold = np.linspace(0.50, 0.95, 10)
    a = (
        100
        * detection[..., None]
        / (1 + np.exp((threshold - location[..., None]) / 0.047))
    )
    return a.mean(axis=-1), a[..., 0], a[..., 5]


def generate_batch(cases, batch_index, write_files=True):
    B = len(cases)
    R = int(cases[0]["round_budget"])
    L = R // 4
    assert all(int(c["round_budget"]) == R for c in cases)
    methods = [norm(c["method"]) for c in cases]
    full = [("CIRA" if m in ABLATIONS else m) for m in methods]
    profiles = np.array(
        [["YOLOv8n", "YOLO11n", "YOLO26n"].index(c["profile"]) for c in cases]
    )
    width = np.array([16, 24, 32])[profiles]
    alpha = np.array(
        [1000.0 if c["alpha"] == "equal" else float(c["alpha"]) for c in cases]
    )
    hetero = 1 / (1 + 2 * alpha)
    fraction = np.array([float(c["D_fraction"]) for c in cases])
    dn = np.maximum(8, np.rint(64 * fraction)).astype(int)
    Fscale = np.array([float(c["F_scale"]) for c in cases])
    gamma = np.array([float(c["gamma_int"]) for c in cases])
    mod = np.array([float(c["module_setting"]) for c in cases])
    isC = np.array([m == "CIRA" for m in full])
    base_params = np.array([METHODS[m] for m in full])
    rate, forget, collab, bonus, work = [base_params[:, j, None] for j in range(5)]
    eta = np.array([0.65, 0.80, 0.95, 1.08, 1.20, 1.30])[
        [int(c["h"]) % 6 for c in cases]
    ][:, None]
    rate = rate * eta * (0.6 + 0.4 * np.sqrt(fraction[:, None]))
    empty = np.array([c["donor_mode"] == "empty" for c in cases])
    weak = np.array([c["donor_mode"] == "weak" for c in cases])
    unit = (0.22 + 0.075 * profiles) * (dn / 64)
    Fcap = (L * 2 * (0.22 + 0.075 * profiles) * Fscale)[:, None] * np.ones((1, 4))
    evalcost = (0.22 + 0.075 * profiles)[:, None] * 0.16
    auditcost = evalcost * 2
    gen = [rng_for(c, "method-events", True) for c in cases]
    paired = [rng_for(c, "paired-task") for c in cases]
    order = np.empty((B, 4, 4), int)
    source = np.empty((B, 4, 4))
    aptitude = np.empty((B, 4, 4))
    env = np.zeros((B, 4))
    innovations = np.empty((B, R, 4), np.float32)
    uniforms = np.empty((B, R, 4, 3), np.float32)
    for b, c in enumerate(cases):
        common = paired[b].permutation(4)

        independent_orders = np.stack([paired[b].permutation(4) for _ in range(4)])
        mode = c["order_mode"]
        if mode not in ("common", "common_seeded", "independent", "independent_seeded"):
            raise ValueError("Unrecognized order_mode: " + str(mode))
        order[b] = (
            np.tile(common, (4, 1))
            if mode in ("common", "common_seeded")
            else independent_orders
        )
        ds_penalty = 0.025 if c["dataset"] == "ACDC" else 0.0
        source[b] = (
            0.245
            + 0.030 * profiles[b]
            - 0.019 * np.arange(4)[None, :]
            - ds_penalty
            + paired[b].normal(0, 0.018, (4, 4))
        )
        aptitude[b] = gen[b].normal(0, 0.018, (4, 4))
        innovations[b] = paired[b].normal(0, 1, (R, 4))
        uniforms[b] = gen[b].random((R, 4, 3))
    q = np.clip(source.copy(), 0.03, 0.9)
    target = (
        0.735
        + 0.020 * profiles[:, None, None]
        - 0.026 * np.arange(4)[None, None, :]
        - 0.025 * np.array([c["dataset"] == "ACDC" for c in cases])[:, None, None]
    )
    target = (
        target
        + aptitude
        + bonus[:, :, None] * gamma[:, None, None]
        - 0.032 * hetero[:, None, None]
        - 0.032 * (1 - np.sqrt(fraction[:, None, None]))
    )
    target += 0.006 * (mod[:, None, None] - 1)
    target = np.clip(target, 0.48, 0.88)
    for b, m in enumerate(methods):
        if m in ["NAS-only", "HPO-only"]:
            target[b] -= 0.010
    q0 = q.copy()
    qentry = q.copy()
    released = q.copy()
    best = q.copy()
    family_best = np.repeat(q[:, :, None, :], 2, axis=2)
    family = np.zeros((B, 4), int)
    bestfamily = family.copy()
    family_hist = np.zeros((B, 4, 2), int)
    family_tags = np.zeros((B, 4, 2), int)
    H = np.zeros((B, 4), int)
    bestH = H.copy()
    entryH = H.copy()
    entryfamily = family.copy()
    epochs = np.zeros((B, 4), int)
    steps = np.zeros((B, 4), int)
    totalF = np.zeros((B, 4))
    stageF = np.zeros((B, 4))
    totalE = np.zeros((B, 4), int)
    stageE = totalE.copy()
    attempts = np.zeros((B, 4), int)
    stageM = attempts.copy()
    traffic = np.zeros((B, 4))
    remaining = np.zeros((B, 4), int)
    action = np.zeros((B, 4), int)
    affinity = np.ones((B, 4)) * 0.5
    block_before = np.zeros((B, 4))
    block_H = H.copy()
    block_parent = np.full((B, 4), "", object)
    block_family = family.copy()
    block_steps = np.zeros((B, 4), int)
    block_startF = totalF.copy()
    features = np.zeros((B, 4, 25))
    parent_entry = np.zeros((B, 4), bool)
    import_H = np.full((B, 4), -1, int)
    import_family = np.full((B, 4), -1, int)
    imported = np.zeros((B, 4), bool)
    gain_sum = np.zeros((B, 4, 5))
    gain_n = np.zeros((B, 4, 5), int)
    selected_priority = np.zeros((B, 4))
    selected_mu = np.zeros((B, 4))
    bank_q = []
    bank_H = []
    bank_family = []
    vbias = np.stack(
        [rng_for(c, "V-observation").normal(0, 0.007, (4, 4)) for c in cases]
    )
    abias = np.stack(
        [rng_for(c, "A-observation").normal(0, 0.007, (4, 4)) for c in cases]
    )
    vweights = np.stack(
        [rng_for(c, "V-state-readout").normal(0, 0.04, (4, 4)) for c in cases]
    )
    aweights = np.stack(
        [rng_for(c, "A-state-readout").normal(0, 0.04, (4, 4)) for c in cases]
    )
    diagonal = np.zeros((B, 4, 4))
    max_seen = np.zeros_like(diagonal)
    history = np.zeros((B, 4, 4, 4))
    vals = np.zeros((B, R + 1, 20))
    allAP = np.zeros((B, R + 1))
    epochtrace = np.zeros((B, R + 1), int)
    steptrace = epochtrace.copy()
    active_trace = epochtrace.copy()
    active_flags = np.zeros((B, 4), bool)
    bestV = np.zeros((B, 4))
    familyV = np.zeros((B, 4, 2))
    audit_rows = []
    events = []
    latentLoss = np.zeros((B, R + 1))
    attempt_counts_by_action = np.zeros((B, 5), int)
    bi = np.arange(B)[:, None]
    ci = np.arange(4)[None, :]

    def vscore(state, domains):
        return (
            state[bi, ci, domains]
            + vbias[bi, ci, domains]
            + np.sum(vweights * (state - source), axis=-1)
        )

    def ascore(state, domains):
        return (
            state[bi, ci, domains]
            + abias[bi, ci, domains]
            + np.sum(aweights * (state - source), axis=-1)
        )

    def record(r, s):
        a, a50, a75 = ap_links(q)
        domains = order[:, :, s]
        current = a[bi, ci, domains]
        seen = np.take_along_axis(a, order[:, :, : s + 1], axis=2)
        v = vals[:, r]
        v[:, 0] = seen.mean(axis=(1, 2))
        v[:, 1] = current.mean(axis=1)
        v[:, 2] = seen.mean(axis=2).min(axis=1)
        if s:
            old = order[:, :, :s]
            now = np.take_along_axis(a, old, axis=2)
            v[:, 3] = np.maximum(
                np.take_along_axis(max_seen, old, axis=2) - now, 0
            ).mean(axis=(1, 2))
            v[:, 4] = (now - np.take_along_axis(diagonal, old, axis=2)).mean(
                axis=(1, 2)
            )
        v[:, 5] = a50.mean(axis=(1, 2))
        v[:, 6] = a75.mean(axis=(1, 2))
        v[:, 7:11] = a.mean(axis=1)
        seconds = totalF / np.array([0.40, 0.50, 0.60])[profiles, None] + traffic * 0.04
        v[:, 11] = (seconds * np.array([10.5, 11.4, 12.3])[profiles, None]).sum(axis=1)
        v[:, 12] = (
            12
            + profiles * 4
            + collab[:, 0] * 5
            + 3 * (1 - np.exp(-epochs.mean(axis=1) / 500))
        )
        v[:, 13] = (
            np.array([110, 145, 180])[profiles] + collab[:, 0] * 25 + work[:, 0] * 10
        )
        v[:, 14] = np.array([5.6, 6.2, 7.0])[profiles] + collab[:, 0] * 1.7
        v[:, 15] = totalF.sum(axis=1)
        v[:, 16] = totalE.sum(axis=1)
        v[:, 17] = traffic.sum(axis=1)
        v[:, 18] = seconds.max(axis=1)
        v[:, 19] = attempts.sum(axis=1)
        allAP[:, r] = a.mean(axis=(1, 2))
        epochtrace[:, r] = epochs.sum(axis=1)
        steptrace[:, r] = steps.sum(axis=1)
        active_trace[:, r] = active_flags.sum(axis=1)
        latentLoss[:, r] = (1 - q[bi, ci, domains]).mean(axis=1)

    record(0, 0)
    for r in range(1, R + 1):
        s = min((r - 1) // L, 3)
        phase = ((r - 1) % L) / L
        domains = order[:, :, s]
        current = lambda: q[bi, ci, domains]
        if (r - 1) % L == 0:
            qentry = q.copy()
            entryH = H.copy()
            entryfamily = family.copy()
            best = q.copy()
            bestH = H.copy()
            bestfamily = family.copy()
            stageF[:] = 0
            stageE[:] = 1
            stageM[:] = 0
            totalF += evalcost
            stageF += evalcost
            totalE += 1
            bestV = vscore(q, domains)
            family_best = np.repeat(q[:, :, None, :], 2, axis=2)
            familyV = np.repeat(bestV[:, :, None], 2, axis=2)
            family_hist = np.repeat(H[:, :, None], 2, axis=2)
            family_tags = np.repeat(family[:, :, None], 2, axis=2)
            gain_sum[:] = 0
            gain_n[:] = 0
            imported[:] = False
            remaining[:] = 0
        env = 0.80 * env + 0.60 * innovations[:, r - 1]
        available = np.ones((B, 4), bool)
        for b, c in enumerate(cases):
            if empty[b]:
                available[b] = False
            if c["contact_mode"] == "visibility_half":
                available[b] &= uniforms[b, r - 1, :, 0] > 0.5
            if "interrupt" in c["contact_mode"] and 0.48 <= phase < 0.64:
                available[b] = False

        peer_values = released[
            np.arange(B)[:, None, None],
            np.arange(4)[None, None, :],
            domains[:, :, None],
        ]
        peer_values[:, np.arange(4), np.arange(4)] = -1
        peer_skill = peer_values.max(axis=-1)
        peer_skill = np.where(weak[:, None], 0.25 * peer_skill, peer_skill)
        peer_skill = np.where(available, peer_skill, 0)
        affinity = np.clip(0.50 + (peer_skill - current()), 0.05, 0.95)
        begin = (
            (remaining <= 0)
            & (stageM < np.array([c["M"] for c in cases])[:, None])
            & (stageE < np.array([c["E"] for c in cases])[:, None])
        )
        begin &= stageF + evalcost + auditcost + 0.001 < Fcap
        for b, c in enumerate(cases):
            m = methods[b]
            base = full[b]
            for k in np.flatnonzero(begin[b]):
                if base in [
                    "Local",
                    "DPFL",
                    "DSpodFL",
                    "DA-DPFL",
                    "pFedDIL",
                    "DCFCL",
                    "dFLMoE",
                    "DeModCL",
                    "Fixed-collab",
                ]:
                    act = 0
                elif base == "Serial":
                    ratio = (
                        0.75
                        if "cfg75" in c["serial_rule"]
                        else 0.25
                        if "cfg25" in c["serial_rule"]
                        else 0.5
                    )
                    cfg = (phase < ratio) != (c["serial_rule"] == "collab50_cfg50")
                    act = 1 if cfg and stageM[b, k] % 3 == 0 else 0 if cfg else 2
                elif base in ["Local-NAS-HPO", "PerFedSDE-NAS"]:
                    act = 1 if stageM[b, k] % 4 == 0 else 0
                elif base == "Round-robin":
                    act = int(stageM[b, k] % 3)
                else:
                    lam = float(c["ridge_lambda"])
                    kap = float(c["kappa"])
                    mu = gain_sum[b, k] / (gain_n[b, k] + lam)
                    uncertainty = 1 / np.sqrt(gain_n[b, k] + lam)
                    effective_mu = np.zeros(5) if m == "No-gain" else mu
                    fleft = max(
                        1e-10, float(Fcap[b, k] - stageF[b, k] - auditcost[b, 0])
                    )
                    eleft = max(1, int(c["E"]) - int(stageE[b, k]))
                    estimated = (
                        unit[b] * float(work[b, 0]) * np.array([32, 32, 8.8, 0, 0])
                        + evalcost[b, 0]
                        + 0.001
                    )
                    demand = np.maximum(estimated / fleft, 1 / eleft)
                    gap = np.zeros(5)
                    if m == "No-cross-family-parent":
                        gap = np.maximum(
                            0, bestV[b, k] - familyV[b, k, [0, 0, 1, 1, 1]]
                        )
                    priority = np.maximum(effective_mu + kap * uncertainty - gap, 0) / (
                        0.001 + demand
                    )
                    legal = np.array(
                        [
                            m != "No-Refine",
                            True,
                            bool(available[b, k]),
                            bool(weak[b]),
                            bool(s > 1 and not imported[b, k]),
                        ]
                    )
                    priority[~legal] = -1

                    cold = [0, 1, 2, 4]
                    act = (
                        next(
                            (x for x in cold[int(stageM[b, k]) :] if legal[x]),
                            int(np.argmax(priority)),
                        )
                        if stageM[b, k] < 3
                        else int(np.argmax(priority))
                    )
                    selected_mu[b, k] = effective_mu[act]
                    selected_priority[b, k] = priority[act]
                if act == 2 and not available[b, k]:
                    act = 0
                if m == "No-Refine" and act == 0:
                    act = 1
                desired = 1 if act in [0, 1] else 2
                if isC[b]:
                    if m == "No-cross-family-parent":
                        q[b, k] = family_best[b, k, desired - 1]
                        H[b, k] = family_hist[b, k, desired - 1]
                        family[b, k] = family_tags[b, k, desired - 1]
                    else:
                        q[b, k] = best[b, k]
                        H[b, k] = bestH[b, k]
                        family[b, k] = bestfamily[b, k]
                block_parent[b, k] = (
                    f"{c['case_id']}:s{s + 1}:c{k + 1}:H{H[b, k]}:f{family[b, k]}"
                )
                block_family[b, k] = family[b, k]
                block_H[b, k] = H[b, k]
                block_before[b, k] = (
                    q[b, k, domains[b, k]]
                    + vbias[b, k, domains[b, k]]
                    + float(np.dot(vweights[b, k], q[b, k] - source[b, k]))
                )
                parent_entry[b, k] = bool(
                    H[b, k] == entryH[b, k] and np.array_equal(q[b, k], qentry[b, k])
                )
                import_H[b, k] = -1
                import_family[b, k] = -1
                block_startF[b, k] = totalF[b, k]
                block_steps[b, k] = 0
                action[b, k] = act
                remaining[b, k] = min(
                    16 if act < 2 else 8 if act == 2 else 1, L - (r - 1) % L
                )
                stageM[b, k] += 1
                attempts[b, k] += 1
                attempt_counts_by_action[b, act] += 1
                if act == 1:
                    q[b, k, domains[b, k]] = (
                        0.90 * q[b, k, domains[b, k]]
                        + 0.10 * source[b, k, domains[b, k]]
                    )
                if act == 4:
                    q[b, k] = bank_q[s - 2][b, k]
                    H[b, k] = bank_H[s - 2][b, k]
                    family[b, k] = bank_family[s - 2][b, k]
                    import_H[b, k] = H[b, k]
                    import_family[b, k] = family[b, k]
                    imported[b, k] = True
                if act == 3:
                    q[b, k, domains[b, k]] -= 0.006 if not weak[b] else -0.006
                if act == 2:
                    traffic[b, k] += (0.055 + 0.015 * profiles[b]) * (1 + 0.2 * s)
                f = np.zeros(25)
                f[0] = 1
                f[1 if act < 2 else 4 if act == 4 else 2 if act == 3 else 3] = 1
                f[{1: 5, 0: 6, 2: 7, 3: 8, 4: 10}[act]] = 1
                f[12] = block_before[b, k]
                f[13] = float(act == 2) / 12
                f[14] = 0.35 * float(act == 2) / 12
                f[15] = H[b, k] / max(R * 2 * 8, 1)
                f[16] = remaining[b, k] * 2 * math.ceil(dn[b] / 8) / 256
                f[17] = (
                    (0.055 + 0.015 * profiles[b]) * 1048576 / (32 * 1048576)
                    if act == 2
                    else 0
                )
                f[18] = float(act == 2)
                f[19] = (2 + int(c["z"]) % 2) / 2
                f[20] = 1.0
                f[21] = max(0, 1 - stageF[b, k] / Fcap[b, k])
                f[22] = (int(c["E"]) - stageE[b, k]) / 65
                f[23] = max(0, 1 - traffic[b, k] / 32)
                f[24] = f[21]
                features[b, k] = f
                controller = 0.001 * (1 + np.dot(f, f) * 0.01)
                totalF[b, k] += controller
                stageF[b, k] += controller
        active_flags = (remaining > 0) & (action < 3)
        if any(m == "DSpodFL" for m in full):
            for b, m in enumerate(full):
                if m == "DSpodFL":
                    active_flags[b] &= uniforms[b, r - 1, :, 2] < 0.72
        fitcost = 2 * unit[:, None] * work * np.where(action == 2, 0.55, 1.0)
        active_flags &= stageF + fitcost + evalcost + auditcost <= Fcap
        cur = current().copy()
        targ = target[bi, ci, domains].copy()
        coupled = collab * gamma[:, None] * np.maximum(peer_skill - cur, 0) * available
        if np.any(weak):
            coupled[weak] -= 0.010 * collab[weak]
        dynamic_target = targ + coupled * 0.25
        learning = rate * np.where(action == 2, 0.75, 1.0)
        receiver_rate = np.array([float(c["eta_beta"]) / 0.01 for c in cases])[:, None]
        learning *= np.where(action == 2, np.clip(receiver_rate, 0.1, 4), 1.0)
        for b, m in enumerate(methods):
            if m == "Discrete":
                dynamic_target[b] = targ[b] + np.where(coupled[b] > 0.08, 0.08, 0.0)
        learning *= np.where(action == 1, 0.85, 1.0)
        if any(m == "DA-DPFL" for m in full):
            for b, m in enumerate(full):
                if m == "DA-DPFL":
                    learning[b] *= 1 - 0.18 * s / 3
        gain = learning * (dynamic_target - cur) + 0.00075 * env * (
            0.6 + 0.4 * hetero[:, None]
        )
        overfit = 0.00004 * np.maximum(eta - 1, 0) * (1 - fraction[:, None]) * phase
        gain -= overfit
        q[bi, ci, domains] = np.where(
            active_flags, np.clip(cur + gain, 0.03, 0.96), cur
        )
        for d in range(4):
            seen_old = (
                np.any(order[:, :, :s] == d, axis=2) if s else np.zeros((B, 4), bool)
            )
            decay = forget * np.maximum(q[:, :, d] - 0.10, 0) * active_flags * seen_old
            q[:, :, d] -= decay
        q = np.clip(q, 0.02, 0.97)
        epochs += 2 * active_flags
        newsteps = 2 * np.ceil(dn[:, None] / 8).astype(int) * active_flags
        steps += newsteps
        H += newsteps
        block_steps += newsteps
        totalF += fitcost * active_flags
        stageF += fitcost * active_flags
        remaining = np.maximum(0, remaining - 1)
        finish = (remaining == 0) & (stageM > 0) & (block_steps >= 0)

        finish &= block_parent != ""
        for b, c in enumerate(cases):
            for k in np.flatnonzero(finish[b]):
                act = int(action[b, k])
                score = (
                    q[b, k, domains[b, k]]
                    + vbias[b, k, domains[b, k]]
                    + float(np.dot(vweights[b, k], q[b, k] - source[b, k]))
                )
                totalF[b, k] += evalcost[b, 0]
                stageF[b, k] += evalcost[b, 0]
                totalE[b, k] += 1
                stageE[b, k] += 1
                if act != 4:
                    family[b, k] = 1 if act < 2 else 2
                fam = max(1, int(family[b, k]))
                if score >= familyV[b, k, fam - 1]:
                    family_best[b, k, fam - 1] = q[b, k]
                    familyV[b, k, fam - 1] = score
                    family_hist[b, k, fam - 1] = H[b, k]
                    family_tags[b, k, fam - 1] = family[b, k]
                if score >= bestV[b, k]:
                    best[b, k] = q[b, k]
                    bestV[b, k] = score
                    bestH[b, k] = H[b, k]
                    bestfamily[b, k] = family[b, k]
                predicted = (
                    float(selected_mu[b, k])
                    if isC[b]
                    else max(0, float(target[b, k, domains[b, k]] - block_before[b, k]))
                    * float(rate[b, 0])
                    * 16
                )
                gain_sum[b, k, act] += score - block_before[b, k]
                gain_n[b, k, act] += 1
                events.append(
                    dict(
                        run_id=c["case_id"],
                        round=r,
                        stage=s + 1,
                        client=k + 1,
                        action=["Refine", "Fresh", "Collaborate", "Remove", "Import"][
                            act
                        ],
                        attempt=int(stageM[b, k]),
                        M=int(c["M"]),
                        parent_id=str(block_parent[b, k]),
                        parent_family=int(block_family[b, k]),
                        child_family=int(family[b, k]),
                        parent_is_stage_entry=bool(parent_entry[b, k]),
                        import_target_H=None if act != 4 else int(import_H[b, k]),
                        import_target_family=None
                        if act != 4
                        else int(import_family[b, k]),
                        lineage_before=int(block_H[b, k]),
                        lineage_after=int(H[b, k]),
                        optimizer_updates=int(block_steps[b, k]),
                        selection_before=float(block_before[b, k]),
                        selection_after=float(score),
                        modeled_gain=float(score - block_before[b, k]),
                        predicted_gain=predicted,
                        priority=float(selected_priority[b, k]) if isC[b] else None,
                        remaining_F=max(0, float(Fcap[b, k] - stageF[b, k])),
                        remaining_E=max(0, int(c["E"]) - int(stageE[b, k])),
                        work_GFLOP=float(totalF[b, k] - block_startF[b, k]),
                        features25=json.dumps(
                            features[b, k].round(8).tolist(), separators=(",", ":")
                        ),
                        event_kind="classroom_revision",
                        status="completed",
                    )
                )
                block_parent[b, k] = ""
        if r % L == 0:
            before = q.copy()
            candidate_audit = ascore(best, domains)
            entry_audit = ascore(qentry, domains)
            accept = candidate_audit >= entry_audit - 0.003
            q = np.where(accept[:, :, None], best, qentry)
            H = np.where(accept, bestH, entryH)
            family = np.where(accept, bestfamily, entryfamily)
            totalF += auditcost
            stageF += auditcost
            a = ap_links(q)[0]
            history[:, s] = a
            for b, c in enumerate(cases):
                for k in range(4):
                    acquired = order[b, k, : s + 1]
                    max_seen[b, k, acquired] = np.maximum(
                        max_seen[b, k, acquired], a[b, k, acquired]
                    )
                    diagonal[b, k, domains[b, k]] = a[b, k, domains[b, k]]
                    audit_rows.append(
                        dict(
                            run_id=c["case_id"],
                            stage=s + 1,
                            client=k + 1,
                            round=r,
                            accepted=bool(accept[b, k]),
                            audit_candidate=float(candidate_audit[b, k]),
                            audit_entry=float(entry_audit[b, k]),
                            released_current_AP=float(a[b, k, domains[b, k]]),
                            audit_rule="scenario validation/audit readout, not measured detector AP",
                        )
                    )
            assert np.all(stageM <= np.array([c["M"] for c in cases])[:, None])
            assert np.all(stageE <= np.array([c["E"] for c in cases])[:, None])
            assert np.all(stageF <= Fcap + 1e-7)
            released = q.copy()
            bank_q.append(q.copy())
            bank_H.append(H.copy())
            bank_family.append(family.copy())
            remaining[:] = 0
        record(r, s)
    assert (
        np.isfinite(vals).all()
        and np.all(vals[:, :, [0, 1, 2, 5, 6, 7, 8, 9, 10]] >= 0)
        and np.all(vals[:, :, [0, 1, 2, 5, 6, 7, 8, 9, 10]] <= 100)
    )
    assert np.all(np.diff(vals[:, :, 15], axis=1) >= -1e-8) and np.all(
        epochtrace[:, -1] <= R * 2 * 4
    )
    record_rows = []
    indices = []
    curve_relative = f"curves/rounds_{batch_index:03d}.csv.gz"
    event_relative = f"events/regulation_{batch_index:03d}.csv.gz"
    path = DATA / curve_relative
    if write_files:
        path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_id",
        "round",
        "stage",
        *FIELDS,
        "AllDomainAP",
        "model_loss",
        "local_epochs",
        "optimizer_steps",
        "active_clients",
        "state_role",
    ]
    if write_files:
        with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=4) as f:
            f.write(",".join(header) + "\n")
            for b, c in enumerate(cases):
                lines = []
                for r in range(R + 1):
                    s = 0 if r == 0 else min((r - 1) // L, 3)
                    role = (
                        "entry"
                        if r == 0
                        else "stage_release"
                        if r % L == 0
                        else "process_readout"
                    )
                    fields = [
                        c["case_id"],
                        str(r),
                        str(s + 1),
                        *[format(float(x), ".10g") for x in vals[b, r]],
                        format(float(allAP[b, r]), ".10g"),
                        format(float(latentLoss[b, r]), ".10g"),
                        str(int(epochtrace[b, r])),
                        str(int(steptrace[b, r])),
                        str(int(active_trace[b, r])),
                        role,
                    ]
                    lines.append(",".join(fields))
                f.write("\n".join(lines) + "\n")
        digest = sha(path)
        eventpath = DATA / event_relative
        eventpath.parent.mkdir(exist_ok=True)
        with gzip.open(
            eventpath, "wt", encoding="utf-8", newline="", compresslevel=4
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(events[0]))
            w.writeheader()
            w.writerows(events)
    else:
        digest = None
    for b, c in enumerate(cases):
        ix = dict(
            run_id=c["case_id"],
            path=curve_relative,
            rows=R + 1,
            round_min=0,
            round_max=R,
            row_start=b * (R + 1),
            row_end=(b + 1) * (R + 1) - 1,
            sha256=digest,
        )
        indices.append(ix)
        record_rows.append(
            dict(
                run_id=c["case_id"],
                setting=c,
                metrics=quantized(vals[b, -1]),
                execution_kind="classroom_scenario",
                scenario_revision=REV,
                native_YOLO=False,
                physical_measured=False,
                curve_index=ix,
                regulation_path=event_relative,
                modeled_optimizer_steps=int(steptrace[b, -1]),
                modeled_local_epochs=int(epochtrace[b, -1]),
                domain_order=order[b].tolist(),
                history_AP=history[b].tolist(),
                action_counts=dict(
                    zip(
                        ["Refine", "Fresh", "Collaborate", "Remove", "Import"],
                        map(int, attempt_counts_by_action[b]),
                    )
                ),
                report_T_GFLOP=float(
                    (R + 1) * 4 * 4 * 256 * (0.22 + 0.075 * profiles[b]) / 64 / 6
                ),
                checkpoint=None,
                prediction=None,
                device_trace=None,
            )
        )
    return record_rows, indices, audit_rows, events, vals, allAP


def generate_task(task):
    batch, part = task
    rr, ii, aa, ee, v, allap = generate_batch(part, batch)
    curves = []
    for k, c in enumerate(part):
        if c["experiment"] == "N01":
            curves.append((c, v[k, :, :3].copy(), allap[k].copy()))
    return rr, ii, aa, curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    cases = json.loads(SOURCE.read_bytes())
    cases = cases[: a.limit] if a.limit else cases
    config = dict(
        scenario_revision=REV,
        source_settings_sha256=sha(SOURCE),
        generated_by="latent-state recurrence, not neural-network training or detector predictions",
        method_parameters={
            k: dict(
                zip(
                    [
                        "learning_rate",
                        "historical_decay",
                        "peer_response_strength",
                        "capacity_increment",
                        "work_multiplier",
                    ],
                    v,
                )
            )
            for k, v in METHODS.items()
        },
        role_counts_assumed={"D": 64, "V": 128, "A": 128, "T": 256},
        metric_readout="shared sigmoid detection/localization readout averaged over ten IoU thresholds",
        controller_model="Categorical one-hot ridge of observed latent selection gains, not a native 25-dimensional learned model",
        control_parameters="ridge_lambda shrinks gain estimates/exploration; eta_beta scales receiving updates; no parameter-dependent random seeds",
        randomization="SHA256 named task/seed/base-method streams; matched interventions share the same aptitude and innovations; no requested rank or copied AP inputs",
    )
    lock = dict(
        scenario_revision=REV,
        generator_sha256=sha(Path(__file__)),
        settings_sha256=sha(SOURCE),
        selected_cases=len(cases),
    )
    lock_path = DATA / "generation_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_bytes()) != lock:
        raise RuntimeError(
            "Generator or case set differs from the frozen scenario lock"
        )
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    frozen = DATA / "generation_config.json"
    if frozen.exists() and json.loads(frozen.read_bytes()) != config:
        raise RuntimeError("Refusing conflicting scenario configuration")
    frozen.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA / "study_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    groups = defaultdict(list)
    for c in cases:
        groups[int(c["round_budget"])].append(c)
    records = []
    indices = []
    audits = []
    main_curve = []
    start = time.monotonic()
    batch = 0
    tasks = []
    for R, items in groups.items():
        for j in range(0, len(items), a.batch_size):
            tasks.append((batch, items[j : j + a.batch_size]))
            batch += 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(generate_task, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            rr, ii, aa, curves = future.result()
            records.extend(rr)
            indices.extend(ii)
            audits.extend(aa)
            for c, values, allap in curves:
                for r in range(len(values)):
                    main_curve.append(
                        dict(
                            run_id=c["case_id"],
                            dataset=c["dataset"],
                            profile=c["profile"],
                            method=norm(c["method"]),
                            seed=c["seed"],
                            round=r,
                            SeenAP=float(values[r, 0]),
                            CurrentAP=float(values[r, 1]),
                            AllDomainAP=float(allap[r]),
                        )
                    )
            print(
                json.dumps(
                    dict(
                        generated=len(records),
                        total=len(cases),
                        elapsed_s=round(time.monotonic() - start, 1),
                    )
                ),
                flush=True,
            )
    records.sort(key=lambda r: int(r["run_id"].split("-")[-1]))
    (DATA / "results.json").write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    write(DATA / "curve_index.csv", indices)
    write(DATA / "release_audits.csv", audits)
    sums = defaultdict(list)
    for x in main_curve:
        sums[x["dataset"], x["profile"], x["method"], x["round"]].append(x)
    summary = []
    for (ds, pr, m, r), g in sorted(sums.items()):
        row = dict(dataset=ds, profile=pr, method=m, round=r, n=len(g))
        for key in ["SeenAP", "CurrentAP", "AllDomainAP"]:
            d = np.array([x[key] for x in g])
            row[key + "_mean"] = float(d.mean())
            row[key + "_SD"] = float(d.std(ddof=1))
        summary.append(row)
    write(DATA / "round_summary.csv", summary)
    costs = [
        dict(
            run_id=r["run_id"],
            online_GFLOP=r["metrics"][15],
            report_T_GFLOP=r["report_T_GFLOP"],
            energy_J=r["metrics"][11],
            time_s=r["metrics"][18],
            communication_MiB=r["metrics"][17],
            local_epochs=r["modeled_local_epochs"],
            optimizer_steps=r["modeled_optimizer_steps"],
            basis="scenario resource model",
        )
        for r in records
    ]
    write(DATA / "costs.csv", costs)
    mapping = []
    for m, params in METHODS.items():
        mapping.append(
            dict(
                method=m,
                kind="rule_based_classroom_strategy",
                definition=str(config["method_parameters"][m]),
                official_algorithm_reproduction=False,
            )
        )
    for m in sorted(ABLATIONS):
        mapping.append(
            dict(
                method=m,
                kind="classroom_intervention",
                definition="Declared intervention to latent-state revision or scoring; not native ablation evidence",
                official_algorithm_reproduction=False,
            )
        )
    write(DATA / "method_mapping.csv", mapping)
    metadata = dict(
        scenario_revision=REV,
        metric_fields=FIELDS,
        rounds=1000,
        local_epochs_per_round=2,
        role_counts=config["role_counts_assumed"],
        scenario_generation=config["generated_by"],
        assumptions=config,
        independent_seed_count=5,
        source_identity="one newly generated classroom scenario; old executed feature-model results unchanged",
        run_count=len(records),
        round_budget_counts=dict(
            Counter(r["setting"]["round_budget"] for r in records)
        ),
        total_curve_rows=sum(r["setting"]["round_budget"] + 1 for r in records),
        main_runs=sum(r["setting"]["experiment"] == "N01" for r in records),
        status="GENERATED_CLASSROOM_DATA_NOT_TRAINING",
        native_YOLO=False,
        physical_measured=False,
        original_training_rerun=False,
        original_results_replaced=False,
        hashes="Only checksums of files actually written; no trained-weight or device-evidence hashes",
        intended_use="classroom analysis and scientific-writing exercise; not empirical SOTA or mechanism proof",
    )
    (DATA / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                status="COMPLETE_CLASSROOM_GENERATION",
                runs=len(records),
                curve_rows=metadata["total_curve_rows"],
                elapsed_s=round(time.monotonic() - start, 1),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
