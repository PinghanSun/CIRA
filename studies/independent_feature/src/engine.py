"""A learned feature-space detector and closed-loop teaching controller."""

from pathlib import Path
import numpy as np
import hashlib, json, gzip
from functools import lru_cache

SCOPE = "CLOSED_LOOP_FEATURE_DETECTOR_SIMULATION_NOT_NATIVE_YOLO"
FIELDS = [
    "AllDomainAP",
    "CurrentAP",
    "MinClientAP",
    "Forget_pp",
    "BWT_pp",
    "AP50",
    "AP75",
    "Domain0_AP",
    "Domain1_AP",
    "Domain2_AP",
    "Domain3_AP",
    "fleet_energy_J",
    "P95_ms",
    "peak_memory_MiB",
    "model_size_MiB",
    "fleet_GFLOP",
    "fleet_V_calls",
    "fleet_endpoint_traffic_MiB",
    "barrier_completion_s",
    "fleet_actions",
]


def seed(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")


def digest(*a):
    h = hashlib.sha256()
    for x in a:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    a = np.exp(x - x.max(axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=4)
def projection(width):
    rng = np.random.default_rng(seed("projection-" + str(width)))
    return (rng.normal(0, 0.36, (13, width - 1))).astype(np.float32)


def observations(
    rng, width, classes, obj, boxes, domain=0, client=0, dataset="SOURCE", noise=0.1
):
    y = np.eye(8, dtype=np.float32)[classes] * obj[..., None]
    latent = np.concatenate([y * 2.2, boxes, obj[..., None]], axis=-1)

    if dataset != "SOURCE":
        strength = (0.14 if dataset == "ControlledFog" else 0.23) * (domain + 1)
        latent = latent.copy()
        latent[..., :8] += strength * np.roll(y, domain + 1, axis=-1)
        latent[..., :8] *= 1 - 0.09 * domain
        latent[..., 8:12] += 0.025 * np.sin(client + domain)
    x = np.tanh(
        latent @ projection(width)
        + rng.normal(0, noise + 0.035 * domain, latent.shape[:-1] + (width - 1,))
    )

    x[..., :8] = latent[..., :8] + rng.normal(
        0, noise + 0.06 * domain, latent[..., :8].shape
    )
    x[..., 8:12] = (
        2 * latent[..., 8:12]
        - 1
        + rng.normal(0, noise * 0.5 + 0.025 * domain, latent[..., 8:12].shape)
    )
    x[..., 12] = 2 * obj - 1 + rng.normal(0, noise + 0.04 * domain, obj.shape)
    return np.concatenate([x, np.ones(x.shape[:-1] + (1,))], axis=-1).astype(np.float32)


@lru_cache(maxsize=160)
def fixture(dataset, width, dsseed, alpha, noise):
    out = {}
    prior_rng = np.random.default_rng(seed(f"{dataset}-{dsseed}-{alpha}-priors"))
    alpha_num = 1000 if alpha == "equal" else float(alpha)
    priors = prior_rng.dirichlet(np.ones(8) * alpha_num, size=(4, 4))
    for role in ["D", "V", "A", "T"]:
        X = np.empty((4, 4, 16, width), np.float32)
        C = np.empty((4, 4, 16), int)
        O = np.empty((4, 4, 16), np.float32)
        Y = np.empty((4, 4, 16, 4), np.float32)
        for c in range(4):
            for d in range(4):
                rng = np.random.default_rng(
                    seed(f"{dataset}-{dsseed}-1.0-{role}-{c}-{d}")
                )
                cl = (
                    rng.choice(8, 16, p=priors[c, d])
                    if role == "D"
                    else np.arange(16) % 8
                )
                obj = np.ones(16, np.float32)
                obj[12:] = 0
                box = rng.uniform(0.12, 0.88, (16, 4)).astype(np.float32)
                X[c, d] = observations(rng, width, cl, obj, box, d, c, dataset, noise)
                C[c, d] = cl
                O[c, d] = obj
                Y[c, d] = box
        out[role] = (X, C, O, Y)
    return out


def predict(X, W):
    """Only observations and model parameters enter this function; no GT argument."""
    logits = np.matmul(X, W)
    return softmax(logits[..., :8]), sigmoid(logits[..., 8]), sigmoid(logits[..., 9:13])


def loss_grad(X, C, O, Y, W):
    pc, po, pb = predict(X, W)
    n = X.shape[-2]
    one = np.eye(8, dtype=np.float32)[C]
    dc = (pc - one) * O[..., None] / n
    do = (po - O) / n
    db = 8 * (pb - Y) * pb * (1 - pb) * O[..., None] / n
    delta = np.concatenate([dc, do[..., None], db], axis=-1)
    grad = np.matmul(np.swapaxes(X, -1, -2), delta)
    ce = (
        -np.log(np.maximum(np.take_along_axis(pc, C[..., None], axis=-1)[..., 0], 1e-7))
        * O
    )
    bce = -(
        O * np.log(np.maximum(po, 1e-7)) + (1 - O) * np.log(np.maximum(1 - po, 1e-7))
    )
    mse = 4 * ((pb - Y) ** 2).sum(axis=-1) * O
    return (ce + bce + mse).mean(axis=-1), grad.astype(np.float32)


def masked_loss_grad(X, C, O, Y, W, mask):
    pc, po, pb = predict(X, W)
    sw = mask.astype(np.float32)
    den = np.maximum(sw.sum(axis=-1), 1)
    one = np.eye(8, dtype=np.float32)[C]
    dc = (pc - one) * O[..., None] * sw[..., None] / den[..., None, None]
    do = (po - O) * sw / den[..., None]
    db = (
        8
        * (pb - Y)
        * pb
        * (1 - pb)
        * O[..., None]
        * sw[..., None]
        / den[..., None, None]
    )
    delta = np.concatenate([dc, do[..., None], db], axis=-1)
    grad = np.matmul(np.swapaxes(X, -1, -2), delta)
    ce = (
        -np.log(np.maximum(np.take_along_axis(pc, C[..., None], axis=-1)[..., 0], 1e-7))
        * O
    )
    bce = -(
        O * np.log(np.maximum(po, 1e-7)) + (1 - O) * np.log(np.maximum(1 - po, 1e-7))
    )
    mse = 4 * ((pb - Y) ** 2).sum(axis=-1) * O
    return ((ce + bce + mse) * sw).sum(axis=-1) / den, grad.astype(np.float32)


def xyxy(y):
    center = 0.1 + 0.8 * y[..., :2]
    wh = 0.06 + 0.3 * y[..., 2:]
    return np.concatenate([center - wh / 2, center + wh / 2], axis=-1)


def AP(X, C, O, Y, W):
    pc, po, pb = predict(X, W)
    cl = pc.argmax(axis=-1)
    score = po * pc.max(axis=-1)
    a = xyxy(pb)
    b = xyxy(Y)
    wh = np.maximum(
        0, np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2])
    )
    inter = wh.prod(axis=-1)
    area = (
        (a[..., 2:] - a[..., :2]).prod(axis=-1)
        + (b[..., 2:] - b[..., :2]).prod(axis=-1)
        - inter
    )
    ious = inter / np.maximum(area, 1e-8)
    classes = np.arange(8)
    active = cl[..., None, :] == classes[:, None]
    scores = np.where(active, score[..., None, :], -1)
    order = np.argsort(-scores, axis=-1, kind="stable")
    gt = (C[..., None, :] == classes[:, None]) & (O[..., None, :] > 0)
    ngt = gt.sum(axis=-1)
    correct = (cl == C) & (O > 0)
    hits = correct[..., None, None, :] & (
        ious[..., None, None, :] >= np.linspace(0.5, 0.95, 10)[:, None]
    )
    hits = (
        np.broadcast_to(hits, active.shape[:-1] + (10, active.shape[-1]))
        & active[..., None, :]
    )
    idx = np.broadcast_to(order[..., None, :], hits.shape)
    tp = np.take_along_axis(hits, idx, axis=-1)
    cum = np.cumsum(tp, axis=-1)
    precision = cum / np.arange(1, tp.shape[-1] + 1)
    envelope = np.maximum.accumulate(precision[..., ::-1], axis=-1)[..., ::-1]
    recall = cum / np.maximum(ngt[..., None, None], 1)
    bins = np.floor(np.minimum(100, 100 * recall) + 1e-8)
    dbins = np.diff(bins, axis=-1, prepend=0)
    first = (cum == 1) & tp
    weights = dbins + first
    aps = (weights * envelope).sum(axis=-1) / 101
    valid = ngt > 0

    mean = (aps * valid[..., None]).sum(axis=-2) / np.maximum(
        valid.sum(axis=-1)[..., None], 1
    )
    return (mean * 100).astype(np.float64), dict(
        class_probability=pc,
        objectness=po,
        bbox=xyxy(pb),
        class_id=cl,
        score=score,
        iou_for_evaluator=ious,
    )


@lru_cache(maxsize=4)
def source_model(width):
    rng = np.random.default_rng(seed("frozen-source-" + str(width)))
    C = np.arange(256) % 8
    O = np.ones(256, np.float32)
    O[192:] = 0
    Y = rng.uniform(0.12, 0.88, (256, 4)).astype(np.float32)
    X = observations(rng, width, C, O, Y)
    W = np.zeros((width, 13), np.float32)
    velocity = np.zeros_like(W)
    for _ in range(320):
        _, g = loss_grad(X, C, O, Y, W)
        velocity = 0.85 * velocity + g
        W -= 0.055 * velocity
    return W


def gather_role(data, domain):
    return tuple(
        np.take_along_axis(
            x,
            domain[..., None, None] if x.ndim == 4 else domain[..., None, None, None],
            axis=2,
        ).squeeze(axis=2)
        for x in data
    )


def masks(z, width):
    ports = ((np.arange(6)[None, None, :] + z[..., None]) % 6) < (2 + (z % 4))[
        ..., None
    ]
    groups = np.minimum(np.arange(width) * 6 // max(width - 1, 1), 5)
    m = ports[..., groups].astype(np.float32)
    m[..., -1] = 1
    return m


def training_mask(cases, domain):
    use = np.ones((len(cases), 4, 16), bool)
    for b, c in enumerate(cases):
        if c["D_fraction"] < 1:
            for k in range(4):
                rr = np.random.default_rng(
                    seed(
                        f"{c['dataset']}-{c['seed']}-client{k}-domain{domain[b, k]}-nested-D"
                    )
                )
                keep = rr.permutation(16)[: max(4, int(16 * c["D_fraction"]))]
                use[b, k] = False
                use[b, k, keep] = True
    return use


def training_prototype(X, use, cases):
    proto = X.mean(axis=2)
    for b, c in enumerate(cases):
        if c["D_fraction"] < 1:
            for k in range(4):
                proto[b, k] = X[b, k, use[b, k]].mean(axis=0)
    return proto


def run_batch(cases, profile, outdir, save_fixtures=True, capture_stages=False):
    B = len(cases)
    D = profile["width"]
    R = int(cases[0]["round_budget"])
    assert all(c["round_budget"] == R for c in cases)
    rngs = [
        np.random.default_rng(seed(f"{c['dataset']}-{c['profile']}-{c['seed']}-policy"))
        for c in cases
    ]
    data = {
        role: tuple(
            np.stack(
                [
                    fixture(
                        c["dataset"],
                        D,
                        c["seed"],
                        str(c["alpha"]),
                        profile["projection_noise"],
                    )[role][j]
                    for c in cases
                ]
            )
            for j in range(4)
        )
        for role in ["D", "V", "A", "T"]
    }
    order = np.empty((B, 4, 4), int)
    for b, c in enumerate(cases):
        rr = np.random.default_rng(seed(f"{c['dataset']}-{c['seed']}-order"))
        common = rr.permutation(4)
        for k in range(4):
            order[b, k] = common if "common" in c["order_mode"] else rr.permutation(4)
        if save_fixtures:
            key = f"{c['dataset']}_{c['profile']}_S{c['seed']}_a{c['alpha']}"
            p = outdir / "fixtures" / f"{key}.npz"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                ff = fixture(
                    c["dataset"],
                    D,
                    c["seed"],
                    str(c["alpha"]),
                    profile["projection_noise"],
                )
                np.savez_compressed(
                    p,
                    **{
                        role + "_" + n: ff[role][j]
                        for role in ff
                        for j, n in enumerate(["X", "class", "object", "box"])
                    },
                )
    base = source_model(D)
    W = np.zeros((B, 4, D, 13), np.float32)
    velocity = np.zeros_like(W)
    external = np.zeros_like(W)
    beta = np.zeros((B, 4), np.float64)
    sparse = np.ones((B, 4, D), np.float32)
    z = np.array([[c["z"]] * 4 for c in cases])
    h = np.array([[c["h"]] * 4 for c in cases])
    port = masks(z, D) * sparse
    lr = np.array([0.02, 0.035, 0.05, 0.065, 0.085, 0.11])[h % 6]
    methods = np.array([c["method"].split("（")[0] for c in cases])
    cond = np.array([c["condition"] for c in cases])
    isjoint = np.isin(
        methods,
        [
            "CIRA",
            "NAS-only",
            "HPO-only",
            "No-Refine",
            "No-cross-family-parent",
            "Discrete",
            "No-gain",
        ],
    )

    islocal = np.isin(methods, ["Local", "Local-NAS-HPO"])
    lam = np.array([c["ridge_lambda"] for c in cases])[:, None, None, None]
    inv = np.broadcast_to(np.eye(25), (B, 4, 25, 25)).copy() / lam
    coef = np.zeros((B, 4, 25))
    targetsum = np.zeros_like(coef)
    epoch = np.zeros((B, 4), int)
    steps = np.zeros_like(epoch)
    F = np.zeros((B, 4))
    energy = np.zeros_like(F)
    traffic = np.zeros_like(F)
    clock = np.zeros_like(F)
    temperature = np.full_like(F, 38.0)
    Vcalls = np.zeros_like(epoch)
    actions = np.zeros_like(epoch)
    budgets = np.array([c["F_scale"] for c in cases])[:, None] * R * 2 * 16 * D * 13 * 6
    beta_rate = np.array([c["eta_beta"] for c in cases])[:, None]
    lastgain = np.zeros((B, 4))
    action = np.zeros((B, 4), int)
    phase = np.zeros((B, 4), int)
    released = W.copy()
    peerproto = np.zeros((B, 4, D))
    archive = W.copy()
    archive_v = velocity.copy()
    archive_beta = beta.copy()
    archive_ext = external.copy()
    archive_z = z.copy()
    archive_h = h.copy()
    archive_score = np.full((B, 4), -np.inf)
    values = np.zeros((B, R + 1, 20))
    evalmask = np.zeros((B, R + 1), bool)
    losses = np.zeros((B, R + 1, 4))
    epochcurve = np.zeros((B, R + 1, 4), int)
    fcurve = np.zeros((B, R + 1, 4))
    history = np.full((B, 4, 4, 4), np.nan)
    diagonal = np.full((B, 4, 4), np.nan)
    bestdomain = np.full((B, 4, 4), -np.inf)
    logs = [[] for _ in cases]
    snapshots = [[] for _ in cases]
    predictions = [[] for _ in cases]
    start = None
    before = None
    feat = None
    parentids = None
    decision_no = 0
    stageN = R // 4

    def effective():
        coupling = np.array([c["gamma_int"] for c in cases])[:, None] * (
            0.75 + 0.5 * port.mean(axis=-1)
        )
        return base + W + beta[..., None, None] * external * coupling[..., None, None]

    def charge(ops, bytes_=0):
        nonlocal F, energy, clock, temperature, traffic
        op = np.asarray(ops)
        trans = np.asarray(bytes_)
        bandwidth = np.array(
            [1e6 if "payload" in c["contact_mode"] else 2e6 for c in cases]
        )[:, None]
        seconds = (
            op
            / profile["throughput_ops_s"]
            * (1 + 0.02 * np.maximum(temperature - 45, 0))
            + trans / bandwidth
        )

        power = profile["base_power_W"] + 2.5 * (op > 0) + 0.7 * (trans > 0)
        F += op
        traffic += trans
        clock += seconds
        energy += seconds * power
        temperature = np.maximum(
            32,
            temperature + 0.035 * seconds * power - 0.12 * seconds * (temperature - 32),
        )

    def evaluate_all(round_, stage):
        ap, pred = AP(*data["T"], effective()[:, :, None])
        macro = ap.mean(axis=(1, 2, 3))
        current = ap[np.arange(B)[:, None], np.arange(4)[None, :], domain].mean(
            axis=(1, 2)
        )
        domainAP = ap.mean(axis=(1, 3))
        clientAP = ap.mean(axis=(2, 3))
        val = values[:, round_]
        val[:, 0] = macro
        val[:, 1] = current
        val[:, 2] = clientAP.min(axis=1)
        val[:, 5] = ap[..., 0].mean(axis=(1, 2))
        val[:, 6] = ap[..., 5].mean(axis=(1, 2))
        val[:, 7:11] = domainAP
        if stage > 0:
            old = order[:, :, :stage]
            now = np.take_along_axis(ap.mean(axis=-1), old, axis=2)
            diag = np.take_along_axis(diagonal, old, axis=2)
            best = np.take_along_axis(bestdomain, old, axis=2)
            val[:, 3] = np.maximum(best - now, 0).mean(axis=(1, 2))
            val[:, 4] = (now - diag).mean(axis=(1, 2))
        complexity = port.sum(axis=-1) * 13
        ms = (
            0.7
            + complexity * 16 / 1e4
            + beta * 0.25
            + 0.01 * np.maximum(temperature - 40, 0)
        )
        val[:, 11] = energy.sum(axis=1)
        val[:, 12] = ms.max(axis=1) * 1.15
        val[:, 13] = 20 + complexity.max(axis=1) * 0.012
        val[:, 14] = (
            (base.size + np.count_nonzero(W, axis=(-2, -1)).max(axis=1)) * 4 / 1048576
        )
        val[:, 15] = F.sum(axis=1) / 1e9
        val[:, 16] = Vcalls.sum(axis=1)
        val[:, 17] = traffic.sum(axis=1) / 1048576
        val[:, 18] = clock.max(axis=1)
        val[:, 19] = actions.sum(axis=1)
        evalmask[:, round_] = True
        return ap, pred

    role = entry = entryA = use = None
    for r in range(R + 1):
        stage = min(r // stageN, 3)
        boundary = r % stageN == 0
        domain = order[:, :, stage]

        rngs = [
            np.random.default_rng(
                seed(f"{c['dataset']}-{c['profile']}-{c['seed']}-r{r}-policy")
            )
            for c in cases
        ]
        if r == R:
            domain = order[:, :, 3]
            boundary = False
        if r % 16 == 0 or boundary or r == R:
            if r and start is not None:
                after = AP(*role["V"], effective())[0].mean(axis=-1)
                gain = (after - before) / 100
                Vcalls += 1
                charge(16 * D * 13 * 2)
                candidate_hash = [
                    [
                        digest(
                            W[bb, kk],
                            velocity[bb, kk],
                            beta[bb, kk],
                            external[bb, kk],
                            z[bb, kk],
                            h[bb, kk],
                        )
                        for kk in range(4)
                    ]
                    for bb in range(B)
                ]
                cached_v = after.copy()

                u = np.matmul(inv, feat[..., None])[..., 0]
                den = 1 + np.sum(feat * u, axis=-1)
                inv -= u[..., None] * u[..., None, :] / den[..., None, None]
                targetsum += feat * gain[..., None]
                coef = np.matmul(inv, targetsum[..., None])[..., 0]
                rollback = (gain < -0.001) & isjoint[:, None]
                W = np.where(rollback[..., None, None], start[0], W)
                velocity = np.where(rollback[..., None, None], start[1], velocity)
                beta = np.where(rollback, start[2], beta)
                external = np.where(rollback[..., None, None], start[3], external)
                z = np.where(rollback, start[4], z)
                h = np.where(rollback, start[5], h)
                port = masks(z, D) * sparse
                lr = np.array([0.02, 0.035, 0.05, 0.065, 0.085, 0.11])[h % 6]
                retained = np.where(rollback, before, after)
                cached_v = retained.copy()
                better = retained > archive_score
                archive = np.where(better[..., None, None], W, archive)
                archive_v = np.where(better[..., None, None], velocity, archive_v)
                archive_beta = np.where(better, beta, archive_beta)
                archive_ext = np.where(better[..., None, None], external, archive_ext)
                archive_z = np.where(better, z, archive_z)
                archive_h = np.where(better, h, archive_h)
                archive_score = np.maximum(archive_score, retained)
                lastgain = gain
                for b, c in enumerate(cases):
                    for k in range(4):
                        logs[b][-4 + k].update(
                            V_after=float(after[b, k]),
                            signed_V_gain=float(gain[b, k]),
                            retained_V=float(retained[b, k]),
                            rollback=bool(rollback[b, k]),
                            child_hash=candidate_hash[b][k],
                            retained_hash=digest(
                                W[b, k],
                                velocity[b, k],
                                beta[b, k],
                                external[b, k],
                                z[b, k],
                                h[b, k],
                            ),
                            after_ops=float(F[b, k]),
                            after_epochs=int(epoch[b, k]),
                        )
            if r and (boundary or r == R):
                newA = AP(*role["A"], effective())[0].mean(axis=-1)
                reject = newA < entryA - 0.1
                W = np.where(reject[..., None, None], entry[0], W)
                velocity = np.where(reject[..., None, None], entry[1], velocity)
                beta = np.where(reject, entry[2], beta)
                external = np.where(reject[..., None, None], entry[3], external)
                z = np.where(reject, entry[4], z)
                h = np.where(reject, entry[5], h)
                port = masks(z, D) * sparse
                lr = np.array([0.02, 0.035, 0.05, 0.065, 0.085, 0.11])[h % 6]
                aa, _ = AP(*data["T"], effective()[:, :, None])
                hist = aa.mean(axis=-1)
                completed_stage = (r - 1) // stageN
                history[:, completed_stage] = hist
                bestdomain = np.maximum(bestdomain, hist)
                for bb in range(B):
                    for kk in range(4):
                        diagonal[bb, kk, order[bb, kk, completed_stage]] = hist[
                            bb, kk, order[bb, kk, completed_stage]
                        ]
                if capture_stages:
                    for bb, cc in enumerate(cases):
                        dest = outdir / "runs" / cc["case_id"]
                        dest.mkdir(parents=True, exist_ok=True)
                        cp = np.array(cc["gamma_int"]) * (
                            0.75 + 0.5 * port[bb].mean(axis=-1)
                        )
                        np.savez_compressed(
                            dest / f"stage_{completed_stage + 1}.npz",
                            frozen_base=base,
                            adapter=W[bb],
                            momentum=velocity[bb],
                            external=external[bb],
                            receiving_weight=beta[bb],
                            active_feature_mask=port[bb],
                            coupling=cp,
                            z=z[bb],
                            h=h[bb],
                            order=order[bb],
                            T_AP=aa[bb],
                            A_entry=entryA[bb],
                            A_candidate=newA[bb],
                            A_accepted=~reject[bb],
                            optimizer_steps=steps[bb],
                            scope=np.array(SCOPE),
                        )
                peerproto = training_prototype(role["D"][0], use, cases)
            if boundary and r < R:
                if r:
                    released = W.copy()
                role = {k: gather_role(data[k], domain) for k in ["D", "V", "A"]}
                entry = (
                    W.copy(),
                    velocity.copy(),
                    beta.copy(),
                    external.copy(),
                    z.copy(),
                    h.copy(),
                )
                entryA = AP(*role["A"], effective())[0].mean(axis=-1)
                stageF = F.copy()
                stageV = Vcalls.copy()
                actionsStage = actions.copy()
                use = training_mask(cases, domain)
                proto = training_prototype(role["D"][0], use, cases)
                aff = np.einsum("bcd,bkd->bck", proto, peerproto) / (
                    np.maximum(
                        np.linalg.norm(proto, axis=-1)[..., None]
                        * np.linalg.norm(peerproto, axis=-1)[:, None, :],
                        1e-8,
                    )
                )
                aff[:, np.arange(4), np.arange(4)] = -1e9
                donor = aff.argmax(axis=-1)
                donorW = np.take_along_axis(released, donor[..., None, None], axis=1)
                available = np.full((B, 4), stage > 0)
                available[islocal] = False
                for b, c in enumerate(cases):
                    if c["donor_mode"] == "empty":
                        available[b] = False
                    if c["donor_mode"] == "weak":
                        donorW[b] *= 0.2
                    visibility_rng = np.random.default_rng(
                        seed(
                            f"{c['dataset']}-{c['profile']}-{c['seed']}-stage{stage}-visibility"
                        )
                    )
                if "half" in c["contact_mode"]:
                    available[b] &= visibility_rng.random(4) < 0.5
                affinity = np.take_along_axis(aff, donor[..., None], axis=-1)[..., 0]
                affinity = np.clip(affinity, -1, 1)

                for b, m in enumerate(methods):
                    if m == "pFedDIL" and stage and snapshots[b]:
                        options = [np.zeros_like(W[b])] + [x["W"] for x in snapshots[b]]
                        scores = []
                        for candidate in options:
                            scores.append(
                                loss_grad(*(x[b] for x in role["V"]), base + candidate)[
                                    0
                                ]
                            )
                        scores = np.stack(scores)
                        best = scores.argmin(axis=0)
                        reuse_margin = 0.02 * cases[b]["proxy_setting"]
                        best = np.where(
                            scores[0] - scores.min(axis=0) > reuse_margin, best, 0
                        )
                        for k in range(4):
                            W[b, k] = options[best[k]][k]
                            velocity[b, k] = (
                                0
                                if best[k] == 0
                                else snapshots[b][best[k] - 1]["velocity"][k]
                            )
                        Vcalls[b] += len(options)
                        extraops = np.zeros((B, 4))
                        extraops[b] = len(options) * 16 * D * 13 * 2
                        charge(extraops)
                    if m == "DA-DPFL":
                        sparse[b] = (
                            np.abs(W[b]).mean(axis=-1)
                            >= np.quantile(
                                np.abs(W[b]).mean(axis=-1),
                                min(0.65, 0.35 * cases[b]["proxy_setting"]),
                                axis=-1,
                                keepdims=True,
                            )
                        ).astype(np.float32)
                        sparse[b, :, -1] = 1
                        W[b] *= sparse[b, ..., None]
                    if (
                        m in ["Fixed-collab", "DPFL", "DCFCL", "DeModCL", "dFLMoE"]
                        and stage
                    ):
                        external[b] = donorW[b]
                        beta[b] = np.where(
                            available[b], min(0.7, 0.3 * cases[b]["proxy_setting"]), 0
                        )
                    if m == "Fixed-collab":
                        external[b] = released[b, (np.arange(4) + 1) % 4]
                    if m == "DPFL" and stage:
                        external[b] = (released[b].sum(axis=0)[None] - released[b]) / 3
                    if m == "dFLMoE" and stage:
                        weights = softmax(aff[b] / (0.35 * cases[b]["proxy_setting"]))
                        external[b] = np.einsum("ck,kdi->cdi", weights, released[b])
                        beta[b] = np.where(available[b], 0.4, 0)
                    if m == "DeModCL" and stage:
                        for k in range(4):
                            for indices in np.array_split(np.arange(D), 6):
                                candidates = [
                                    float(np.linalg.norm(released[b, j, indices]))
                                    if j != k
                                    else -1
                                    for j in range(4)
                                ]
                                external[b, k, indices] = released[
                                    b, int(np.argmax(candidates)), indices
                                ]
                    if m == "DCFCL":
                        beta[b] *= affinity[b] > 0.05 * cases[b]["proxy_setting"]
                    if cases[b]["donor_mode"] == "weak" and m in [
                        "Fixed-collab",
                        "DPFL",
                        "DCFCL",
                        "DeModCL",
                        "dFLMoE",
                    ]:
                        external[b] *= 0.2
                peerproto = proto.copy()
                archive_score[:] = -np.inf
            if boundary or r == R or r % 64 == 0:
                ap, pred = evaluate_all(r, stage)
            if boundary or r == R:
                for b, c in enumerate(cases):
                    snapshots[b].append(
                        dict(
                            stage=stage + 1,
                            round=r,
                            W=W[b].copy(),
                            velocity=velocity[b].copy(),
                            beta=beta[b].copy(),
                        )
                    )
                    if r == R:
                        predictions[b] = pred
            if r == R:
                break
            if boundary or r == 0:
                cached_v = AP(*role["V"], effective())[0].mean(axis=-1)
                Vcalls += 1
                charge(16 * D * 13 * 2)
            before = cached_v.copy()
            vloss, _ = loss_grad(*role["V"], effective())
            charge(np.full((B, 4), 8 * 25 * 25))
            features = np.zeros((B, 4, 5, 25))
            features[..., 0] = 1
            features[..., 1:6] = np.eye(5)
            extras = np.stack(
                [
                    np.full((B, 4), (stage + 1) / 4),
                    port.mean(axis=-1),
                    lr / 0.12,
                    beta,
                    available.astype(float),
                    affinity,
                    np.full((B, 4), r / R),
                    np.maximum(0, 1 - (F - stageF) / np.maximum(budgets / 4, 1)),
                    np.maximum(0, 1 - (Vcalls - stageV) / 65),
                    lastgain,
                    np.ones((B, 4)),
                    vloss / 5,
                    np.linalg.norm(W, axis=(-2, -1)) / 10,
                    np.linalg.norm(velocity, axis=(-2, -1)) / 10,
                    np.full((B, 4), stage / 4),
                    np.array(
                        [
                            1 if c["alpha"] == "equal" else float(c["alpha"])
                            for c in cases
                        ]
                    )[:, None]
                    * np.ones((B, 4)),
                    available.astype(float),
                    np.ones((B, 4)),
                    np.array([c["gamma_int"] for c in cases])[:, None]
                    * np.ones((B, 4)),
                ],
                axis=-1,
            )
            features[..., 6:] = extras[:, :, None, :]
            mu = np.einsum("bcad,bcd->bca", features, coef)
            unc = np.sqrt(
                np.maximum(np.einsum("bcai,bcij,bcaj->bca", features, inv, features), 0)
            )
            kap = np.array([c["kappa"] for c in cases])[:, None, None]
            estimated = np.array([1.0, 1.0, 0.5, 0.02, 0.04])
            priorities = (mu + kap * unc) / estimated
            available_now = available.copy()
            for bb, cc in enumerate(cases):
                if (
                    "interrupt" in cc["contact_mode"]
                    and 0.48 <= (r % stageN) / stageN < 0.64
                ):
                    available_now[bb] = False
            priorities[:, :, 2] = np.where(available_now, priorities[:, :, 2], -1e9)
            priorities[:, :, 3] = np.where(beta > 0, priorities[:, :, 3], -1e9)
            priorities[:, :, 4] = np.where(
                archive_score > before + 0.1, priorities[:, :, 4], -1e9
            )
            for bb, cc in enumerate(cases):
                if cc["M"] < 5:
                    priorities[bb, :, int(cc["M"]) :] = -1e9
            action = priorities.argmax(axis=-1)
            for b, c in enumerate(cases):
                m = methods[b]
                co = c["condition"]
                if m == "No-Refine" or co == "No-Refine":
                    action[b] = np.where(action[b] == 0, 1, action[b])
                if m == "No-cross-family-parent" or co == "No-cross-family-parent":
                    action[b] = np.where(action[b] == 4, 0, action[b])
                if m == "No-gain" or co == "No-gain":
                    action[b] = rngs[b].integers(0, 3, 4)
                    action[b] = np.where(available_now[b], action[b], 0)
                if m in [
                    "Local",
                    "Fixed-collab",
                    "DPFL",
                    "pFedDIL",
                    "DA-DPFL",
                    "DeModCL",
                    "DCFCL",
                    "dFLMoE",
                    "DSpodFL",
                ]:
                    action[b] = 0
                if m == "Local-NAS-HPO" or m == "PerFedSDE-NAS":
                    period = (
                        64
                        if m == "Local-NAS-HPO"
                        else max(16, 16 * round(4 * c["proxy_setting"]))
                    )
                    action[b] = np.where(r % period == 0, 1, 0)
                if m == "Round-robin":
                    action[b] = (r // 16) % 3
                    action[b] = np.where(
                        available_now[b], action[b], np.minimum(action[b], 1)
                    )
                if m == "Serial":
                    ratio = (
                        0.75
                        if "cfg75" in c["serial_rule"]
                        else (0.25 if "cfg25" in c["serial_rule"] else 0.5)
                    )
                    first_cfg = "collab50_cfg50" != c["serial_rule"]
                    cfg = ((r % stageN) / stageN < ratio) == first_cfg
                    action[b] = 1 if cfg and r % 64 == 0 else (0 if cfg else 2)
                    action[b] = np.where(available_now[b], action[b], 0)
            start = (
                W.copy(),
                velocity.copy(),
                beta.copy(),
                external.copy(),
                z.copy(),
                h.copy(),
            )
            feat = np.take_along_axis(
                features, action[..., None, None], axis=2
            ).squeeze(axis=2)
            for b, c in enumerate(cases):
                for k in range(4):
                    act = int(action[b, k])
                    parent = digest(
                        W[b, k],
                        velocity[b, k],
                        beta[b, k],
                        external[b, k],
                        z[b, k],
                        h[b, k],
                    )
                    if act == 1:
                        if methods[b] != "HPO-only" and c["condition"] != "HPO-only":
                            z[b, k] = int(rngs[b].integers(0, 42))
                        if methods[b] != "NAS-only" and c["condition"] != "NAS-only":
                            h[b, k] = int(rngs[b].integers(0, 6))
                        W[b, k] = 0
                        velocity[b, k] = 0
                    if act == 2:
                        external[b, k] = donorW[b, k]
                        beta[b, k] = 0.25
                        charge_bytes = np.zeros((B, 4))
                        charge_bytes[b, k] = D * 13 * 4 + 128
                        charge(0, charge_bytes)
                    if act == 3:
                        beta[b, k] = 0
                        external[b, k] = 0
                    if act == 4:
                        W[b, k] = archive[b, k]
                        velocity[b, k] = archive_v[b, k]
                        beta[b, k] = archive_beta[b, k]
                        external[b, k] = archive_ext[b, k]
                        z[b, k] = archive_z[b, k]
                        h[b, k] = archive_h[b, k]
                    logs[b].append(
                        dict(
                            case_id=c["case_id"],
                            client=k + 1,
                            stage=stage + 1,
                            round=r,
                            action=[
                                "Refine",
                                "Fresh",
                                "Collaborate",
                                "Remove",
                                "Import",
                            ][act],
                            parent_hash=parent,
                            features=feat[b, k].tolist(),
                            predicted_mu=float(mu[b, k, act]),
                            priority=float(priorities[b, k, act]),
                            candidate_priorities=priorities[b, k].tolist(),
                            candidate_count=min(5, c["M"]),
                            algorithm_seed_key=f"{c['dataset']}-{c['profile']}-{c['seed']}-r{r}-policy",
                            M_binding=bool(c["M"] < 5),
                            before_ops=float(F[b, k]),
                            before_epochs=int(epoch[b, k]),
                            V_before=float(before[b, k]),
                            member_version=(
                                digest(released[b, donor[b, k]])
                                if stage > 0 and (act == 2 or beta[b, k] > 0)
                                else None
                            ),
                            member_available_stage=(stage if stage > 0 else None),
                            used_T_for_selection=False,
                        )
                    )
            actions += 1
            port = masks(z, D) * sparse
            lr = np.array([0.02, 0.035, 0.05, 0.065, 0.085, 0.11])[h % 6]
            phase[:] = 0
        if r == R:
            break

        reserve = (65 - (Vcalls - stageV)) * (
            16 * D * 13 * 2 + 8 * 25 * 25
        ) + 16 * D * 13 * 4
        next_ops = 2 * np.maximum(6 * 16 * 13 * port.sum(axis=-1), 2 * 16 * 13 * D)
        active = (F - stageF + next_ops + reserve < budgets / 4) & (
            Vcalls - stageV < 65
        )
        local = (action < 2) & active
        receiver = (action == 2) & (phase < 8) & active
        for b, m in enumerate(methods):
            if m == "DSpodFL":
                local[b] &= rngs[b].random(4) < min(
                    0.95, 0.65 * cases[b]["proxy_setting"]
                )
        df = np.array([c["D_fraction"] for c in cases])
        Drole = tuple(x[:, :, :] for x in role["D"])
        for _ in range(2):
            X, C, Ogt, Y = Drole
            loss, g = (
                masked_loss_grad(X, C, Ogt, Y, effective(), use)
                if np.any(df < 1)
                else loss_grad(X, C, Ogt, Y, effective())
            )
            velocity = np.where(local[..., None, None], 0.85 * velocity + g, velocity)
            W -= (
                lr[..., None, None]
                * velocity
                * port[..., None]
                * local[..., None, None]
            )
            if receiver.any():
                _, gv = loss_grad(*role["V"], effective())
                gb = (
                    np.sum(gv * external, axis=(-2, -1))
                    * np.array([c["gamma_int"] for c in cases])[:, None]
                    * (0.75 + 0.5 * port.mean(axis=-1))
                )
                beta = np.clip(beta - beta_rate * gb * receiver, 0, 0.8)
                for b, m in enumerate(methods):
                    if m == "Discrete" or cases[b]["condition"] == "Discrete":
                        beta[b] = np.where(beta[b] > 0.2, 0.35, 0)
            steps += local | receiver
            epoch += local | receiver
            ops = 6 * 16 * 13 * port.sum(axis=-1) * local + 2 * 16 * 13 * D * receiver
            charge(ops)
        phase += 1
        losses[:, r + 1] = loss
        epochcurve[:, r + 1] = epoch
        fcurve[:, r + 1] = F
        values[:, r + 1] = values[:, r]
        values[:, r + 1, 11] = energy.sum(axis=1)
        values[:, r + 1, 15] = F.sum(axis=1) / 1e9
        values[:, r + 1, 16] = Vcalls.sum(axis=1)
        values[:, r + 1, 17] = traffic.sum(axis=1) / 1048576
        values[:, r + 1, 18] = clock.max(axis=1)
        values[:, r + 1, 19] = actions.sum(axis=1)
    out = []
    for b, c in enumerate(cases):
        case_dir = outdir / "runs" / c["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "trajectory.npz",
            round=np.arange(R + 1),
            values=values[b],
            fields=np.array(FIELDS),
            AP_evaluated=evalmask[b],
            loss=losses[b],
            optimizer_epochs=epochcurve[b],
            ops=fcurve[b],
            order=order[b],
            history_AP=history[b],
            scope=np.array(SCOPE),
        )
        np.savez_compressed(
            case_dir / "state.npz",
            frozen_base=base,
            adapter=W[b],
            momentum=velocity[b],
            external=external[b],
            receiving_weight=beta[b],
            active_feature_mask=port[b],
            coupling=np.array(c["gamma_int"]) * (0.75 + 0.5 * port[b].mean(axis=-1)),
            z=z[b],
            h=h[b],
            optimizer_steps=steps[b],
            scope=np.array(SCOPE),
        )

        _, pred = AP(
            *(x[b : b + 1] for x in data["T"]), effective()[b : b + 1, :, None]
        )
        np.savez_compressed(
            case_dir / "prediction.npz",
            class_id=pred["class_id"][0],
            score=pred["score"][0],
            bbox=pred["bbox"][0],
            GT_class=data["T"][1][b],
            GT_object=data["T"][2][b],
            GT_box=xyxy(data["T"][3][b]),
            scope=np.array(SCOPE),
        )
        with gzip.open(case_dir / "decisions.json.gz", "wt", encoding="utf-8") as f:
            json.dump(logs[b], f, ensure_ascii=False, separators=(",", ":"))
        record = dict(
            run_id=c["case_id"],
            matched_case_id=c["matched_case_id"],
            setting=c,
            metrics=values[b, -1].tolist(),
            scope=SCOPE,
            optimizer_steps=int(steps[b].sum()),
            state_hash=digest(base, W[b], velocity[b], external[b], beta[b]),
            trajectory=str((case_dir / "trajectory.npz").relative_to(outdir)).replace(
                "\\", "/"
            ),
            state=str((case_dir / "state.npz").relative_to(outdir)).replace("\\", "/"),
            prediction=str((case_dir / "prediction.npz").relative_to(outdir)).replace(
                "\\", "/"
            ),
            decisions=str((case_dir / "decisions.json.gz").relative_to(outdir)).replace(
                "\\", "/"
            ),
            decision_count=len(logs[b]),
            native_YOLO=False,
            physical_measured=False,
        )
        out.append(record)
    return out
