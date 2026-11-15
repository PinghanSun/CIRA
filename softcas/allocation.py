"""Whole-group allocation with hard image quotas and bounded equal-size swaps."""

from collections import defaultdict
import numpy as np


def hard_group_assignment(groups, targets, seed, quotas, max_passes=8, swap_pairs=512):
    """Cells are domain/role blocks. Every group's cell footprint stays intact."""
    p = np.asarray(targets, float)
    classes, clients = p.shape
    if (
        not groups
        or len({g["id"] for g in groups}) != len(groups)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1)
    ):
        raise ValueError("Unique groups and valid class/client target matrix required")
    if (
        type(seed) is not int
        or type(max_passes) is not int
        or max_passes < 0
        or type(swap_pairs) is not int
        or swap_pairs < 0
    ):
        raise ValueError("Integer seed and nonnegative search limits required")
    cells = sorted(quotas)
    q = np.asarray([quotas[c] for c in cells], float)
    if (
        q.shape != (len(cells), clients)
        or not np.isfinite(q).all()
        or (q < 0).any()
        or (q != np.floor(q)).any()
    ):
        raise ValueError(
            "Nonnegative integer image quotas for every cell/client required"
        )
    footprints = np.array(
        [
            [g.get("cell_images", {"all": g["images"]}).get(c, 0) for c in cells]
            for g in groups
        ],
        int,
    )
    if (
        (footprints < 0).any()
        or (footprints.sum(1) <= 0).any()
        or not np.array_equal(footprints.sum(0), q.sum(1))
    ):
        raise ValueError("Image quotas do not sum to the supplied whole groups")
    values = np.array(
        [
            [
                g.get("cell_classes", {"all": g["class_boxes"]}).get(c, [0] * classes)
                for c in cells
            ]
            for g in groups
        ],
        float,
    )
    if (
        values.shape != (len(groups), len(cells), classes)
        or (values < 0).any()
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid class counts")
    totals = values.sum(0)
    support = totals > 0
    den = np.maximum(totals, 1)
    rng = np.random.Generator(np.random.PCG64(seed))
    order = list(rng.permutation(len(groups)))
    order.sort(key=lambda i: -int(footprints[i].sum()))
    counts = np.zeros((len(cells), clients, classes))
    remaining = q.copy()
    assignment = np.full(len(groups), -1, int)

    def objective(counts):
        errors = (counts / den[:, None, :] - p.T[None, :, :]) ** 2
        return float(
            sum(
                errors[k, :, support[k]].mean()
                for k in range(len(cells))
                if support[k].any()
            )
        )

    fallback = False
    for i in order:
        candidates = []
        for client in range(clients):
            if np.all(footprints[i] <= remaining[:, client]):
                before = (counts[:, client] / den - p[:, client]) ** 2
                after = ((counts[:, client] + values[i]) / den - p[:, client]) ** 2
                delta = sum(
                    float((after[k, support[k]] - before[k, support[k]]).mean())
                    for k in range(len(cells))
                    if support[k].any()
                )
                candidates.append((delta, client))
        if not candidates:
            fallback = True
            break
        client = min(candidates)[1]
        assignment[i] = client
        counts[:, client] += values[i]
        remaining[:, client] -= footprints[i]
    if fallback:
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import lil_matrix

        n = len(groups)
        a = lil_matrix((n + len(cells) * clients, n * clients))
        for i in range(n):
            for client in range(clients):
                a[i, i * clients + client] = 1
                for k in np.flatnonzero(footprints[i]):
                    a[n + k * clients + client, i * clients + client] = footprints[i, k]
        b = np.concatenate((np.ones(n), q.reshape(-1)))
        result = milp(
            rng.uniform(0, 1e-6, n * clients),
            integrality=np.ones(n * clients),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(a.tocsc(), b, b),
            options={"time_limit": 30},
        )
        if result.x is None:
            raise ValueError(
                "Whole-group image quotas infeasible or feasibility limit reached"
            )
        solution = result.x.reshape(n, clients)
        if not np.allclose(solution, np.round(solution), atol=1e-6):
            raise ValueError("No integer feasible allocation")
        assignment = np.argmax(solution, axis=1)
        counts.fill(0)
        for i, client in enumerate(assignment):
            counts[:, client] += values[i]
    initial = objective(counts)
    buckets = defaultdict(list)
    for i in order:
        buckets[tuple(footprints[i])].append(i)
    eligible = [v for v in buckets.values() if len(v) > 1]
    changes = []
    for pass_id in range(max_passes):
        changed = False
        for _ in range(swap_pairs if eligible else 0):
            bucket = eligible[int(rng.integers(len(eligible)))]
            i, j = rng.choice(bucket, 2, replace=False)
            ci, cj = assignment[i], assignment[j]
            if ci == cj:
                continue
            old = objective(counts)
            trial = counts.copy()
            trial[:, ci] += values[j] - values[i]
            trial[:, cj] += values[i] - values[j]
            if objective(trial) < old - 1e-14:
                counts = trial
                assignment[i], assignment[j] = cj, ci
                changed = True
                changes.append(
                    {"pass": pass_id, "left": groups[i]["id"], "right": groups[j]["id"]}
                )
        if not changed:
            break
    achieved = np.zeros_like(q)
    for i, client in enumerate(assignment):
        achieved[:, client] += footprints[i]
    if not np.array_equal(achieved, q):
        raise ValueError("Hard image quota verification failed")
    return {
        "assignment": {g["id"]: int(c) for g, c in zip(groups, assignment)},
        "algorithm": "hard_quota_greedy_equal_footprint_swaps_v1",
        "milp_feasibility_fallback": fallback,
        "quotas": {c: list(map(int, q[k])) for k, c in enumerate(cells)},
        "achieved": {c: list(map(int, achieved[k])) for k, c in enumerate(cells)},
        "initial_class_objective": initial,
        "class_objective": objective(counts),
        "changes": changes,
        "max_passes": max_passes,
        "swap_pairs_per_pass": swap_pairs,
        "seed": seed,
    }


def balanced_quotas(groups, clients=4):
    totals = defaultdict(int)
    for g in groups:
        for cell, n in g.get("cell_images", {"all": g["images"]}).items():
            totals[cell] += n
    return {
        c: [n // clients + int(i < n % clients) for i in range(clients)]
        for c, n in sorted(totals.items())
    }
