"""Backend-independent reference for manuscript Eqs.22--29 and Table S6."""

from dataclasses import dataclass, asdict
import hashlib, json, math
import numpy as np

TYPES = ("configuration", "cached", "acquisition", "import")
MODES = ("Fresh", "Refine", "Add", "Remove", "Replace", "Import", "discrete")
RESOURCES = ("F", "E", "B", "W")
ALLELES = ((0, 1),) * 6 + ((1 / 16, 1 / 8), (1e-4, 3e-4, 1e-3), (1e-4, 1e-3))


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def key(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def named_rng(seed_tuple, operation, attempt=0):
    return np.random.Generator(
        np.random.PCG64(
            int.from_bytes(
                hashlib.sha256(
                    canonical([seed_tuple, operation, attempt]).encode()
                ).digest()[:8],
                "little",
            )
        )
    )


@dataclass(frozen=True)
class Tensor:
    shape: tuple
    dtype: str
    data: bytes

    @classmethod
    def from_array(cls, a):
        a = np.ascontiguousarray(a)
        assert a.dtype.kind in "fiu"
        return cls(a.shape, a.dtype.str, a.tobytes())

    def array(self):
        return np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)

    @property
    def digest(self):
        return key([self.shape, self.dtype, hashlib.sha256(self.data).hexdigest()])


@dataclass(frozen=True)
class Peer:
    port: int
    version: str
    parameters: Tensor

    @property
    def member(self):
        return self.port, self.version


@dataclass(frozen=True)
class State:
    genes: tuple
    phi: Tensor
    peers: tuple = ()
    logits: tuple = ()
    history_steps: int = 0
    family: str = "entry"
    lineage: tuple = ()

    def __post_init__(self):
        assert len(self.genes) == 9 and 1 <= sum(self.genes[:6]) <= 2
        assert all(g in allowed for g, allowed in zip(self.genes, ALLELES))
        members = tuple(sorted(p.member for p in self.peers))
        assert len(members) == len(set(members)) <= 12
        assert all(0 <= p.port < 6 for p in self.peers)
        assert all(sum(p.port == k for p in self.peers) <= 2 for k in range(6))
        assert tuple((k, v) for k, v, b in self.logits) == members
        assert (
            all(math.isfinite(b) for k, v, b in self.logits) and self.history_steps >= 0
        )

    @property
    def digest(self):
        return key(
            [
                self.genes,
                self.phi.digest,
                [
                    (p.port, p.version, p.parameters.digest)
                    for p in sorted(self.peers, key=lambda p: p.member)
                ],
                self.logits,
                self.history_steps,
                self.family,
                self.lineage,
            ]
        )

    def coefficients(self, port):
        b = np.array([0.0] + [b for k, v, b in self.logits if k == port])
        a = np.exp(b - b.max())
        return a / a.sum()


@dataclass(frozen=True)
class Scored:
    state: State
    AP: float
    context: str

    def __post_init__(self):
        assert math.isfinite(self.AP) and 0 <= self.AP <= 1


class Archive:
    def __init__(self, entry, capacity=8):
        assert capacity >= 2
        self.entry = entry
        self.capacity = capacity
        self.context = entry.context
        self.items = {entry.state.digest: entry}

    def best(self):
        return min(self.items.values(), key=lambda s: (-s.AP, s.state.digest))

    def insert(self, scored):
        assert scored.context == self.context, "stale-context parent/evaluation"
        before = self.best().AP
        q = dict(self.items)
        if scored.state.digest in q:
            assert abs(q[scored.state.digest].AP - scored.AP) < 1e-12, (
                "conflicting exact evaluation"
            )
            return False
        q[scored.state.digest] = scored
        champion = min(q.values(), key=lambda s: (-s.AP, s.state.digest))
        protected = {
            self.entry.state.digest: self.entry,
            champion.state.digest: champion,
        }
        for s in sorted(q.values(), key=lambda s: (-s.AP, s.state.digest)):
            if len(protected) >= self.capacity:
                break
            protected.setdefault(s.state.digest, s)
        assert champion.AP >= before
        self.items = protected
        return scored.state.digest in protected

    def tournament(self, rng, family=None):
        eligible = [
            s
            for s in self.items.values()
            if family is None or s.state.family in ["entry", family]
        ]
        eligible.sort(key=lambda s: s.state.digest)
        assert eligible
        chosen = [eligible[int(i)] for i in rng.integers(0, len(eligible), 2)]
        return min(chosen, key=lambda s: (-s.AP, s.state.digest))


def crossover(primary, secondary, rng, mutable=tuple(range(9))):
    assert mutable
    genes = list(primary)
    for i in mutable:
        genes[i] = primary[i] if rng.random() < 0.5 else secondary[i]
    i = int(rng.choice(mutable))
    genes[i] = rng.choice([v for v in ALLELES[i] if v != genes[i]]).item()
    active = np.flatnonzero(genes[:6])
    if len(active) == 0:
        genes[int(rng.integers(0, 6))] = 1
    elif len(active) > 2:
        keep = set(map(int, rng.choice(active, 2, replace=False)))
        genes[:6] = [int(k in keep) for k in range(6)]
    return tuple(genes)


def initialize(parent, mode, genes=None, peers=None, fresh_phi=None, imported=None):
    if mode == "Import":
        assert imported is not None
        return imported
    assert mode in ["Fresh", "Refine", "Add", "Remove", "Replace", "discrete"]
    if mode == "Fresh":
        assert fresh_phi is not None and genes is not None
        return State(
            tuple(genes),
            fresh_phi,
            parent.peers,
            parent.logits,
            parent.history_steps,
            "configuration",
            (parent.digest,),
        )
    if mode == "Refine":
        return State(
            parent.genes,
            parent.phi,
            parent.peers,
            parent.logits,
            parent.history_steps,
            "configuration",
            (parent.digest,),
        )
    peers = tuple(sorted(peers, key=lambda p: p.member))
    old = {(k, v): b for k, v, b in parent.logits}
    logits = tuple((p.port, p.version, old.get(p.member, 0.0)) for p in peers)
    return State(
        parent.genes,
        parent.phi,
        peers,
        logits,
        parent.history_steps,
        "collaboration",
        (parent.digest,),
    )


@dataclass(frozen=True)
class Proposal:
    parent: Scored
    scheduling_type: str
    mode: str
    identity: str
    costs: tuple
    fidelity: int
    advertised_bytes: int = 0
    not_cached: bool = False
    proposed_genes: tuple | None = None
    import_target: State | None = None

    def __post_init__(self):
        assert self.scheduling_type in TYPES and self.mode in MODES
        assert len(self.costs) == 4 and all(
            math.isfinite(c) and c >= 0 for c in self.costs
        )
        assert self.fidelity >= 0 and (self.mode != "Import" or self.fidelity == 0)

    @property
    def tie(self):
        return (
            TYPES.index(self.scheduling_type),
            self.identity,
            self.parent.state.digest,
        )


def features(proposal, entry_AP, remaining, scales):
    assert len(scales) == 13 and all(s > 0 for s in scales)
    p = proposal.parent.state
    genes = proposal.proposed_genes or p.genes
    raw = [
        proposal.parent.AP - entry_AP,
        len(p.peers),
        sum(b for k, v, b in p.logits),
        p.history_steps,
        proposal.fidelity,
        proposal.advertised_bytes,
        int(proposal.not_cached),
        sum(genes[:6]),
        genes[6],
        *remaining,
    ]
    f = np.array(
        [
            1.0,
            *[float(proposal.scheduling_type == t) for t in TYPES],
            *[float(proposal.mode == m) for m in MODES],
            *(np.array(raw) / np.array(scales)),
        ]
    )
    assert f.shape == (25,) and np.isfinite(f).all()
    f.flags.writeable = False
    return f


class Controller:
    def __init__(
        self,
        entry,
        budget,
        scales,
        seed_tuple,
        lam=1.0,
        kappa=0.01,
        epsilon_c=1e-8,
        M=64,
        K=3,
        capacity=8,
        no_gain=False,
        restrict_families=False,
    ):
        assert (
            lam > 0
            and kappa >= 0
            and epsilon_c > 0
            and len(budget) == 4
            and min(budget) >= 0
        )
        self.archive = Archive(entry, capacity)
        self.remaining = np.array(budget, float)
        self.scales = tuple(scales)
        self.seed_tuple = seed_tuple
        self.lam = lam
        self.kappa = kappa
        self.epsilon = epsilon_c
        self.M = M
        self.K = K
        self.no_gain = no_gain
        self.restrict_families = restrict_families
        self.A = np.eye(25) * lam
        self.b = np.zeros(25)
        self.n = 0
        self.log = []
        self.cold_visit = -1

    def affordable(self, p):
        family = (
            "configuration" if p.scheduling_type == "configuration" else "collaboration"
        )
        eligible = not self.restrict_families or p.parent.state.family in [
            "entry",
            family,
        ]
        return (
            eligible
            and p.parent.context == self.archive.context
            and p.parent.state.digest in self.archive.items
            and np.all(np.array(p.costs) <= self.remaining)
        )

    def charge_controller(self, cost):
        cost = np.array(cost, float)
        assert cost.shape == (4,) and np.isfinite(cost).all() and min(cost) >= 0
        self.remaining -= cost

    def rank(self, p, f):
        w = np.linalg.solve(self.A, self.b)
        mu = float(f @ w)
        coverage = math.sqrt(max(0.0, float(f @ np.linalg.solve(self.A, f))))
        gap = self.archive.best().AP - p.parent.AP
        fractions = [
            c / r if r > 0 else 0.0 if c == 0 else math.inf
            for c, r in zip(p.costs, self.remaining)
        ]
        demand = max(fractions)
        numerator = max(
            0.0, (0.0 if self.no_gain else mu) + self.kappa * coverage - gap
        )
        return dict(
            mu=mu,
            used_mu=0.0 if self.no_gain else mu,
            coverage=coverage,
            gap=gap,
            demand=demand,
            priority=numerator / (self.epsilon + demand),
        )

    def bounded(self, pool):
        groups = {t: {} for t in TYPES}
        for p in pool:
            if self.affordable(p):
                old = groups[p.scheduling_type].get(p.identity)
                if old is not None:
                    assert old == p, "same proposal id with conflicting payload"
                groups[p.scheduling_type][p.identity] = p
        out = []
        for t, g in groups.items():
            candidates = sorted(g.values(), key=lambda p: p.tie)
            rng = named_rng(self.seed_tuple, "pool-" + t, self.n)
            out.extend(
                candidates[int(i)] for i in rng.permutation(len(candidates))[: self.K]
            )
        return out

    def choose(self, pool):
        if self.n >= self.M or min(self.remaining) < 0:
            return None
        admitted = self.bounded(pool)
        cold = None
        if self.cold_visit == -1:
            self.cold_visit = 0
            candidates = [
                p for p in admitted if p.mode == "Fresh" and not p.parent.state.peers
            ]
            if candidates:
                cold = min(candidates, key=lambda p: p.tie)
        while cold is None and self.cold_visit < len(TYPES):
            t = TYPES[self.cold_visit]
            self.cold_visit += 1
            candidates = [p for p in admitted if p.scheduling_type == t]
            if candidates:
                cold = min(candidates, key=lambda p: p.tie)
        ranked = []
        for p in admitted:
            f = features(p, self.archive.entry.AP, self.remaining, self.scales)
            r = self.rank(p, f)
            ranked.append((p, f, r))
        chosen = next((x for x in ranked if cold is not None and x[0] == cold), None)
        if chosen is None:
            positive = [x for x in ranked if x[2]["priority"] > 0]
            if not positive:
                return None
            chosen = min(positive, key=lambda x: (-x[2]["priority"], x[0].tie))
        p, f, r = chosen
        saved = f.copy()
        saved.flags.writeable = False
        return p, saved, r

    def commit(
        self, selection, actual_cost, child=None, child_AP=None, status="complete"
    ):
        p, f, r = selection
        actual = np.array(actual_cost, float)
        assert actual.shape == (4,) and np.isfinite(actual).all() and min(actual) >= 0
        self.remaining -= actual
        completed = status == "complete" and min(self.remaining) >= 0
        reason = None
        if completed:
            try:
                assert (
                    child is not None
                    and child_AP is not None
                    and math.isfinite(child_AP)
                    and 0 <= child_AP <= 1
                )
                if p.mode == "Import":
                    assert child == p.import_target
                else:
                    assert (
                        child.history_steps == p.parent.state.history_steps + p.fidelity
                    )
                    if p.mode in ["Add", "Remove", "Replace", "discrete"]:
                        assert child.phi == p.parent.state.phi
                    if p.mode in ["Fresh", "Refine"]:
                        assert child.peers == p.parent.state.peers
                    if p.mode != "Fresh":
                        assert child.genes == p.parent.state.genes
                    elif p.proposed_genes is not None:
                        assert child.genes == p.proposed_genes
                scored = Scored(child, float(child_AP), self.archive.context)
                before = self.archive.best().AP
                retained = self.archive.insert(scored)
                after = self.archive.best().AP
                y = child_AP - p.parent.AP
                assert (
                    abs((after - before) - max(0.0, y - (before - p.parent.AP))) < 1e-12
                )
            except AssertionError:
                completed = False
                reason = "invalid_child_or_archive_transaction"
        if not completed:
            y = 0.0
            retained = False
            child_AP = None
        self.A += np.outer(f, f)
        self.b += f * y
        self.n += 1
        record = dict(
            attempt=self.n,
            parent=p.parent.state.digest,
            proposal=p.identity,
            mode=p.mode,
            status="complete" if completed else "failed",
            failure_reason=reason,
            signed_gain=y,
            child_AP=child_AP,
            archive_retained=retained,
            rollback=not completed,
            negative_completed=completed and y < 0,
            actual_cost=actual.tolist(),
            remaining=self.remaining.tolist(),
            features=f.tolist(),
            **r,
        )
        self.log.append(record)
        return record


def locked_release(controller, audit, epsilon_A=0):
    champion = controller.archive.best()
    entry = controller.archive.entry

    a0 = float(audit(entry.state))
    a1 = float(audit(champion.state))
    return (champion if a1 - a0 >= epsilon_A else entry), dict(
        incumbent_A=a0, candidate_A=a1, accepted=a1 - a0 >= epsilon_A
    )
