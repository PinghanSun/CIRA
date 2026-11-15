"""Whole-known-group allocation and BDD100K common-class conversion."""

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .paper_protocol import LABELS, bdd_domain
from .partition import shared_targets, assign_groups
from .data import validate_registry, registry_options
from .state import digest


def import_allocation(spec):
    """Bridge existing allocation/catalog CSVs and per-resource xywh annotation JSON."""
    import csv, io, zipfile
    from PIL import Image

    def rows(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def yes(value):
        return str(value).lower() in ("true", "1", "yes")

    destination = Path(spec["output_directory"]).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new import directory")
    catalogs = {r["resource_id"]: r for p in spec.get("catalogs", []) for r in rows(p)}
    annotations = {}
    for p in spec.get("annotations", []):
        d = _read(p)
        for key, value in d.items():
            if key in annotations and annotations[key] != value:
                raise ValueError("Conflicting annotation dictionaries")
            annotations[key] = value
    client_map = spec.get(
        "client_map",
        {
            "C1": "0",
            "C2": "1",
            "C3": "2",
            "C4": "3",
            "PUBLIC": "shared",
            "shared": "shared",
        },
    )
    domain_map = spec.get("domain_map", {})
    selected = []
    for p in spec["allocations"]:
        selected.extend(
            r
            for r in rows(p)
            if yes(r.get("active", r.get("active_for_pretraining", "True")))
        )
    if spec.get("dataset"):
        selected = [r for r in selected if r["dataset"] == spec["dataset"]]
    resources = [r["resource_id"] for r in selected]
    if len(set(resources)) != len(resources):
        raise ValueError("Duplicate active resource IDs in allocation inputs")

    prepared = []
    for r in selected:
        c = {**catalogs.get(r["resource_id"], {}), **r}
        role = c["role"]
        if not __import__("re").fullmatch(r"[A-Za-z0-9_.-]+", c["resource_id"]):
            raise ValueError("Unsafe resource ID")
        if role.startswith("DEV_"):
            role = role[4:]
        if role in ("SOURCE_TRAIN", "SOURCE_VAL", "U"):
            pass
        elif role not in ("D", "V", "A", "T"):
            raise ValueError("Unknown supplied role " + role)
        known = role != "U" and yes(c.get("has_detection_labels", "True"))
        if (
            known
            and c["resource_id"] not in annotations
            and int(c.get("noncrowd_boxes", -1)) != 0
        ):
            raise ValueError("Missing human GT dictionary for " + c["resource_id"])
        if not c.get("archive") or not c.get("entry"):
            raise ValueError("Bind archive and entry for every original")
        prepared.append((c, role, known))
    destination.mkdir(parents=True, exist_ok=True)
    out = []
    archives = {}
    try:
        for c, role, known in prepared:
            archive = Path(c["archive"])
            if str(archive) not in archives:
                archives[str(archive)] = zipfile.ZipFile(archive)
            z = archives[str(archive)]
            entry = z.getinfo(c["entry"])
            if c.get("entry_crc32") and entry.CRC != int(c["entry_crc32"]):
                raise ValueError("Archive CRC changed")
            raw = z.read(entry)
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
            if c.get("width") and (w, h) != (int(c["width"]), int(c["height"])):
                raise ValueError("RGB/annotation dimensions differ")
            boxes = []
            evaluation_annotations = []
            for b in annotations.get(c["resource_id"], []) if known else []:
                x, y, bw, bh = map(float, b["bbox"])
                category = int(b["class_id"])
                if (
                    category not in range(8)
                    or not 0 <= x < x + bw <= w
                    or not 0 <= y < y + bh <= h
                ):
                    raise ValueError("Invalid source box")
                box = [category, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
                crowd = bool(b.get("iscrowd", False))
                ignore = bool(b.get("ignore", False))
                evaluation_annotations.append(
                    {
                        "box": box,
                        "iscrowd": crowd,
                        "ignore": ignore,
                        "area_normalized": b.get("area", bw * bh) / (w * h),
                    }
                )
                if not crowd and not ignore:
                    boxes.append(box)
            image = (
                destination
                / "rgb"
                / (c["resource_id"] + Path(c["entry"]).suffix.lower())
            )
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(raw)
            identity = (
                c.get("confirmed_identity_group")
                or c.get("allocation_group")
                or (c["resource_id"] if role == "U" else None)
            )
            if not identity:
                raise ValueError("Known identity key is required")
            stage = int(c["stage"]) if str(c.get("stage", "")).isdigit() else 0
            client = client_map.get(
                c.get("client", "shared"), c.get("client", "shared")
            )
            out.append(
                {
                    "resource_id": c["resource_id"],
                    "image": str(image),
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "scene_id": identity,
                    "allocation_group": c.get("allocation_group", identity),
                    "role": role,
                    "client": client,
                    "stage": stage,
                    "domain": domain_map.get(c["domain"], c["domain"]),
                    "appearance": domain_map.get(c["domain"], c["domain"]),
                    "base_id": identity,
                    "boxes": boxes,
                    "evaluation_annotations": evaluation_annotations,
                    "verified_known_identity": spec.get(
                        "known_identities_checked", False
                    ),
                    "human_detection_labels": known,
                    "source_archive": str(archive),
                    "source_entry": c["entry"],
                }
            )
    finally:
        for z in archives.values():
            z.close()
    result = {
        "schema": "cira.imported_registry.20261113",
        "labels": list(LABELS),
        "records": out,
        "grouping_basis": "known_identity",
        "exposure_policy": spec.get("exposure_policy", "single_exposure"),
        "scene_independence_verified": False,
        "phase": spec.get("phase", "formal"),
        "augmentation": {"D": "acdc_four_view", "seed": spec["seed"]}
        if spec.get("augment_acdc")
        else {},
    }
    _write(destination / "registry.json", result)
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    p = Path(path)
    if p.exists():
        raise FileExistsError("Existing data manifests are preserved")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def single_exposure(spec):
    """Freeze one balanced weather appearance per online base; retain T intact."""
    source = Path(spec["registry"])
    doc = _read(source)
    if doc.get("exposure_policy") != "repeated_content":
        raise ValueError("Repeated-content source registry required")
    domains = spec["domains"]
    seed = spec["seed"]
    if len(set(domains)) != 4 or type(seed) is not int:
        raise ValueError("Four weather domains and integer seed required")
    groups = defaultdict(list)
    kept = []
    for row in doc["records"]:
        if row["role"] in ("D", "V", "A"):
            groups[row.get("base_id", row["scene_id"])].append(row)
        else:
            kept.append(copy.deepcopy(row))
    strata = defaultdict(list)
    for base, rows in groups.items():
        if len(rows) != 4 or set(r["domain"] for r in rows) != set(domains):
            raise ValueError("Exactly four appearances per online base required")
        contexts = {(str(r["client"]), r["role"]) for r in rows}
        if len(contexts) != 1:
            raise ValueError("Weather base crosses clients or roles")
        strata[next(iter(contexts))].append(base)
    counts = []
    for (client, role), bases in sorted(strata.items()):
        bases = sorted(
            bases,
            key=lambda b: digest(
                {"seed": seed, "base": b, "client": client, "role": role}
            ),
        )
        offset = int(digest({"seed": seed, "client": client, "role": role})[:8], 16) % 4
        chosen = Counter()
        for i, base in enumerate(bases):
            domain = domains[(offset + i) % 4]
            row = next(r for r in groups[base] if r["domain"] == domain)
            kept.append(copy.deepcopy(row))
            chosen[domain] += 1
        counts.append(
            {
                "client": client,
                "role": role,
                "bases": len(bases),
                "weather_counts": dict(chosen),
            }
        )
    if not groups:
        raise ValueError("No online bases")
    for r in kept:
        p = Path(r["image"])
        r["image"] = (
            str((source.parent / p).resolve()) if not p.is_absolute() else str(p)
        )
    out = {
        **doc,
        "records": kept,
        "exposure_policy": "single_exposure",
        "parent_registry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "single_exposure": {
            "seed": seed,
            "bases": len(groups),
            "strata": counts,
            "reporting_appearances_preserved": True,
        },
    }
    validate_registry(
        out["records"], spec.get("formal", False), **registry_options(out)
    )
    _write(spec["output"], out)
    return out


def scalable_group_assignment(groups, targets, seed):
    """Incremental O(groups * clients * classes), without all-pairs swaps."""
    rng = np.random.Generator(np.random.PCG64(seed))
    order = list(rng.permutation(len(groups)))
    order.sort(key=lambda i: -groups[i]["images"])
    total = np.sum([g["class_boxes"] for g in groups], axis=0).astype(float)
    support = total > 0
    den = np.maximum(total, 1)
    images = sum(g["images"] for g in groups)
    counts = np.zeros((targets.shape[1], targets.shape[0]))
    sizes = np.zeros(targets.shape[1])
    assignment = {}
    for i in order:
        g = groups[i]
        v = np.asarray(g["class_boxes"], float)
        before = counts / den - targets.T
        after = (counts + v) / den - targets.T
        change = (
            np.mean((after**2 - before**2)[:, support], axis=1)
            if support.any()
            else np.zeros(len(sizes))
        )
        change += ((sizes + g["images"]) / images - 1 / len(sizes)) ** 2 - (
            sizes / images - 1 / len(sizes)
        ) ** 2
        client = int(np.argmin(change))
        assignment[g["id"]] = client
        counts[client] += v
        sizes[client] += g["images"]
    return {"assignment": assignment, "algorithm": "incremental_whole_group_v1"}


def make_registry(spec):
    """Use supplied identity groups, then allocate whole groups reproducibly."""
    doc = _read(spec["source_records"])
    records = copy.deepcopy(doc["records"])
    if tuple(doc["labels"]) != LABELS:
        raise ValueError("Fixed common eight-class order required")
    seed = spec["seed"]
    domains = spec["domains"]
    clients = spec.get("clients", 4)
    if (
        type(seed) is not int
        or len(domains) != 4
        or len(set(domains)) != 4
        or clients != 4
    ):
        raise ValueError("Four clients/domains and integer seed required")
    fractions = spec.get("role_fractions", {"D": 0.76, "V": 0.12, "A": 0.12})
    if (
        set(fractions) != {"D", "V", "A"}
        or any(v <= 0 for v in fractions.values())
        or abs(sum(fractions.values()) - 1) > 1e-10
    ):
        raise ValueError("D/V/A fractions must sum to one")
    groups = defaultdict(list)
    for r in records:
        if not r.get("scene_id"):
            raise ValueError("Bind a known identity/allocation group explicitly")
        groups[r.get("allocation_group", r["scene_id"])].append(r)

    rng = np.random.Generator(np.random.PCG64(seed))
    keys = sorted(groups)
    rng.shuffle(keys)
    unknown = [
        g for g in keys if all(r.get("role") in (None, "unassigned") for r in groups[g])
    ]
    if any(
        any(r.get("role") in (None, "unassigned") for r in rows) and g not in unknown
        for g, rows in groups.items()
    ):
        raise ValueError("Partially assigned group")
    total = sum(len(groups[g]) for g in unknown)
    used = Counter()
    for g in sorted(unknown, key=lambda g: -len(groups[g])):
        role = max(fractions, key=lambda k: (total * fractions[k] - used[k], k))
        used[role] += len(groups[g])
        for r in groups[g]:
            r["role"] = role
    online = {
        g: rows for g, rows in groups.items() if rows[0]["role"] in ("D", "V", "A")
    }
    if not online:
        raise ValueError("No online groups to allocate")
    if any(len({r["role"] for r in rows}) != 1 for rows in groups.values()):
        raise ValueError("Global group role leakage")
    alpha = spec.get("alpha", 1.0)
    targets = shared_targets(
        seed, clients, 8, 1.0 if alpha == "equal" else alpha, equal=alpha == "equal"
    )
    assignment = []
    for g, rows in sorted(online.items()):
        counts = np.zeros(8, dtype=int)
        cell_images = Counter()
        cell_classes = defaultdict(lambda: [0] * 8)
        for r in rows:
            cell = r["domain"] + "|" + r["role"]
            cell_images[cell] += 1
            for b in r["boxes"]:
                counts[int(b[0])] += 1
                cell_classes[cell][int(b[0])] += 1
        assignment.append(
            {
                "id": g,
                "images": len(rows),
                "class_boxes": counts.tolist(),
                "cell_images": dict(cell_images),
                "cell_classes": dict(cell_classes),
            }
        )
    existing = all(
        str(r.get("client")) in ("0", "1", "2", "3") and r.get("stage") in (1, 2, 3, 4)
        for rows in online.values()
        for r in rows
    )
    preserve = spec.get("preserve_allocation", existing)
    if preserve and not existing:
        raise ValueError("Cannot preserve an incomplete allocation")
    if preserve:
        result = {"assignment": {}, "algorithm": "preserved_frozen_allocation"}
        for g, rows in online.items():
            if len({str(r["client"]) for r in rows}) != 1:
                raise ValueError("Existing group crosses clients")
            result["assignment"][g] = int(rows[0]["client"])
    else:
        from .allocation import hard_group_assignment, balanced_quotas

        result = hard_group_assignment(
            assignment,
            targets,
            seed,
            spec.get("image_quotas", balanced_quotas(assignment, clients)),
            spec.get("max_passes", 8),
            spec.get("swap_pairs", 512),
        )
    orders = {
        str(i): np.random.Generator(np.random.PCG64(seed + i))
        .permutation(domains)
        .tolist()
        for i in range(clients)
    }
    if preserve:
        for client in orders:
            seen = {
                r["stage"]: r["domain"]
                for r in records
                if str(r.get("client")) == client and r["role"] in ("D", "V", "A")
            }
            if set(seen) != {1, 2, 3, 4} or len(set(seen.values())) != 4:
                raise ValueError("Frozen allocation requires all four domain stages")
            if any(
                seen[r["stage"]] != r["domain"]
                for r in records
                if str(r.get("client")) == client and r["role"] in ("D", "V", "A")
            ):
                raise ValueError("Conflicting frozen domain order")
            orders[client] = [seen[i] for i in range(1, 5)]
    root = Path(spec["source_records"]).resolve().parent
    for r in records:
        path = Path(r["image"])
        path = path if path.is_absolute() else root / path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if r.get("image_sha256", actual) != actual:
            raise ValueError("Source RGB changed")
        r["image"] = str(path.resolve())
        r["image_sha256"] = actual
        if r["role"] in ("D", "V", "A"):
            if r["domain"] not in domains:
                raise ValueError("Online domain outside frozen stream")
            r["client"] = str(
                result["assignment"][r.get("allocation_group", r["scene_id"])]
            )
            r["stage"] = orders[r["client"]].index(r["domain"]) + 1
        else:
            r.setdefault("client", "shared")
            r.setdefault("stage", 0)
    out = {
        "schema": "cira.image_registry.20261113",
        "labels": list(LABELS),
        "records": records,
        "exposure_policy": spec.get("exposure_policy", "single_exposure"),
        "grouping_basis": spec.get("grouping_basis", "known_identity"),
        "scene_independence_verified": False,
        "seed": seed,
        "domain_orders": orders,
        "alpha": alpha,
        "targets": targets.tolist(),
        "role_targets": fractions,
        "assignment_algorithm": result["algorithm"],
        "assignment_evidence": {k: v for k, v in result.items() if k != "assignment"},
        "phase": spec.get("phase", "formal"),
        "augmentation": {"D": "acdc_four_view", "seed": seed}
        if spec.get("augment_acdc", False)
        else {},
    }
    validate_registry(records, spec.get("formal", False), **registry_options(out))
    census = Counter((str(r["client"]), r["stage"], r["role"]) for r in records)
    out["role_census"] = [
        {"client": c, "stage": s, "role": role, "images": n}
        for (c, s, role), n in sorted(census.items())
    ]
    out["missing_online_cells"] = [
        f"{i}/{s}/{role}"
        for i in range(4)
        for s in range(1, 5)
        for role in ("D", "V", "A")
        if not census[(str(i), s, role)]
    ]
    out["class_support"] = [
        {
            "client": str(i),
            "stage": s,
            "role": role,
            "boxes_per_class": [
                sum(
                    int(b[0]) == k
                    for r in records
                    if str(r["client"]) == str(i)
                    and r["stage"] == s
                    and r["role"] == role
                    for b in r["boxes"]
                )
                for k in range(8)
            ],
        }
        for i in range(4)
        for s in range(1, 5)
        for role in ("D", "V", "A")
    ]
    if spec.get("require_full_support", False) and out["missing_online_cells"]:
        raise ValueError("Whole-group allocation cannot satisfy all requested cells")
    out["input_sha256"] = digest(doc)
    _write(spec["output"], out)
    return out


def convert_bdd(spec):
    """Official Detection 2020 JSON -> common vocabulary, retaining explicit IDs."""
    from PIL import Image

    source = _read(spec["annotations"])
    groups = _read(spec["identity_map"])
    records = []
    reserved = []
    aliases = {"pedestrian": "person", "motor": "motorcycle", "bike": "bicycle"}
    skipped = Counter()
    for item in source:
        name = item["name"]
        domain = bdd_domain(item.get("attributes", {}))
        if domain is None:
            reserved.append(name)
            continue
        if name not in groups:
            raise ValueError("Bind BDD video/ride/known identity group for " + name)
        g = groups[name]
        path = Path(spec["image_root"]) / name
        with Image.open(path) as image:
            w, h = image.size
        boxes = []
        evaluation_annotations = []
        for label in item.get("labels", []):
            category = aliases.get(label["category"], label["category"])
            if category not in LABELS:
                skipped[category] += 1
                continue
            if "box2d" not in label:
                continue
            b = label["box2d"]
            x0, y0, x1, y1 = b["x1"], b["y1"], b["x2"], b["y2"]
            if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
                raise ValueError("Resolve invalid source box")
            box = [
                LABELS.index(category),
                (x0 + x1) / (2 * w),
                (y0 + y1) / (2 * h),
                (x1 - x0) / w,
                (y1 - y0) / h,
            ]
            attributes = label.get("attributes", {})
            crowd = bool(label.get("iscrowd", attributes.get("crowd", False)))
            ignore = bool(label.get("ignore", False))
            evaluation_annotations.append(
                {
                    "box": box,
                    "iscrowd": crowd,
                    "ignore": ignore,
                    "area_normalized": (x1 - x0) * (y1 - y0) / (w * h),
                }
            )
            if not crowd and not ignore:
                boxes.append(box)
        records.append(
            {
                "image": str(path.resolve()),
                "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scene_id": g["scene_id"],
                "verified_known_identity": g.get("verified_known_identity", False),
                "role": g.get("role", "unassigned"),
                "client": g.get("client", "shared"),
                "stage": g.get("stage", 0),
                "domain": domain,
                "boxes": boxes,
                "evaluation_annotations": evaluation_annotations,
                "source_id": name,
                "attributes": item.get("attributes", {}),
            }
        )
    out = {
        "labels": list(LABELS),
        "records": records,
        "reserved_unknown_attributes": reserved,
        "excluded_categories": dict(skipped),
        "metric_vocabulary": "common_eight_classes",
        "source_sha256": hashlib.sha256(
            Path(spec["annotations"]).read_bytes()
        ).hexdigest(),
    }
    _write(spec["output"], out)
    return out
