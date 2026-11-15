"""Supervised and teacher-student initialization."""

from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from .state import digest


def validate_recipe(recipe, condition):
    if condition not in ("C0", "C1", "C2"):
        raise ValueError("Unknown initialization condition")
    for key in ("updates", "validation_interval"):
        if type(recipe.get(key)) is not int or recipe[key] < 1:
            raise ValueError(f"Lock positive integer {key}")
    if (
        not isinstance(recipe.get("lr"), (int, float))
        or not math.isfinite(recipe["lr"])
        or recipe["lr"] <= 0
    ):
        raise ValueError("Lock positive learning rate")
    if condition == "C2":
        for key in ("teacher_momentum", "pseudo_confidence", "unlabeled_weight"):
            v = recipe.get(key)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                raise ValueError("Lock " + key)
        if (
            not 0 <= recipe["teacher_momentum"] < 1
            or not 0 < recipe["pseudo_confidence"] <= 1
            or recipe["unlabeled_weight"] < 0
        ):
            raise ValueError("Invalid teacher recipe")


@torch.no_grad()
def ema_update(teacher, student, momentum):
    t, s = teacher.state_dict(), student.state_dict()
    if t.keys() != s.keys():
        raise ValueError("Teacher and student structures differ")
    for name, value in t.items():
        if value.is_floating_point():
            value.mul_(momentum).add_(s[name].detach(), alpha=1 - momentum)
        else:
            value.copy_(s[name])


def train_initialization(
    student, labeled, unlabeled, recipe, condition, loss_fn, pseudo_fn, validation_fn
):
    """Real optimizer loop. pseudo_fn returns None for no accepted pseudo boxes."""
    validate_recipe(recipe, condition)
    if not len(labeled) or (
        condition == "C2" and (unlabeled is None or not len(unlabeled))
    ):
        raise ValueError("Missing preparation data")
    teacher = (
        copy.deepcopy(student).eval().requires_grad_(False)
        if condition == "C2"
        else None
    )
    optimizer = torch.optim.AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=recipe["lr"],
        betas=tuple(recipe.get("betas", (0.9, 0.999))),
        eps=recipe.get("epsilon", 1e-8),
        weight_decay=recipe.get("weight_decay", 0.0),
    )
    logs = []
    best = None
    best_score = -math.inf
    best_update = None
    u_reads = 0
    accepted_images = 0
    accepted_boxes = 0
    start = time.perf_counter()
    for update in range(recipe["updates"]):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        batch = labeled[update % len(labeled)]
        sup = loss_fn(student, batch)
        unsup = sup.new_zeros(())
        pseudo = None
        if condition == "C2":
            weak = unlabeled[update % len(unlabeled)]
            u_reads += int(weak["img"].shape[0])
            with torch.no_grad():
                pseudo = pseudo_fn(teacher, weak, recipe["pseudo_confidence"])
            if pseudo is not None:
                n = int(len(pseudo["cls"]))
                accepted_boxes += n
                if n:
                    accepted_images += int(torch.unique(pseudo["batch_idx"]).numel())
                    unsup = loss_fn(student, pseudo)
        total = sup + (recipe["unlabeled_weight"] if condition == "C2" else 0.0) * unsup
        if not torch.isfinite(total):
            raise ValueError("Nonfinite preparation loss")
        total.backward()
        optimizer.step()
        if teacher is not None:
            ema_update(teacher, student, recipe["teacher_momentum"])
        row = {
            "update": update + 1,
            "supervised_loss": float(sup.detach()),
            "unlabeled_loss": float(unsup.detach()),
            "accepted_boxes": 0 if pseudo is None else len(pseudo["cls"]),
        }
        if (update + 1) % recipe["validation_interval"] == 0 or update + 1 == recipe[
            "updates"
        ]:
            student.eval()
            score = float(validation_fn(student))
            if not math.isfinite(score):
                raise ValueError("Nonfinite SOURCE validation score")
            row["SOURCE_clear_validation"] = score
            if score > best_score:
                best_score = score
                best_update = update + 1
                best = {
                    k: v.detach().cpu().clone() for k, v in student.state_dict().items()
                }
        logs.append(row)
    student.load_state_dict(best)
    return {
        "condition": condition,
        "actual_optimizer_updates": recipe["updates"],
        "best_update": best_update,
        "SOURCE_clear_validation": best_score,
        "U_image_reads": u_reads,
        "accepted_pseudo_images": accepted_images,
        "accepted_pseudo_boxes": accepted_boxes,
        "elapsed_seconds": time.perf_counter() - start,
        "rows": logs,
        "teacher_state": None
        if teacher is None
        else {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
    }


def _manifest(path, roles, allow_boxes=True):
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    from .paper_protocol import LABELS

    if tuple(d["labels"]) != LABELS or not d["records"]:
        raise ValueError("Expected eight-class nonempty preparation pool")
    for r in d["records"]:
        if r["role"] not in roles:
            raise ValueError("Forbidden data role during preparation: " + r["role"])
        if not allow_boxes and (r.get("boxes") or r.get("evaluation_annotations")):
            raise ValueError("U must not contain human detection targets")
    return p, d


def run_yolo_initialization(spec):
    """Explicit native preparation command, with SOURCE validation and U pseudo labels."""
    session_started = time.perf_counter()
    from .detector_baselines import new_detector
    from .data import DetectionBatches
    from .postprocess import decode
    from .coco_evaluation import coco_bbox_report, batch_targets

    validate_recipe(spec["recipe"], spec["condition"])
    output = Path(spec["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing preparation outputs")
    lp, L = _manifest(spec["labeled_manifest"], {"SOURCE", "SOURCE_TRAIN"})
    vp, V = _manifest(spec["validation_manifest"], {"SOURCE_VAL"})
    up, U = (
        _manifest(spec["unlabeled_manifest"], {"U"}, False)
        if spec["condition"] == "C2"
        else (None, None)
    )
    groups = lambda d: {r["scene_id"] for r in d["records"]}
    if groups(L) & groups(V) or (U and (groups(U) & (groups(L) | groups(V)))):
        raise ValueError("Preparation pool identity leakage")
    clear = {r.get("appearance", r.get("domain")) for r in V["records"]}
    if clear != {"clear"}:
        raise ValueError("Checkpoint selection must use clear SOURCE validation")
    bases = {r.get("base_id", r["scene_id"]) for r in L["records"]}
    actual_counts = {
        "labeled_rgb": len(L["records"]),
        "validation_rgb": len(V["records"]),
        "unlabeled_rgb": 0 if U is None else len(U["records"]),
        "labeled_bases": len(bases),
    }
    for key, value in spec.get("expected_counts", {}).items():
        if actual_counts.get(key) != value:
            raise ValueError("Preparation count differs from locked " + key)
    if spec["condition"] == "C0" and any(
        r.get("appearance", r.get("domain")) != "clear" for r in L["records"]
    ):
        raise ValueError("C0 uses clear fitting images only")
    if not bases:
        raise ValueError("Empty labeled bases")
    device = spec["device"]
    size = spec.get("image_size", 640)
    bs = spec["batch_size"]
    seed = spec["seed"]
    torch.manual_seed(seed)
    checkpoint = spec.get("initial_checkpoint")
    kind = spec.get("initial_checkpoint_type", "eight_class")
    if not checkpoint and spec.get("allow_random_initialization") is not True:
        raise ValueError(
            "Bind initial weights or explicitly authorize a random-initialization software/development run"
        )
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if kind not in ("eight_class", "coco_compatible"):
        raise ValueError("Explicit checkpoint transfer type required")
    model = new_detector(
        spec["detector"],
        device,
        checkpoint=checkpoint if kind == "eight_class" else None,
        seed=seed,
    )
    transferred = 0
    if checkpoint and kind == "coco_compatible":
        from ultralytics import YOLO

        source = YOLO(str(checkpoint)).model
        if [type(m).__name__ for m in source.model] != [
            type(m).__name__ for m in model.model
        ]:
            raise ValueError("COCO architecture differs from detector profile")
        wanted = model.state_dict()
        compatible = {
            k: v
            for k, v in source.state_dict().items()
            if k in wanted and wanted[k].shape == v.shape
        }
        transferred = sum(v.numel() for v in compatible.values())
        if transferred / sum(v.numel() for v in wanted.values()) < 0.7:
            raise ValueError("Insufficient matching COCO weights")
        model.load_state_dict(compatible, strict=False)
    labeled = DetectionBatches(L["records"], lp.parent, size, bs)
    unlabeled = (
        DetectionBatches(
            [{**r, "boxes": []} for r in U["records"]], up.parent, size, bs
        )
        if U
        else None
    )
    validation = DetectionBatches(V["records"], vp.parent, size, 1)

    def move(batch):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def loss(m, b):
        b = move(b)
        raw = m(b["img"])
        value, _ = m.loss(b, preds=raw)
        return value.sum() / len(b["img"])

    def predict(m, b):
        m.eval()
        b = move(b)
        return decode(m(b["img"]), 8, end2end=getattr(m, "end2end", False)), b

    def pseudo(m, b, threshold):
        outputs, b = predict(m, b)
        boxes = []
        classes = []
        indices = []
        for i, p in enumerate(outputs):
            p = p[p[:, 4] >= threshold]
            if not len(p):
                continue
            xy = p[:, :4]
            hw = xy[:, 2:] - xy[:, :2]
            centers = (xy[:, :2] + xy[:, 2:]) / 2
            shape = torch.tensor([size, size], device=xy.device)
            boxes.append(torch.cat((centers / shape, hw / shape), 1))
            classes.append(p[:, 5:6])
            indices.append(torch.full((len(p),), i, device=xy.device))
        if not boxes:
            return None
        return {
            "img": b["img"],
            "bboxes": torch.cat(boxes),
            "cls": torch.cat(classes),
            "batch_idx": torch.cat(indices),
        }

    @torch.no_grad()
    def validate(m):
        predictions = []
        truth = []
        for b in validation:
            outputs, bb = predict(m, b)
            predictions.extend(outputs)
            xywh = bb["bboxes"]
            center = xywh[:, :2] * size
            half = xywh[:, 2:] * size / 2
            truth.extend(batch_targets(bb))
        report = coco_bbox_report(predictions, truth)
        if report["AP50_95"] is None:
            raise ValueError("SOURCE validation has no nonignored GT")
        return report["AP50_95"]

    from .metering import ExecutedFlops

    counter = ExecutedFlops(strict=spec.get("strict_computation_meter", False))
    with counter:
        result = train_initialization(
            model,
            labeled,
            unlabeled,
            spec["recipe"],
            spec["condition"],
            loss,
            pseudo,
            validate,
        )
    result["computation"] = {
        "counted_scalar_operations": counter.total,
        "unknown_operators": dict(counter.unknown),
        "complete_operator_coverage": not bool(counter.unknown),
        "boundary": "teacher creation, fitting, EMA and validation; model construction and serialization excluded",
    }
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = output / "best.pt"
    torch.save({"model": copy.deepcopy(model).cpu().eval(), "ema": None}, checkpoint)
    teacher = result.pop("teacher_state")
    if teacher is not None:
        torch.save(teacher, output / "teacher.pt")
    result.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        protocol=spec,
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (lp, vp, up)
            if p
        },
        labeled_originals=len(L["records"]),
        labeled_bases=len(bases),
        unlabeled_originals=0 if U is None else len(U["records"]),
    )
    result["teacher_student_view"] = "shared_identity_view"
    result["initial_checkpoint_sha256"] = (
        hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if checkpoint
        else None
    )
    result["COCO_transferred_elements"] = transferred
    result["end_to_end_elapsed_seconds"] = time.perf_counter() - session_started
    (output / "preparation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result
