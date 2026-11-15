"""BBox AP with score-ordered COCO matching, crowd and explicit ignore regions."""

import numpy as np
import torch


def _array(value, columns=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    a = np.asarray(value, dtype=float)
    if columns is not None:
        if a.size == 0:
            a = a.reshape(0, columns)
        if a.ndim != 2 or a.shape[1] != columns:
            raise ValueError("Invalid detection/target shape")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite detection evidence")
    return a


def coco_bbox_report(predictions, targets, class_names=None, max_detections=100):
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Paired nonempty image evidence required")
    if type(max_detections) is not int or max_detections < 1:
        raise ValueError("Positive max detections required")
    prepared = []
    categories = set()
    thresholds = np.linspace(0.5, 0.95, 10)
    for pred, target in zip(predictions, targets):
        d = _array(pred, 6)
        if isinstance(target, dict):
            g = _array(target["boxes"], 5)
            crowd = _array(target.get("iscrowd", np.zeros(len(g)))).reshape(-1)
            ignored = _array(target.get("ignore", np.zeros(len(g)))).reshape(-1)
        else:
            g = _array(target, 5)
            crowd = ignored = np.zeros(len(g))
        if (
            len(crowd) != len(g)
            or len(ignored) != len(g)
            or not np.isin(crowd, [0, 1]).all()
            or not np.isin(ignored, [0, 1]).all()
        ):
            raise ValueError("One boolean crowd/ignore flag per GT box required")
        for ids in (d[:, 5], g[:, 0]):
            if (ids < 0).any() or (ids != np.floor(ids)).any():
                raise ValueError("Integer category IDs required")
            if class_names is not None and (ids >= len(class_names)).any():
                raise ValueError("Category outside vocabulary")
        if ((d[:, 4] < 0) | (d[:, 4] > 1)).any():
            raise ValueError("Confidence outside [0,1]")
        if (len(d) and ((d[:, 2:4] - d[:, :2]) <= 0).any()) or (
            len(g) and ((g[:, 3:5] - g[:, 1:3]) <= 0).any()
        ):
            raise ValueError("Positive area xyxy boxes required")
        categories.update(g[:, 0].astype(int))
        categories.update(d[:, 5].astype(int))
        prepared.append((d, g, crowd.astype(bool), np.logical_or(ignored, crowd)))
    rows = []
    vocabulary = (
        range(len(class_names)) if class_names is not None else sorted(categories)
    )
    for category in vocabulary:
        scores = []
        matches = []
        ignores = []
        support = 0
        prediction_count = 0
        ignored_gt = 0
        for d, g, crowd, ignored in prepared:
            gt = g[:, 0] == category
            g = g[gt]
            crowd = crowd[gt]
            ignored = ignored[gt]
            gi = np.argsort(ignored, kind="stable")
            g = g[gi]
            crowd = crowd[gi]
            ignored = ignored[gi]
            support += int((~ignored).sum())
            ignored_gt += int(ignored.sum())
            d = d[d[:, 5] == category]
            prediction_count += len(d)
            d = d[np.argsort(-d[:, 4], kind="stable")[:max_detections]]
            hit = np.zeros((len(d), 10), bool)
            skip = np.zeros_like(hit)
            if len(d) and len(g):
                lo = np.maximum(d[:, None, :2], g[None, :, 1:3])
                hi = np.minimum(d[:, None, 2:4], g[None, :, 3:5])
                inter = np.maximum(hi - lo, 0).prod(2)
                da = (d[:, 2:4] - d[:, :2]).prod(1)
                ga = (g[:, 3:5] - g[:, 1:3]).prod(1)
                denominator = da[:, None] + ga[None, :] - inter
                denominator[:, crowd] = np.broadcast_to(da[:, None], denominator.shape)[
                    :, crowd
                ]
                iou = inter / np.maximum(denominator, np.finfo(float).eps)
                for t, threshold in enumerate(thresholds):
                    used = np.zeros(len(g), bool)
                    for i in range(len(d)):
                        best = -1
                        level = min(threshold, 1 - 1e-10)
                        for j in range(len(g)):
                            if used[j] and not crowd[j]:
                                continue
                            if best >= 0 and not ignored[best] and ignored[j]:
                                break
                            if iou[i, j] < level:
                                continue
                            level = iou[i, j]
                            best = j
                        if best >= 0:
                            used[best] = True
                            skip[i, t] = ignored[best]
                            hit[i, t] = not ignored[best]
            scores.extend(d[:, 4])
            matches.extend(hit)
            ignores.extend(skip)
        ap = np.zeros(10) if support else None
        if support and scores:
            order = np.argsort(-np.asarray(scores), kind="stable")
            hit = np.asarray(matches, bool)[order]
            skip = np.asarray(ignores, bool)[order]
            tp = np.cumsum(hit & ~skip, axis=0)
            fp = np.cumsum(~hit & ~skip, axis=0)
            recall = tp / support
            precision = tp / (tp + fp + np.spacing(1))
            for t in range(10):
                envelope = np.maximum.accumulate(precision[::-1, t])[::-1]
                indexes = np.searchsorted(
                    recall[:, t], np.linspace(0, 1, 101), side="left"
                )
                values = np.zeros(101)
                valid = indexes < len(envelope)
                values[valid] = envelope[indexes[valid]]
                ap[t] = values.mean()
        rows.append(
            {
                "class_index": int(category),
                "class_name": class_names[category] if class_names else str(category),
                "GT_support": support,
                "ignored_GT": ignored_gt,
                "prediction_count": prediction_count,
                "AP50": float(ap[0]) if support else None,
                "AP75": float(ap[5]) if support else None,
                "AP50_95": float(ap.mean()) if support else None,
            }
        )
    supported = [r for r in rows if r["GT_support"]]
    return {
        "schema": "cira.coco_bbox_report.v1",
        "evaluator": "cira_coco_bbox_101recall_max100_ignore_v1",
        "score_units": "fraction",
        "images": len(predictions),
        "supported_classes": len(supported),
        "GT_boxes": sum(r["GT_support"] for r in rows),
        "ignored_GT": sum(r["ignored_GT"] for r in rows),
        "prediction_boxes": sum(r["prediction_count"] for r in rows),
        "per_class": rows,
        **{
            k: float(np.mean([r[k] for r in supported])) if supported else None
            for k in ("AP50", "AP75", "AP50_95")
        },
        "max_detections_per_image_category": max_detections,
        "IoU_thresholds": thresholds.tolist(),
        "unsupported_class_rule": "null_not_zero",
    }


def batch_targets(batch):
    """Evaluation metadata is separate from nonignored fitting targets."""
    if "evaluation_targets" in batch:
        return batch["evaluation_targets"]
    out = []
    height, width = batch["img"].shape[-2:]
    for i in range(len(batch["img"])):
        keep = batch["batch_idx"] == i
        boxes = batch["bboxes"][keep]
        center, half = boxes[:, :2], boxes[:, 2:] / 2
        xyxy = torch.cat((center - half, center + half), 1) * boxes.new_tensor(
            [width, height, width, height]
        )
        out.append(
            {
                "boxes": torch.cat((batch["cls"][keep].reshape(-1, 1), xyxy), 1),
                "iscrowd": torch.zeros(len(boxes)),
                "ignore": torch.zeros(len(boxes)),
            }
        )
    return out
