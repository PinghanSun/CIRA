"""Exact evaluation reuse and AP on complete detectors."""

from __future__ import annotations
import numpy as np
import torch
from .resources import Cost
from .backend import work
from .state import digest


def detection_report(predictions, targets, class_names=None):
    """One prediction/GT pool supplies all AP thresholds and per-class support."""
    from ultralytics.utils.metrics import ap_per_class, box_iou

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Equal, nonempty prediction and target image lists required")
    tp, confidence, predicted, truth = [], [], [], []
    for pred, target in zip(predictions, targets):
        pred = torch.as_tensor(pred).detach().float().cpu()
        target = torch.as_tensor(target).detach().float().cpu()
        if (
            pred.ndim != 2
            or pred.shape[1] != 6
            or target.ndim != 2
            or target.shape[1] != 5
        ):
            raise ValueError(
                "Expected predictions Nx6 xyxy/conf/class and targets Mx5 class/xyxy"
            )
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite detection evidence")
        for ids in (pred[:, 5], target[:, 0]):
            if (ids < 0).any() or (ids != ids.round()).any():
                raise ValueError("Nonnegative integer class IDs required")
            if class_names is not None and (ids >= len(class_names)).any():
                raise ValueError("Class ID outside declared vocabulary")
        if ((pred[:, 4] < 0) | (pred[:, 4] > 1)).any():
            raise ValueError("Detection confidence must be in [0,1]")
        for boxes in (pred[:, :4], target[:, 1:]):
            if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
                raise ValueError("Positive-area xyxy boxes required")
        correct = torch.zeros((len(pred), 10), dtype=torch.bool)
        if len(pred) and len(target):
            iou = box_iou(target[:, 1:], pred[:, :4])
            same = target[:, :1] == pred[:, 5][None]
            for j, threshold in enumerate(torch.linspace(0.5, 0.95, 10)):
                gs, ps = torch.where((iou >= threshold) & same)
                matches = sorted(
                    zip(gs.tolist(), ps.tolist()), key=lambda ij: (-float(iou[ij]), ij)
                )
                used_g, used_p = set(), set()
                for g, p in matches:
                    if g not in used_g and p not in used_p:
                        correct[p, j] = True
                        used_g.add(g)
                        used_p.add(p)
        tp.append(correct.numpy())
        confidence.append(pred[:, 4].numpy())
        predicted.append(pred[:, 5].numpy())
        truth.append(target[:, 0].numpy())
    gt = np.concatenate(truth).astype(int)
    pd = np.concatenate(predicted).astype(int)
    all_ap = {}
    if len(gt) and len(pd):
        result = ap_per_class(
            np.concatenate(tp), np.concatenate(confidence), pd, gt, plot=False
        )
        all_ap = {int(c): row for c, row in zip(result[6], result[5])}
    supported = sorted(set(gt.tolist()))
    vocabulary = (
        range(len(class_names))
        if class_names is not None
        else sorted(set(gt.tolist()) | set(pd.tolist()))
    )
    rows = []
    for c in vocabulary:
        support = int((gt == c).sum())
        ap = np.asarray(all_ap.get(c, np.zeros(10))) if support else None
        rows.append(
            dict(
                class_index=int(c),
                class_name=class_names[c] if class_names is not None else str(c),
                GT_support=support,
                prediction_count=int((pd == c).sum()),
                AP50=float(ap[0]) if support else None,
                AP75=float(ap[5]) if support else None,
                AP50_95=float(ap.mean()) if support else None,
            )
        )
    used = [r for r in rows if r["GT_support"]]
    return dict(
        schema="cira.detection_report.v1",
        score_units="fraction",
        images=len(predictions),
        GT_boxes=int(len(gt)),
        prediction_boxes=int(len(pd)),
        supported_classes=len(supported),
        AP50=float(np.mean([r["AP50"] for r in used])) if used else None,
        AP75=float(np.mean([r["AP75"] for r in used])) if used else None,
        AP50_95=float(np.mean([r["AP50_95"] for r in used])) if used else None,
        per_class=rows,
        IoU_thresholds=[round(0.5 + 0.05 * i, 2) for i in range(10)],
        evaluator="ultralytics_ap_per_class_project_greedy_iou_v1",
        unsupported_class_rule="null_not_zero",
        evidence_scope="provided_prediction_and_GT_tensors",
    )


def detection_ap(predictions, targets):
    from ultralytics.utils.metrics import ap_per_class, box_iou

    tp, conf, pred_cls, target_cls = [], [], [], []
    for pred, target in zip(predictions, targets):
        correct = torch.zeros((len(pred), 10), dtype=torch.bool)
        if len(pred) and len(target):
            iou = box_iou(target[:, 1:], pred[:, :4])
            same = target[:, 0:1] == pred[:, 5].reshape(1, -1)
            for j, threshold in enumerate(torch.linspace(0.5, 0.95, 10)):
                rows, cols = torch.where((iou >= threshold) & same)
                matches = sorted(
                    zip(rows.tolist(), cols.tolist()),
                    key=lambda ij: (-float(iou[ij]), ij),
                )
                used_gt, used_pred = set(), set()
                for gt, pd in matches:
                    if gt not in used_gt and pd not in used_pred:
                        correct[pd, j] = True
                        used_gt.add(gt)
                        used_pred.add(pd)
        tp.append(correct.numpy())
        conf.append(pred[:, 4].numpy())
        pred_cls.append(pred[:, 5].numpy())
        target_cls.append(target[:, 0].numpy())
    if not target_cls or sum(map(len, target_cls)) == 0:
        raise ValueError("AP unavailable: no positive ground truth")
    if sum(map(len, conf)) == 0:
        return 0.0
    result = ap_per_class(
        np.concatenate(tp),
        np.concatenate(conf),
        np.concatenate(pred_cls),
        np.concatenate(target_cls),
        plot=False,
    )

    return float(result[5].mean())


def detection_ap_torch(predictions, targets):
    """Same 10 IoU thresholds / 101-point PR interpolation, traceable tensor math."""
    from ultralytics.utils.metrics import box_iou

    correct, confidences, predicted, truth = [], [], [], []
    for pred, target in zip(predictions, targets):
        pred, target = pred.detach().float().cpu(), target.detach().float().cpu()
        hit = torch.zeros((len(pred), 10), dtype=torch.bool)
        if len(pred) and len(target):
            overlaps = box_iou(target[:, 1:], pred[:, :4])
            same = target[:, :1] == pred[:, 5][None]
            for j, threshold in enumerate(torch.linspace(0.5, 0.95, 10)):
                g, p = torch.where((overlaps >= threshold) & same)
                if len(g):
                    order = torch.argsort(overlaps[g, p], descending=True, stable=True)
                    used_g, used_p = set(), set()
                    for k in order.tolist():
                        gi, pi = int(g[k]), int(p[k])
                        if gi not in used_g and pi not in used_p:
                            hit[pi, j] = True
                            used_g.add(gi)
                            used_p.add(pi)
        correct.append(hit)
        confidences.append(pred[:, 4])
        predicted.append(pred[:, 5])
        truth.append(target[:, 0])
    if not truth or sum(len(v) for v in truth) == 0:
        raise ValueError("AP unavailable: no positive ground truth")
    if sum(len(v) for v in confidences) == 0:
        return 0.0
    tp, confidence, classes, gt = (
        torch.cat(correct),
        torch.cat(confidences),
        torch.cat(predicted),
        torch.cat(truth),
    )
    order = torch.argsort(confidence, descending=True, stable=True)
    tp, classes = tp[order], classes[order]
    ap = []
    grid = torch.linspace(0, 1, 101)
    for category in torch.unique(gt, sorted=True):
        present = classes == category
        if not present.any():
            ap.append(torch.zeros(10))
            continue
        positive = tp[present].cumsum(0).float()
        negative = (~tp[present]).cumsum(0).float()
        recall = positive / (gt == category).sum()
        precision = positive / (positive + negative + 1e-16)
        values = []
        for j in range(10):
            x = torch.cat((torch.zeros(1), recall[:, j], torch.ones(1)))
            y = torch.cat((torch.ones(1), precision[:, j], torch.zeros(1)))
            y = y.flip(0).cummax(0).values.flip(0)
            index = (
                torch.searchsorted(x.contiguous(), grid, right=True)
                .sub(1)
                .clamp(0, len(x) - 2)
            )
            x0, x1 = x[index], x[index + 1]
            y0, y1 = y[index], y[index + 1]
            interpolated = y0 + (grid - x0) * (y1 - y0) / (x1 - x0).clamp(min=1e-16)
            interpolated = torch.where(grid >= x[-1], y[-1], interpolated)
            values.append(
                (
                    (interpolated[1:] + interpolated[:-1])
                    * 0.5
                    * (grid[1:] - grid[:-1])
                ).sum()
            )
        ap.append(torch.stack(values))
    return float(torch.stack(ap).mean())


class EvaluationCache:
    def __init__(self):
        self.scores = {}
        self.records = []

    def key(self, backend, state, split):
        return digest(
            {
                "model": state.model_hash(backend.full_base_identity),
                "evaluator": backend.evaluator,
                "data": split.identity,
                "execution": backend.execution_identity,
            }
        )

    def evaluate(
        self,
        backend,
        state,
        cache,
        split,
        ledger,
        scope="search",
        reuse=True,
        prepared_model=None,
    ):
        if scope == "search" and split.role != "V":
            raise ValueError("Search selection may use V only")
        if scope == "release" and split.role != "A":
            raise ValueError("Locked release audit may use A only")
        key = self.key(backend, state, split)
        if reuse and key in self.scores:
            self.records.append(
                {"key": key, "state": state.hash, "role": split.role, "reuse": True}
            )
            return self.scores[key], True
        is_new = key not in self.scores
        if scope == "search":
            ledger.guard(Cost(E=int(is_new)))
        with work(ledger, "selection" if scope == "search" else "audit", scope):
            model = (
                prepared_model
                if prepared_model is not None
                else backend.assembly(state, cache)
            )
            if prepared_model is not None:
                snapshot = model.snapshot(state)
                if snapshot.phi != state.phi or snapshot.beta != state.beta:
                    raise ValueError(
                        "Prepared evaluation model differs from its complete checkpoint"
                    )
            score = backend.score(model, split)
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid complete-model score")
        if scope == "search" and is_new:
            ledger.charge("unique_selection_completion", Cost(E=1), evaluation_key=key)
        self.scores[key] = score
        self.records.append(
            {"key": key, "state": state.hash, "role": split.role, "reuse": False}
        )
        return score, False
