"""One replayable geometric view per original per logical fitting epoch."""

import hashlib
import math
import numpy as np
from PIL import Image

VIEWS = ("original", "hflip", "rotate_minus5", "rotate_plus5")


def view_for(identity, seed, epoch):
    if type(epoch) is not int or epoch < 0 or type(seed) is not int:
        raise ValueError("Integer seed/nonnegative epoch required")
    offset = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "big"
        )
        % 4
    )
    return VIEWS[(offset + epoch) % 4]


def _clip(poly, w, h):
    for axis, bound, sign in ((0, 0, 1), (0, w, -1), (1, 0, 1), (1, h, -1)):
        out = []
        for a, b in zip(poly, poly[1:] + poly[:1]):
            ina = (a[axis] - bound) * sign >= 0
            inb = (b[axis] - bound) * sign >= 0
            if ina:
                out.append(a)
            if ina != inb:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        poly = out
        if not poly:
            break
    return poly


def apply_view(pixels, boxes, view):
    """Float RGB [0,1], normalized class/cx/cy/w/h. Clip rotated polygons first."""
    if view not in VIEWS:
        raise ValueError("Unknown view")
    p = np.asarray(pixels, dtype=np.float32)
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if (
        p.ndim != 3
        or p.shape[2] != 3
        or not np.isfinite(p).all()
        or p.min() < 0
        or p.max() > 1
    ):
        raise ValueError("Expected finite float RGB in [0,1]")
    if not np.isfinite(b).all() or (
        len(b)
        and ((b[:, 1:] < 0).any() or (b[:, 1:] > 1).any() or (b[:, 3:] <= 0).any())
    ):
        raise ValueError("Invalid boxes")
    if view == "original":
        return p.copy(), b.copy()
    h, w = p.shape[:2]
    if view == "hflip":
        b = b.copy()
        b[:, 1] = 1 - b[:, 1]
        return p[:, ::-1].copy(), b
    angle = -5 if view == "rotate_minus5" else 5
    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))

    image = np.stack(
        [
            np.asarray(
                Image.fromarray(p[:, :, k]).rotate(
                    angle, resample=Image.Resampling.BILINEAR, fillcolor=114 / 255
                )
            )
            for k in range(3)
        ],
        axis=-1,
    )
    out = []
    for cls, cx, cy, bw, bh in b:
        x0, x1 = (cx - bw / 2) * w, (cx + bw / 2) * w
        y0, y1 = (cy - bh / 2) * h, (cy + bh / 2) * h
        poly = [
            (
                w / 2 + c * (x - w / 2) + s * (y - h / 2),
                h / 2 - s * (x - w / 2) + c * (y - h / 2),
            )
            for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        ]
        poly = _clip(poly, w, h)
        if len(poly) < 3:
            continue
        area = (
            abs(sum(x * v - y * u for (x, y), (u, v) in zip(poly, poly[1:] + poly[:1])))
            / 2
        )
        if area <= 1e-8:
            continue
        lo = np.min(poly, axis=0)
        hi = np.max(poly, axis=0)
        out.append(
            [
                cls,
                (lo[0] + hi[0]) / (2 * w),
                (lo[1] + hi[1]) / (2 * h),
                (hi[0] - lo[0]) / w,
                (hi[1] - lo[1]) / h,
            ]
        )
    return image.astype(np.float32), np.asarray(out, dtype=np.float32).reshape(-1, 5)
