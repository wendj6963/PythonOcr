from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .yolo_backend import predict


@dataclass(frozen=True)
class OcrBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _extract_boxes(pred: Any) -> list[OcrBox]:
    # Ultralytics returns a list-like Results
    results = pred
    if isinstance(results, (list, tuple)):
        results = results[0]

    boxes = getattr(results, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

    out: list[OcrBox] = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        out.append(
            OcrBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                conf=float(conf[i]),
                cls=int(cls[i]),
            )
        )
    return out


def _extract_polys_norm(pred: Any, img_w: int, img_h: int) -> list[dict]:
    """Extract polygons (normalized 8 points) from Ultralytics prediction if available.

    We try OBB first (results.obb), then fallback to boxes.xyxy to create a rectangular poly.
    Returned list items: {"conf": float, "cls": int, "poly": [x1,y1,...,x4,y4]} with x/y normalized.
    """

    results = pred
    if isinstance(results, (list, tuple)):
        results = results[0]

    out: list[dict] = []

    obb = getattr(results, "obb", None)
    if obb is not None and hasattr(obb, "xyxyxyxy"):
        polys = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else np.array(obb.xyxyxyxy)
        conf = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else np.array(obb.conf)
        cls = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else np.array(obb.cls)
        for i in range(len(polys)):
            pts = polys[i].reshape(4, 2)
            ptsn: list[float] = []
            for (x, y) in pts.tolist():
                ptsn.append(float(x) / float(img_w))
                ptsn.append(float(y) / float(img_h))
            out.append({"conf": float(conf[i]), "cls": int(cls[i]), "poly": ptsn})
        return out

    # fallback: axis-aligned boxes
    boxes = getattr(results, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        ptsn = [
            float(x1) / float(img_w),
            float(y1) / float(img_h),
            float(x2) / float(img_w),
            float(y1) / float(img_h),
            float(x2) / float(img_w),
            float(y2) / float(img_h),
            float(x1) / float(img_w),
            float(y2) / float(img_h),
        ]
        out.append({"conf": float(conf[i]), "cls": int(cls[i]), "poly": ptsn})
    return out


def _sort_left_to_right(boxes: list[OcrBox]) -> list[OcrBox]:
    return sorted(boxes, key=lambda b: (b.x1 + b.x2) / 2.0)


def load_vocab(vocab_path: Path | None) -> dict[int, str]:
    """Load class-id to character mapping.

    File format (UTF-8): one character per line, line index = class id.
    Example:
      0
      1
      2
      A
      B
    """

    if vocab_path is None:
        return {}
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab 文件不存在: {vocab_path}")
    mapping: dict[int, str] = {}
    for i, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()):
        s = line.strip("\r\n")
        if not s:
            continue
        mapping[i] = s
    return mapping


def infer_ocr(
    det_weights: Path,
    rec_weights: Path,
    image: Path,
    device: str = "cpu",
    vocab: Path | None = None,
) -> dict:
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image}")

    h, w = img.shape[:2]

    det_pred = predict(det_weights, image=image, imgsz=640, device=device)
    det_boxes = _sort_left_to_right(_extract_boxes(det_pred))
    det_polys = _extract_polys_norm(det_pred, img_w=w, img_h=h)

    vocab_map = load_vocab(vocab)

    # Recognition: detection + class on each ROI.
    chars: list[dict] = []
    for b in det_boxes:
        x1 = _clip(b.x1, 0, w - 1)
        y1 = _clip(b.y1, 0, h - 1)
        x2 = _clip(b.x2, 0, w)
        y2 = _clip(b.y2, 0, h)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = img[y1:y2, x1:x2]

        # Ultralytics predict accepts numpy arrays
        rec_pred = predict(rec_weights, image=roi, imgsz=320, device=device)
        rec_boxes = _sort_left_to_right(_extract_boxes(rec_pred))
        rec_polys = _extract_polys_norm(rec_pred, img_w=max(1, roi.shape[1]), img_h=max(1, roi.shape[0]))

        roi_chars: list[dict] = []
        for rb in rec_boxes:
            ch = vocab_map.get(rb.cls, str(rb.cls))
            roi_chars.append(
                {
                    "char": ch,
                    "cls": rb.cls,
                    "conf": rb.conf,
                    "box": {"x1": rb.x1, "y1": rb.y1, "x2": rb.x2, "y2": rb.y2},
                }
            )

        chars.append(
            {
                "det_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": b.conf},
                "det_poly": None,
                "roi_chars": roi_chars,
                "roi_polys": rec_polys,
            }
        )

    # attach det poly by matching to det_box center (best-effort)
    for region in chars:
        db = region.get("det_box", {})
        cx = (float(db.get("x1", 0)) + float(db.get("x2", 0))) / 2.0
        cy = (float(db.get("y1", 0)) + float(db.get("y2", 0))) / 2.0
        best = None
        best_d = 1e18
        for p in det_polys:
            poly = p.get("poly")
            if not poly or len(poly) != 8:
                continue
            # poly center in px
            px = [(poly[i] * w, poly[i + 1] * h) for i in range(0, 8, 2)]
            pcx = sum([t[0] for t in px]) / 4.0
            pcy = sum([t[1] for t in px]) / 4.0
            d = (pcx - cx) * (pcx - cx) + (pcy - cy) * (pcy - cy)
            if d < best_d:
                best_d = d
                best = poly
        region["det_poly"] = best

    text = "".join([c["char"] for roi in chars for c in roi["roi_chars"]])
    # for visualization convenience
    boxes_vis = []
    for region in chars:
        if region.get("det_poly"):
            boxes_vis.append({"poly": region["det_poly"], "cls": 0, "conf": float(region.get("det_box", {}).get("conf", 0.0))})
    return {"image": str(image), "text": text, "regions": chars, "boxes": boxes_vis}


def save_json(obj: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

