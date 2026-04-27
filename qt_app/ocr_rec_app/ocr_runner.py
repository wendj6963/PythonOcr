from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.pyocr.pipeline import load_vocab, _extract_boxes, _sort_left_to_right
from src.pyocr.yolo_backend import predict


# OCR 推理参数（由 UI 传入）
@dataclass(frozen=True)
class OcrParams:
    det_conf: float = 0.25
    det_iou: float = 0.7
    det_max_det: int = 1
    rec_conf: float = 0.25
    rec_iou: float = 0.7
    det_imgsz: int = 640
    rec_imgsz: int = 320
    rec_backend: str = "yolo"
    use_gpu: bool = False
    num_threads: int = 0
    vocab_path: Path | None = None
    sort_by: str = "x"
    mock_mode: bool = False
    rec_top_n: int = 10
    rec_min_score: float = 0.5
    rec_flip_enable: bool = False
    rec_flip_min_score: float = 0.5
    roi_pad_ratio: float = 0.03
    roi_pad_px: int = 2
    rec_min_box: int = 4
    rec_row_thresh: float = 0.6
    rec_auto_imgsz: bool = False
    rec_imgsz_min: int = 160
    rec_imgsz_max: int = 640
    roi_expected_lengths: tuple[int, ...] | None = None


# 单个识别结果框（矩形 + 文本）
@dataclass(frozen=True)
class OcrBox:
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    score: float
    quad: tuple[float, ...] | None = None
    roi_index: int | None = None


# 推理结果汇总
@dataclass(frozen=True)
class OcrResult:
    boxes: list[OcrBox]
    det_time_ms: float
    rec_time_ms: float
    total_time_ms: float
    mean_score: float
    roi_previews: list[np.ndarray]


# OBB 结果兜底：将四点框转换为矩形框
def _extract_boxes_fallback(pred: Any, img_w: int, img_h: int) -> list[Any]:
    """Fallback for OBB results: create axis-aligned boxes from quadrangles."""
    results = pred
    if isinstance(results, (list, tuple)):
        results = results[0]
    obb = getattr(results, "obb", None)
    if obb is None or not hasattr(obb, "xyxyxyxy"):
        return []

    polys = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else np.array(obb.xyxyxyxy)
    conf = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else np.array(obb.conf)
    cls = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else np.array(obb.cls)

    out = []
    for i in range(len(polys)):
        pts = polys[i].reshape(4, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        out.append(
            type("Tmp", (), {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(conf[i]), "cls": int(cls[i])})
        )
    return out


# 提取 OBB 四点框（像素坐标）
def _extract_obb_polys(pred: Any) -> list[dict[str, Any]]:
    results = pred
    if isinstance(results, (list, tuple)):
        results = results[0]
    obb = getattr(results, "obb", None)
    if obb is None or not hasattr(obb, "xyxyxyxy"):
        return []

    polys = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else np.array(obb.xyxyxyxy)
    conf = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else np.array(obb.conf)
    cls = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else np.array(obb.cls)

    out: list[dict[str, Any]] = []
    for i in range(len(polys)):
        pts = polys[i].reshape(4, 2).astype(float).tolist()
        out.append({"poly": pts, "conf": float(conf[i]), "cls": int(cls[i])})
    return out


# 四点排序为左上、右上、右下、左下
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视矫正并裁剪 ROI
def _warp_quad(img: np.ndarray, quad: list[list[float]]) -> np.ndarray:
    import cv2

    pts = np.array(quad, dtype=np.float32)
    rect = _order_points(pts)
    w1 = np.linalg.norm(rect[0] - rect[1])
    w2 = np.linalg.norm(rect[2] - rect[3])
    h1 = np.linalg.norm(rect[0] - rect[3])
    h2 = np.linalg.norm(rect[1] - rect[2])
    width = max(int(round(w1)), int(round(w2)), 1)
    height = max(int(round(h1)), int(round(h2)), 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, m, (width, height))


def _expand_quad(quad: list[list[float]], pad_ratio: float, pad_px: int) -> list[list[float]]:
    pts = np.array(quad, dtype=np.float32)
    rect = _order_points(pts)
    w1 = np.linalg.norm(rect[0] - rect[1])
    w2 = np.linalg.norm(rect[2] - rect[3])
    h1 = np.linalg.norm(rect[0] - rect[3])
    h2 = np.linalg.norm(rect[1] - rect[2])
    width = max(float(max(w1, w2)), 1.0)
    height = max(float(max(h1, h2)), 1.0)
    pad = max(width, height) * float(max(pad_ratio, 0.0)) + float(max(pad_px, 0)) * 2.0
    scale = 1.0 + pad / max(width, height)
    center = rect.mean(axis=0)
    expanded = (rect - center) * scale + center
    return expanded.astype(float).tolist()


def _pad_axis_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_ratio: float, pad_px: int) -> tuple[int, int, int, int]:
    pad = int(round(max(w, h) * float(max(pad_ratio, 0.0))))
    pad = max(pad, int(max(pad_px, 0)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


# 按 x 或 y 方向排序检测框
def _sort_boxes_line(boxes: list[Any], row_thresh: float) -> list[Any]:
    if not boxes:
        return []
    heights = [max(1.0, float(b.y2) - float(b.y1)) for b in boxes]
    avg_h = float(np.mean(heights)) if heights else 1.0
    threshold = max(1.0, avg_h * float(max(row_thresh, 0.0)))
    sorted_by_y = sorted(boxes, key=lambda b: (float(b.y1) + float(b.y2)) / 2.0)

    rows: list[list[Any]] = []
    for b in sorted_by_y:
        cy = (float(b.y1) + float(b.y2)) / 2.0
        placed = False
        for row in rows:
            rcy = (float(row[0].y1) + float(row[0].y2)) / 2.0
            if abs(cy - rcy) <= threshold:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    out: list[Any] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: (float(b.x1) + float(b.x2)) / 2.0))
    return out


def _sort_boxes(boxes: list[Any], sort_by: str, row_thresh: float = 0.6) -> list[Any]:
    if sort_by.lower() == "y":
        return sorted(boxes, key=lambda b: (b.y1 + b.y2) / 2.0)
    if sort_by.lower() == "line":
        return _sort_boxes_line(boxes, row_thresh)
    return _sort_left_to_right(boxes)


def _sort_polys(polys: list[dict[str, Any]], sort_by: str, row_thresh: float) -> list[dict[str, Any]]:
    if not polys:
        return []
    tmps: list[tuple[dict[str, Any], Any]] = []
    for p in polys:
        quad = p.get("poly") or []
        if len(quad) != 4:
            continue
        xs = [t[0] for t in quad]
        ys = [t[1] for t in quad]
        x1, x2 = float(min(xs)), float(max(xs))
        y1, y2 = float(min(ys)), float(max(ys))
        tmp = type("Tmp", (), {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        tmps.append((p, tmp))
    if not tmps:
        return []
    sorted_tmps = _sort_boxes([t[1] for t in tmps], sort_by, row_thresh)
    tmp_to_poly = {id(t[1]): t[0] for t in tmps}
    sorted_polys: list[dict[str, Any]] = []
    for tb in sorted_tmps:
        p = tmp_to_poly.get(id(tb))
        if p is not None:
            sorted_polys.append(p)
    return sorted_polys


def _filter_rec_boxes(
    boxes: list[Any],
    sort_by: str,
    min_score: float,
    top_n: int,
    min_box: int,
    row_thresh: float,
) -> list[Any]:
    filtered = [
        b
        for b in boxes
        if float(getattr(b, "conf", 0.0)) >= min_score
        and (float(b.x2) - float(b.x1)) >= float(min_box)
        and (float(b.y2) - float(b.y1)) >= float(min_box)
    ]
    filtered.sort(key=lambda b: float(getattr(b, "conf", 0.0)), reverse=True)
    if top_n > 0:
        filtered = filtered[:top_n]
    return _sort_boxes(filtered, sort_by, row_thresh)


def _rec_text_from_boxes(boxes: list[Any], vocab_map: dict[int, str]) -> tuple[list[str], list[float]]:
    text_parts: list[str] = []
    scores: list[float] = []
    for rb in boxes:
        text_parts.append(vocab_map.get(rb.cls, str(rb.cls)))
        scores.append(float(rb.conf))
    return text_parts, scores


# 将参数转换为推理设备字符串
def _device_from_params(params: OcrParams) -> str:
    return "0" if params.use_gpu else "cpu"


# 设置 CPU 线程数（可选）
def _apply_threads(num_threads: int) -> None:
    if num_threads <= 0:
        return
    try:
        import torch

        torch.set_num_threads(num_threads)
    except Exception:
        pass


# mock 模式：生成随机框和文本用于 UI 测试
def _mock_boxes(w: int, h: int, vocab: dict[int, str]) -> list[OcrBox]:
    rng = random.Random(0)
    out: list[OcrBox] = []
    chars = list(vocab.values()) or list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i in range(3):
        x1 = rng.randint(0, max(1, w - 80))
        y1 = rng.randint(0, max(1, h - 40))
        x2 = min(w - 1, x1 + rng.randint(40, 120))
        y2 = min(h - 1, y1 + rng.randint(20, 60))
        text = "".join(rng.choice(chars) for _ in range(rng.randint(1, 4)))
        out.append(OcrBox(float(x1), float(y1), float(x2), float(y2), text, rng.uniform(0.5, 0.99), roi_index=i + 1))
    return out


# 兼容中文路径的图片读取
def _read_image_unicode(path: Path) -> np.ndarray:
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def _choose_rec_imgsz(roi: np.ndarray, params: OcrParams) -> int:
    if not params.rec_auto_imgsz:
        return int(params.rec_imgsz)
    h, w = roi.shape[:2]
    side = int(max(h, w))
    side = max(int(params.rec_imgsz_min), min(int(params.rec_imgsz_max), side))
    return max(32, int(round(side / 32.0)) * 32)


def _expected_len_for_roi(params: OcrParams, roi_index: int) -> int | None:
    if not params.roi_expected_lengths:
        return None
    idx = int(roi_index) - 1
    if idx < 0 or idx >= len(params.roi_expected_lengths):
        return None
    val = int(params.roi_expected_lengths[idx])
    return val if val > 0 else None


def _infer_roi_text_yolo(
    roi: np.ndarray,
    rec_model: Path,
    device: str,
    params: OcrParams,
    vocab_map: dict[int, str],
    expected_len: int | None,
    min_score: float,
    flip_min_score: float,
) -> tuple[str, float]:
    rec_imgsz = _choose_rec_imgsz(roi, params)
    rec_pred = predict(
        rec_model,
        image=roi,
        imgsz=rec_imgsz,
        device=device,
        conf=params.rec_conf,
        iou=params.rec_iou,
    )
    rec_boxes = _extract_boxes(rec_pred)
    if not rec_boxes:
        rec_boxes = _extract_boxes_fallback(rec_pred, max(1, roi.shape[1]), max(1, roi.shape[0]))

    per_top_n = int(expected_len) if expected_len is not None else int(params.rec_top_n)
    rec_boxes = _filter_rec_boxes(
        rec_boxes,
        params.sort_by,
        min_score,
        per_top_n,
        params.rec_min_box,
        params.rec_row_thresh,
    )
    text_parts, scores = _rec_text_from_boxes(rec_boxes, vocab_map)

    if expected_len is not None and len(text_parts) > expected_len:
        text_parts = text_parts[:expected_len]
        scores = scores[:expected_len]

    if params.rec_flip_enable:
        roi_flip = np.ascontiguousarray(np.flip(roi, (0, 1)))
        rec_imgsz_flip = _choose_rec_imgsz(roi_flip, params)
        rec_pred_flip = predict(
            rec_model,
            image=roi_flip,
            imgsz=rec_imgsz_flip,
            device=device,
            conf=params.rec_conf,
            iou=params.rec_iou,
        )
        flip_boxes = _extract_boxes(rec_pred_flip)
        if not flip_boxes:
            flip_boxes = _extract_boxes_fallback(rec_pred_flip, max(1, roi_flip.shape[1]), max(1, roi_flip.shape[0]))
        flip_boxes = _filter_rec_boxes(
            flip_boxes,
            params.sort_by,
            flip_min_score,
            per_top_n,
            params.rec_min_box,
            params.rec_row_thresh,
        )
        flip_text, flip_scores = _rec_text_from_boxes(flip_boxes, vocab_map)
        if expected_len is not None and len(flip_text) > expected_len:
            flip_text = flip_text[:expected_len]
            flip_scores = flip_scores[:expected_len]

        base_mean = float(np.mean(scores)) if scores else 0.0
        flip_mean = float(np.mean(flip_scores)) if flip_scores else 0.0
        if flip_mean > base_mean and flip_text:
            text_parts = flip_text
            scores = flip_scores

    text = "".join(text_parts)
    score = float(np.mean(scores)) if scores else 0.0
    return text, score


def _infer_roi_text_ctc(
    roi: np.ndarray,
    rec_model: Path,
    device: str,
    params: OcrParams,
    expected_len: int | None,
) -> tuple[str, float]:
    from src.pyocr import rec_ctc_backend

    rec = rec_ctc_backend.infer_ctc_array(
        weights=rec_model,
        image_bgr=roi,
        img_h=48,
        img_w=max(64, int(params.rec_imgsz)),
        device=device,
    )
    text = str(rec.get("text", ""))
    score = float(rec.get("score", 0.0))

    if expected_len is not None and len(text) > expected_len:
        text = text[:expected_len]

    if params.rec_flip_enable:
        roi_flip = np.ascontiguousarray(np.flip(roi, (0, 1)))
        rec_flip = rec_ctc_backend.infer_ctc_array(
            weights=rec_model,
            image_bgr=roi_flip,
            img_h=48,
            img_w=max(64, int(params.rec_imgsz)),
            device=device,
        )
        flip_text = str(rec_flip.get("text", ""))
        if expected_len is not None and len(flip_text) > expected_len:
            flip_text = flip_text[:expected_len]
        flip_score = float(rec_flip.get("score", 0.0))
        if flip_score > score:
            text, score = flip_text, flip_score

    return text, score


# 入口：执行 det + rec 推理，返回可视化结果结构
def run_ocr(image: Path, det_model: Path, rec_model: Path, params: OcrParams) -> OcrResult:
    _apply_threads(params.num_threads)
    device = _device_from_params(params)

    img = _read_image_unicode(image)
    h, w = img.shape[:2]

    vocab_map = load_vocab(params.vocab_path) if params.vocab_path else {}

    if params.mock_mode:
        boxes = _mock_boxes(w, h, vocab_map)
        mean_score = float(np.mean([b.score for b in boxes])) if boxes else 0.0
        return OcrResult(
            boxes=boxes,
            det_time_ms=0.0,
            rec_time_ms=0.0,
            total_time_ms=0.0,
            mean_score=mean_score,
            roi_previews=[],
        )

    det_start = time.perf_counter()
    det_pred = predict(
        det_model,
        image=image,
        imgsz=params.det_imgsz,
        device=device,
        conf=params.det_conf,
        iou=params.det_iou,
        max_det=params.det_max_det,
    )
    det_polys = _extract_obb_polys(det_pred)
    det_boxes = _sort_boxes(_extract_boxes(det_pred), params.sort_by, params.rec_row_thresh)
    if not det_boxes:
        det_boxes = _sort_boxes(_extract_boxes_fallback(det_pred, w, h), params.sort_by, params.rec_row_thresh)
    det_time_ms = (time.perf_counter() - det_start) * 1000.0

    rec_start = time.perf_counter()
    out_boxes: list[OcrBox] = []
    roi_previews: list[np.ndarray] = []

    min_score = max(params.rec_conf, params.rec_min_score)
    flip_min_score = max(params.rec_conf, params.rec_flip_min_score)
    rec_backend = str(params.rec_backend or "yolo").strip().lower()

    if det_polys:
        det_polys = _sort_polys(det_polys, params.sort_by, params.rec_row_thresh)
        for roi_idx, p in enumerate(det_polys, start=1):
            conf = float(p.get("conf", 0.0))
            if conf < params.det_conf:
                continue
            quad = p.get("poly") or []
            if len(quad) != 4:
                continue
            xs = [t[0] for t in quad]
            ys = [t[1] for t in quad]
            x1, x2 = int(max(0, min(w - 1, min(xs)))), int(max(0, min(w, max(xs))))
            y1, y2 = int(max(0, min(h - 1, min(ys)))), int(max(0, min(h, max(ys))))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = _warp_quad(img, quad)
            if params.roi_pad_ratio > 0 or params.roi_pad_px > 0:
                quad_roi = _expand_quad(quad, params.roi_pad_ratio, params.roi_pad_px)
                roi = _warp_quad(img, quad_roi)
            roi_previews.append(roi)

            expected_len = _expected_len_for_roi(params, roi_idx)
            if rec_backend == "ctc":
                text, score = _infer_roi_text_ctc(roi, rec_model, device, params, expected_len)
            else:
                text, score = _infer_roi_text_yolo(
                    roi,
                    rec_model,
                    device,
                    params,
                    vocab_map,
                    expected_len,
                    min_score,
                    flip_min_score,
                )
            out_boxes.append(
                OcrBox(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    text,
                    score,
                    quad=tuple([v for pt in quad for v in pt]),
                    roi_index=int(roi_idx),
                )
            )
    else:
        for roi_idx, b in enumerate(det_boxes, start=1):
            if b.conf < params.det_conf:
                continue
            x1 = int(max(0, min(w - 1, b.x1)))
            y1 = int(max(0, min(h - 1, b.y1)))
            x2 = int(max(0, min(w, b.x2)))
            y2 = int(max(0, min(h, b.y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            x1, y1, x2, y2 = _pad_axis_roi(x1, y1, x2, y2, w, h, params.roi_pad_ratio, params.roi_pad_px)
            roi = img[y1:y2, x1:x2]
            roi_previews.append(roi)
            expected_len = _expected_len_for_roi(params, roi_idx)
            if rec_backend == "ctc":
                text, score = _infer_roi_text_ctc(roi, rec_model, device, params, expected_len)
            else:
                text, score = _infer_roi_text_yolo(
                    roi,
                    rec_model,
                    device,
                    params,
                    vocab_map,
                    expected_len,
                    min_score,
                    flip_min_score,
                )
            out_boxes.append(
                OcrBox(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    text,
                    score,
                    roi_index=int(roi_idx),
                )
            )

    rec_time_ms = (time.perf_counter() - rec_start) * 1000.0

    total_time_ms = det_time_ms + rec_time_ms
    mean_score = float(np.mean([b.score for b in out_boxes])) if out_boxes else 0.0
    return OcrResult(
        boxes=out_boxes,
        det_time_ms=det_time_ms,
        rec_time_ms=rec_time_ms,
        total_time_ms=total_time_ms,
        mean_score=mean_score,
        roi_previews=roi_previews,
    )
