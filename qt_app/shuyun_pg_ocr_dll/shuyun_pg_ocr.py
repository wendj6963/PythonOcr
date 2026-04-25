from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from qt_app.ocr_rec_app.ocr_runner import OcrParams, OcrResult, run_ocr

ROOT_DIR = Path(__file__).resolve().parents[2]

# =========================
# 默认路径与参数（与测试工具一致）
# =========================

# 默认模型路径（可改为绝对路径）
DEFAULT_DET_MODEL = Path("det.pt")
DEFAULT_REC_MODEL = Path("rec.pt")
DEFAULT_VOCAB_PATH = Path("vocab_rec.txt")

# 定位默认参数
DEFAULT_DET_CONF = 0.25
DEFAULT_DET_IOU = 0.7
DEFAULT_DET_IMGSZ = 640
DEFAULT_DET_MAX_DET = 3

# 识别默认参数
DEFAULT_REC_CONF = 0.25
DEFAULT_REC_IOU = 0.7
DEFAULT_REC_IMGSZ = 320
DEFAULT_REC_AUTO_IMGSZ = True
DEFAULT_REC_IMGSZ_MIN = 60
DEFAULT_REC_IMGSZ_MAX = 640
DEFAULT_REC_TOP_N = 14
DEFAULT_REC_MIN_SCORE = 0.5
DEFAULT_REC_FLIP_ENABLE = False
DEFAULT_REC_FLIP_MIN_SCORE = 0.5
DEFAULT_REC_MIN_BOX = 4
DEFAULT_REC_ROW_THRESH = 0.6
DEFAULT_ROI_EXPECTED_LENGTHS = "8,14,2"

# 公共默认参数
DEFAULT_USE_GPU = False
DEFAULT_NUM_THREADS = 0
DEFAULT_SORT_BY = "x"
DEFAULT_MOCK_MODE = False
DEFAULT_ROI_PAD_RATIO = 0.06
DEFAULT_ROI_PAD_PX = 4


# =========================
# 结果结构
# =========================

@dataclass(frozen=True)
class OcrRoiResult:
    """单个 ROI 的识别结果。"""

    roi_index: int
    rect: tuple[int, int, int, int]
    text: str
    score: float
    quad: tuple[float, ...] | None
    roi_image_path: str | None


@dataclass(frozen=True)
class OcrDecodeResult:
    """整张图片 OCR 结果。"""

    image_path: str
    det_model_path: str
    rec_model_path: str
    decode_image_path: str | None
    det_time_ms: float
    rec_time_ms: float
    total_time_ms: float
    mean_score: float
    rois: list[OcrRoiResult]


# =========================
# 参数封装（带默认值）
# =========================

def build_det_params(
    det_conf: float = DEFAULT_DET_CONF,
    det_iou: float = DEFAULT_DET_IOU,
    det_imgsz: int = DEFAULT_DET_IMGSZ,
    det_max_det: int = DEFAULT_DET_MAX_DET,
) -> dict[str, Any]:
    """构建定位模型参数（带默认值）。"""

    return {
        "det_conf": float(det_conf),
        "det_iou": float(det_iou),
        "det_imgsz": int(det_imgsz),
        "det_max_det": int(det_max_det),
    }


def build_rec_params(
    rec_conf: float = DEFAULT_REC_CONF,
    rec_iou: float = DEFAULT_REC_IOU,
    rec_imgsz: int = DEFAULT_REC_IMGSZ,
    rec_auto_imgsz: bool = DEFAULT_REC_AUTO_IMGSZ,
    rec_imgsz_min: int = DEFAULT_REC_IMGSZ_MIN,
    rec_imgsz_max: int = DEFAULT_REC_IMGSZ_MAX,
    rec_top_n: int = DEFAULT_REC_TOP_N,
    rec_min_score: float = DEFAULT_REC_MIN_SCORE,
    rec_flip_enable: bool = DEFAULT_REC_FLIP_ENABLE,
    rec_flip_min_score: float = DEFAULT_REC_FLIP_MIN_SCORE,
    rec_min_box: int = DEFAULT_REC_MIN_BOX,
    rec_row_thresh: float = DEFAULT_REC_ROW_THRESH,
    roi_expected_lengths: str = DEFAULT_ROI_EXPECTED_LENGTHS,
) -> dict[str, Any]:
    """构建识别模型参数（带默认值）。"""

    return {
        "rec_conf": float(rec_conf),
        "rec_iou": float(rec_iou),
        "rec_imgsz": int(rec_imgsz),
        "rec_auto_imgsz": bool(rec_auto_imgsz),
        "rec_imgsz_min": int(rec_imgsz_min),
        "rec_imgsz_max": int(rec_imgsz_max),
        "rec_top_n": int(rec_top_n),
        "rec_min_score": float(rec_min_score),
        "rec_flip_enable": bool(rec_flip_enable),
        "rec_flip_min_score": float(rec_flip_min_score),
        "rec_min_box": int(rec_min_box),
        "rec_row_thresh": float(rec_row_thresh),
        "roi_expected_lengths": _parse_roi_expected_lengths(roi_expected_lengths),
    }


def build_common_params(
    use_gpu: bool = DEFAULT_USE_GPU,
    num_threads: int = DEFAULT_NUM_THREADS,
    sort_by: str = DEFAULT_SORT_BY,
    mock_mode: bool = DEFAULT_MOCK_MODE,
    roi_pad_ratio: float = DEFAULT_ROI_PAD_RATIO,
    roi_pad_px: int = DEFAULT_ROI_PAD_PX,
    vocab_path: Path | str | None = DEFAULT_VOCAB_PATH,
) -> dict[str, Any]:
    """构建通用参数（带默认值）。"""

    vocab = Path(vocab_path) if vocab_path else None
    return {
        "use_gpu": bool(use_gpu),
        "num_threads": int(num_threads),
        "sort_by": str(sort_by),
        "mock_mode": bool(mock_mode),
        "roi_pad_ratio": float(roi_pad_ratio),
        "roi_pad_px": int(roi_pad_px),
        "vocab_path": vocab,
    }


def build_ocr_params(
    det_params: dict[str, Any] | None = None,
    rec_params: dict[str, Any] | None = None,
    common_params: dict[str, Any] | None = None,
) -> OcrParams:
    """合并参数并生成 OcrParams。"""

    merged: dict[str, Any] = {}
    if det_params:
        merged.update(det_params)
    if rec_params:
        merged.update(rec_params)
    if common_params:
        merged.update(common_params)
    return OcrParams(**merged)


# =========================
# OCR 识别主入口
# =========================

def run_ocr_by_path(
    image_path: str | Path,
    det_model: str | Path = DEFAULT_DET_MODEL,
    rec_model: str | Path = DEFAULT_REC_MODEL,
    params: OcrParams | None = None,
    save_dir: str | Path | None = None,
) -> OcrDecodeResult:
    """根据图片路径执行 OCR，并保存回显图与 ROI 图片。"""

    img_path = Path(image_path)
    det_path = Path(det_model)
    rec_path = Path(rec_model)
    if params is None:
        params = build_ocr_params(build_det_params(), build_rec_params(), build_common_params())

    if not img_path.exists():
        raise FileNotFoundError(f"找不到图片: {img_path}")
    if not det_path.exists():
        raise FileNotFoundError(f"找不到定位模型: {det_path}")
    if not rec_path.exists():
        raise FileNotFoundError(f"找不到识别模型: {rec_path}")

    result = run_ocr(img_path, det_path, rec_path, params)
    output_dir = Path(save_dir) if save_dir else ROOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    decode_img_path = output_dir / "decode_img.bmp"
    roi_paths: list[str] = []

    _save_decode_image(img_path, result, decode_img_path)

    for idx, roi in enumerate(result.roi_previews, start=1):
        roi_path = output_dir / f"roi_{idx}.png"
        _save_roi_image(roi, roi_path)
        roi_paths.append(str(roi_path))

    rois: list[OcrRoiResult] = []
    for i, box in enumerate(result.boxes, start=1):
        roi_index = int(box.roi_index or i)
        roi_path = roi_paths[roi_index - 1] if 0 < roi_index <= len(roi_paths) else None
        rois.append(
            OcrRoiResult(
                roi_index=roi_index,
                rect=(int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
                text=str(box.text),
                score=float(box.score),
                quad=box.quad,
                roi_image_path=roi_path,
            )
        )

    return OcrDecodeResult(
        image_path=str(img_path),
        det_model_path=str(det_path),
        rec_model_path=str(rec_path),
        decode_image_path=str(decode_img_path),
        det_time_ms=float(result.det_time_ms),
        rec_time_ms=float(result.rec_time_ms),
        total_time_ms=float(result.total_time_ms),
        mean_score=float(result.mean_score),
        rois=rois,
    )


def run_ocr_by_path_json(
    image_path: str | Path,
    det_model: str | Path = DEFAULT_DET_MODEL,
    rec_model: str | Path = DEFAULT_REC_MODEL,
    params: OcrParams | None = None,
    save_dir: str | Path | None = None,
) -> str:
    """执行 OCR 并返回 JSON 字符串（便于 C# 调用）。"""

    result = run_ocr_by_path(image_path, det_model, rec_model, params, save_dir)
    payload = _result_to_dict(result)
    return json.dumps(payload, ensure_ascii=True)


def _result_to_dict(result: OcrDecodeResult) -> dict[str, Any]:
    """将识别结果转换为可 JSON 序列化的字典。"""

    return {
        "image_path": result.image_path,
        "det_model_path": result.det_model_path,
        "rec_model_path": result.rec_model_path,
        "decode_image_path": result.decode_image_path,
        "det_time_ms": result.det_time_ms,
        "rec_time_ms": result.rec_time_ms,
        "total_time_ms": result.total_time_ms,
        "mean_score": result.mean_score,
        "rois": [
            {
                "roi_index": roi.roi_index,
                "rect": list(roi.rect),
                "text": roi.text,
                "score": roi.score,
                "quad": list(roi.quad) if roi.quad else None,
                "roi_image_path": roi.roi_image_path,
            }
            for roi in result.rois
        ],
    }


# =========================
# 工具函数
# =========================

def _parse_roi_expected_lengths(value: str | None) -> tuple[int, ...] | None:
    """解析 ROI 期望长度配置（如: "8,14,2"）。"""

    if not value:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            continue
    return tuple(out) if out else None


def _read_image_unicode(path: Path) -> np.ndarray:
    """使用 imdecode 读取图片，兼容中文路径。"""

    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def _save_roi_image(roi: np.ndarray, path: Path) -> None:
    """保存 ROI 图片到本地。"""

    import cv2

    if roi is None or roi.size == 0:
        return
    cv2.imwrite(str(path), roi)


def _save_decode_image(image_path: Path, result: OcrResult, out_path: Path) -> None:
    """将定位框与识别结果绘制到原图并保存。"""

    import cv2

    img = _read_image_unicode(image_path)
    for i, box in enumerate(result.boxes, start=1):
        color = (0, 255, 0)
        thickness = 2
        if box.quad and len(box.quad) == 8:
            pts = np.array(box.quad, dtype=np.float32).reshape(4, 2)
            pts = pts.astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = f"{box.roi_index or i}:{box.text}" if box.text else f"{box.roi_index or i}"
        text_pos = (max(0, x1), max(0, y1 - 5))
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(str(out_path), img)


# =========================
# 调用测试案例
# =========================

def demo() -> None:
    """示例：读取一张图片，执行 OCR 并输出结果。"""

    sample_img = Path(r"C:\Users\admin\Desktop\08#\2_6_190608768.bmp")
    if not sample_img.exists():
        raise FileNotFoundError(f"未找到示例图片: {sample_img}")

    params = build_ocr_params(build_det_params(), build_rec_params(), build_common_params())
    result = run_ocr_by_path(sample_img, DEFAULT_DET_MODEL, DEFAULT_REC_MODEL, params)
    print("识别完成:")
    print(result)


if __name__ == "__main__":
    demo()
