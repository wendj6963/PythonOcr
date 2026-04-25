from __future__ import annotations

# 本项目要求 Python 3.9+（pyproject.toml 已声明）。
# 若 IDE 误用 Python2 解释器进行静态检查，会出现一系列 2.7 警告，可忽略。

import json
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


_CN_FONT_CACHE: dict[int, object] = {}


def _get_cn_font(size: int = 24):
    """Return a PIL font that supports Chinese on Windows.

    OpenCV Hershey fonts cannot render Chinese, which appears as乱码/方块.
    We prefer Windows system fonts; fallback to PIL default font (may not support Chinese).
    """

    if ImageFont is None:
        return None

    key = int(size)
    if key in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[key]

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),  # 微软雅黑
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),  # 黑体
        Path("C:/Windows/Fonts/simsun.ttc"),  # 宋体
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    font = None
    for p in candidates:
        try:
            if p.exists():
                font = ImageFont.truetype(str(p), size=size)
                break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    _CN_FONT_CACHE[key] = font
    return font


def draw_text_unicode(
    img_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    color_bgr: tuple[int, int, int] = (255, 0, 0),
    size: int = 24,
) -> np.ndarray:
    """Draw unicode text onto BGR image using PIL.

    If PIL is unavailable, fall back to cv2.putText (ASCII only).
    """

    if Image is None or ImageDraw is None:
        cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        return img_bgr

    font = _get_cn_font(size)
    # Convert BGR->RGB for PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # PIL uses RGB color
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text(org, text, fill=color_rgb, font=font)
    # Convert back
    out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:, :, :] = out
    return img_bgr


"""Labeling tool.

This module provides an OpenCV-based labeling UI.

Important: to support rotated rectangles, labels are saved as YOLO-OBB points by default:
  cls x1 y1 x2 y2 x3 y3 x4 y4
(normalized coordinates, 4 corners in clockwise order as drawn by the box model)

For backward compatibility, we can still LOAD classic YOLO horizontal boxes:
  cls x y w h
"""


def load_vocab(vocab_path: Path) -> list[str]:
    txt = vocab_path.read_text(encoding="utf-8").splitlines()
    vocab = [t.strip() for t in txt if t.strip()]
    if not vocab:
        raise ValueError("vocab 文件为空")
    return vocab


class SimpleLabeler:
    """OpenCV-based labeling tool.

    Supports two modes:
    - det: one or multiple region boxes, all class=0
    - rec: character boxes, class selected by keyboard (0-9, a-z, etc.) according to vocab

    Keys:
      n: 下一张
      p: 上一张
      s: 保存
      r: 清空当前图片框
      z: 撤销最后一个框
      q / ESC: 退出

    Mouse:
      左键拖拽：新建框
      选中框后：
        - 移动：鼠标放在框内部（显示十字），左键拖动
        - 缩放：鼠标放在边上（显示双向箭头），左键拖动
        - 旋转：鼠标放在四个角（显示旋转箭头），左键拖动

    rec mode class selection:
      [ and ] : 切换类别

    Auto-increment:
      - rec 模式下，每画完一个新框，会自动将类别索引 +1（可用 [ / ] 手动调整）
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        mode: str,
        vocab_path: Path | None = None,
        window_name: str = "PythonOcr 标注工具",
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.mode = mode
        self.window_name = window_name

        self.images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}]
        self.images.sort()
        if not self.images:
            raise FileNotFoundError(f"未找到图片: {images_dir}")

        self.vocab: list[str] = []
        if self.mode == "rec":
            if vocab_path is None:
                raise ValueError("rec 模式必须提供 vocab 文件")
            self.vocab = load_vocab(vocab_path)

        self.idx = 0
        # 支持两种标注格式：
        # 1) 传统 YOLO：cls x y w h（水平框）
        # 2) YOLO-OBB：cls x1 y1 x2 y2 x3 y3 x4 y4（四点归一化，顺时针）
        # 内部统一使用 RotBox 表示（可旋转矩形）
        self.boxes: list[RotBox] = []
        self.dragging = False
        self.p1: tuple[int, int] | None = None
        self.p2: tuple[int, int] | None = None
        self.cur_cls = 0

        # editing state
        self.active_idx: int | None = None
        self.hover_hit: HitType = HitType.NONE
        self._edit_start_mouse: tuple[int, int] | None = None
        self._edit_start_box: RotBox | None = None
        self._new_box_mode: bool = False

        # cache current image size to avoid IO in mouse callbacks
        self._cur_w: int = 0
        self._cur_h: int = 0

    def _img_path(self) -> Path:
        return self.images[self.idx]

    def _label_path(self) -> Path:
        return self.labels_dir / (self._img_path().stem + ".txt")

    def _load_existing(self, img_w: int, img_h: int) -> None:
        self.boxes = []
        p = self._label_path()
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 5 columns: cls x y w h
            # 9 columns: cls x1 y1 x2 y2 x3 y3 x4 y4  (YOLO-OBB)
            if len(parts) == 5:
                cls = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:])
                self.boxes.append(RotBox.from_yolo_xywh(cls, x, y, bw, bh, angle=0.0))
            elif len(parts) == 9:
                cls = int(float(parts[0]))
                pts = list(map(float, parts[1:]))
                self.boxes.append(RotBox.from_yolo_obb_points(cls, pts))
            else:
                continue

    def _draw(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        self._cur_h, self._cur_w = h, w

        # draw saved boxes
        for i, b in enumerate(self.boxes):
            color = (0, 255, 0) if self.active_idx != i else (0, 165, 255)
            pts = b.points_px(w, h)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

            # corners
            for (cx, cy) in pts:
                cv2.circle(out, (int(cx), int(cy)), 4, color, -1)

            x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
            label = str(b.cls)
            if self.mode == "rec" and 0 <= b.cls < len(self.vocab):
                label = f"{b.cls}:{self.vocab[b.cls]}"
            cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw dragging rect
        if self._new_box_mode and self.dragging and self.p1 and self.p2:
            cv2.rectangle(out, self.p1, self.p2, (0, 0, 255), 2)

        # cursor hint (visual)
        cursor_text = {
            HitType.ROTATE: "旋转",
            HitType.MOVE: "移动",
            HitType.EDGE_N: "缩放",
            HitType.EDGE_S: "缩放",
            HitType.EDGE_E: "缩放",
            HitType.EDGE_W: "缩放",
            HitType.CORNER_TL: "旋转",
            HitType.CORNER_TR: "旋转",
            HitType.CORNER_BR: "旋转",
            HitType.CORNER_BL: "旋转",
            HitType.NONE: "",
        }.get(self.hover_hit, "")
        if cursor_text:
            cv2.putText(out, f"操作: {cursor_text}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # header
        header = f"[{self.mode}] {self.idx+1}/{len(self.images)}  {self._img_path().name}"
        if self.mode == "rec":
            cur = self.vocab[self.cur_cls] if 0 <= self.cur_cls < len(self.vocab) else str(self.cur_cls)
            header += f"  当前类别: {self.cur_cls}:{cur}"
        cv2.putText(out, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        hint = "n下一张 p上一张 s保存 r清空 z撤销 q退出"
        cv2.putText(out, hint, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return out

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:  # noqa: ANN001
        # update hover hit
        if event == cv2.EVENT_MOUSEMOVE and not self.dragging:
            self._update_hover((x, y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self._edit_start_mouse = (x, y)

            # if hovering an existing box handle -> edit, else create new box
            if self.active_idx is not None and self.hover_hit != HitType.NONE:
                self._edit_start_box = self.annos[self.active_idx].box.copy()
                self._new_box_mode = False
            else:
                self._new_box_mode = True
                self.p1 = (x, y)
                self.p2 = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self._new_box_mode:
                self.p2 = (x, y)
            else:
                self._apply_edit((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self._new_box_mode:
                self.p2 = (x, y)
                if self.p1 and self.p2:
                    self._finalize_box(self.p1, self.p2)
                self.p1 = None
                self.p2 = None
                self._new_box_mode = False
            else:
                # finish edit
                self._edit_start_mouse = None
                self._edit_start_box = None

    def _update_hover(self, pt: tuple[int, int]) -> None:
        w, h = self._cur_w, self._cur_h
        if w <= 0 or h <= 0:
            self.active_idx = None
            self.hover_hit = HitType.NONE
            return
        idx, hit = _best_hit(self.boxes, pt, w, h)
        self.active_idx = idx
        self.hover_hit = hit

    def _apply_edit(self, pt: tuple[int, int]) -> None:
        if self.active_idx is None or self._edit_start_mouse is None or self._edit_start_box is None:
            return
        w, h = self._cur_w, self._cur_h
        if w <= 0 or h <= 0:
            return
        sx, sy = self._edit_start_mouse
        cx, cy = pt
        dx = float(cx - sx)
        dy = float(cy - sy)

        b0 = self._edit_start_box
        hit = self.hover_hit
        if _hit_is_move(hit):
            self.boxes[self.active_idx] = _apply_move(b0, dx, dy, w, h)
        elif _hit_is_resize(hit):
            self.boxes[self.active_idx] = _apply_resize(b0, hit, dx, dy, w, h)
        elif _hit_is_rotate(hit):
            self.boxes[self.active_idx] = _apply_rotate(
                b0, (sx, sy), (cx, cy), w, h, start_angle=b0.angle
            )

    def _finalize_box(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        # use cached size
        w, h = self._cur_w, self._cur_h
        if w <= 0 or h <= 0:
            img = cv2.imread(str(self._img_path()), cv2.IMREAD_COLOR)
            if img is None:
                return
            h, w = img.shape[:2]
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return
        cls = 0 if self.mode == "det" else int(self.cur_cls)
        rb = RotBox.from_xyxy_px(cls=cls, x1=x1, y1=y1, x2=x2, y2=y2, img_w=w, img_h=h, angle=0.0)
        self.boxes.append(rb)
        self.active_idx = len(self.boxes) - 1

        # auto increment class for recognition labeling
        if self.mode == "rec" and self.vocab:
            self.cur_cls = min(len(self.vocab) - 1, self.cur_cls + 1)

    def _save(self) -> None:
        # 为了支持旋转框，默认保存为 YOLO-OBB 格式（cls + 8 点）。
        # 如果全部为水平框(angle=0)，也兼容被常规 YOLO 读取。
        save_yolo_txt_rot(self._label_path(), self.boxes)

    def _next(self) -> None:
        self.idx = min(len(self.images) - 1, self.idx + 1)

    def _prev(self) -> None:
        self.idx = max(0, self.idx - 1)

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            img_path = self._img_path()
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {img_path}")

            h, w = img.shape[:2]
            self._load_existing(w, h)

            while True:
                frame = self._draw(img)
                cv2.imshow(self.window_name, frame)
                k = cv2.waitKey(20) & 0xFF

                if k in (27, ord("q")):
                    cv2.destroyAllWindows()
                    return
                if k == ord("n"):
                    self._save()
                    self._next()
                    break
                if k == ord("p"):
                    self._save()
                    self._prev()
                    break
                if k == ord("s"):
                    self._save()
                if k == ord("r"):
                    self.boxes = []
                    self.active_idx = None
                    self._save()
                if k == ord("z"):
                    if self.boxes:
                        self.boxes.pop()
                        if self.active_idx is not None:
                            self.active_idx = min(self.active_idx, len(self.boxes) - 1) if self.boxes else None
                        self._save()
                if self.mode == "rec":
                    if k == ord("["):
                        self.cur_cls = max(0, self.cur_cls - 1)
                    if k == ord("]"):
                        self.cur_cls = min(len(self.vocab) - 1, self.cur_cls + 1)


class HitType(Enum):
    NONE = "none"
    MOVE = "move"
    ROTATE = "rotate"
    EDGE_N = "edge_n"
    EDGE_S = "edge_s"
    EDGE_E = "edge_e"
    EDGE_W = "edge_w"
    CORNER_TL = "corner_tl"
    CORNER_TR = "corner_tr"
    CORNER_BR = "corner_br"
    CORNER_BL = "corner_bl"


@dataclass
class RotBox:
    cls: int
    cx: float  # normalized
    cy: float  # normalized
    w: float   # normalized
    h: float   # normalized
    angle: float  # radians, CCW

    def copy(self) -> "RotBox":
        return RotBox(self.cls, self.cx, self.cy, self.w, self.h, self.angle)

    @staticmethod
    def from_xyxy_px(cls: int, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, angle: float) -> "RotBox":
        x1c = min(x1, x2)
        x2c = max(x1, x2)
        y1c = min(y1, y2)
        y2c = max(y1, y2)
        bw = max(1, x2c - x1c)
        bh = max(1, y2c - y1c)
        cx = (x1c + bw / 2.0) / img_w
        cy = (y1c + bh / 2.0) / img_h
        return RotBox(cls=cls, cx=float(cx), cy=float(cy), w=float(bw / img_w), h=float(bh / img_h), angle=float(angle))

    @staticmethod
    def from_yolo_xywh(cls: int, x: float, y: float, w: float, h: float, angle: float) -> "RotBox":
        return RotBox(cls=cls, cx=x, cy=y, w=w, h=h, angle=angle)

    @staticmethod
    def from_yolo_obb_points(cls: int, pts8: list[float]) -> "RotBox":
        # pts8 are normalized x1 y1 ... x4 y4
        pts = np.array(pts8, dtype=np.float32).reshape(4, 2)
        # Fit minAreaRect on a unit canvas (normalized coords); we only need center/w/h/angle.
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), ang_deg = rect
        # cv2 angle definition depends on whether it reports the "width" or "height" side.
        # Normalize to make round-trip more stable: enforce w>=h and adjust angle accordingly.
        w = float(rw)
        h = float(rh)
        angle = float(np.deg2rad(ang_deg))
        if h > w:
            w, h = h, w
            angle = angle + float(np.pi / 2.0)
        angle = _normalize_angle(angle)
        return RotBox(cls=cls, cx=float(cx), cy=float(cy), w=float(w), h=float(h), angle=float(angle))

    def points_px(self, img_w: int, img_h: int) -> np.ndarray:
        # returns int32 points shape (4,2)
        cx = self.cx * img_w
        cy = self.cy * img_h
        bw = self.w * img_w
        bh = self.h * img_h
        ca = float(np.cos(self.angle))
        sa = float(np.sin(self.angle))

        # local corners (clockwise starting top-left in local frame)
        local = np.array(
            [
                [-bw / 2, -bh / 2],
                [bw / 2, -bh / 2],
                [bw / 2, bh / 2],
                [-bw / 2, bh / 2],
            ],
            dtype=np.float32,
        )
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        pts = local @ rot.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        return pts.astype(np.int32)

    def points_norm(self) -> np.ndarray:
        """Return float32 normalized points (4,2) without rounding."""
        cx = self.cx
        cy = self.cy
        bw = self.w
        bh = self.h
        ca = float(np.cos(self.angle))
        sa = float(np.sin(self.angle))
        local = np.array(
            [
                [-bw / 2, -bh / 2],
                [bw / 2, -bh / 2],
                [bw / 2, bh / 2],
                [-bw / 2, bh / 2],
            ],
            dtype=np.float32,
        )
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        pts = local @ rot.T
        pts[:, 0] += cx
        pts[:, 1] += cy
        return pts


def save_yolo_txt_rot(out_txt: Path, boxes: list[RotBox]) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for b in boxes:
        # Save as OBB 4 points in normalized coordinates
        pts = b.points_norm().astype(np.float32)
        pts8 = pts.reshape(-1).tolist()
        lines.append(
            f"{b.cls} "
            + " ".join([f"{v:.6f}" for v in pts8])
        )
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------- editing helpers ----------------------
def _point_poly_test(pt: tuple[int, int], poly: np.ndarray) -> float:
    return float(cv2.pointPolygonTest(poly.astype(np.float32), pt, True))


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _segment_distance(p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> float:
    # distance from point p to segment ab
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-6:
        return _dist(p, a)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return float(np.hypot(px - cx, py - cy))


def _edge_midpoints(pts: np.ndarray) -> list[tuple[int, int]]:
    mids: list[tuple[int, int]] = []
    for i in range(4):
        j = (i + 1) % 4
        mx = int((int(pts[i, 0]) + int(pts[j, 0])) / 2)
        my = int((int(pts[i, 1]) + int(pts[j, 1])) / 2)
        mids.append((mx, my))
    return mids


def _angle_between(center: tuple[float, float], p: tuple[int, int]) -> float:
    cx, cy = center
    return float(np.arctan2(p[1] - cy, p[0] - cx))


def _normalize_angle(a: float) -> float:
    # map to [-pi, pi]
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _ensure_min_size(w: float, h: float, min_norm: float = 0.002) -> tuple[float, float]:
    return (max(w, min_norm), max(h, min_norm))


def _box_center_px(b: RotBox, img_w: int, img_h: int) -> tuple[float, float]:
    return (b.cx * img_w, b.cy * img_h)


def _cursor_by_hit(hit: HitType) -> int:
    # OpenCV cursor limited on Windows; we simulate by drawing hints.
    return 0


def _rotbox_hit_test(b: RotBox, pt: tuple[int, int], img_w: int, img_h: int) -> HitType:
    pts = b.points_px(img_w, img_h)
    # corners threshold
    corner_th = 10.0
    corners = [tuple(map(int, pts[i])) for i in range(4)]
    corner_hits = [HitType.CORNER_TL, HitType.CORNER_TR, HitType.CORNER_BR, HitType.CORNER_BL]
    for c, ht in zip(corners, corner_hits, strict=False):
        if _dist(pt, c) <= corner_th:
            return ht

    # edges threshold
    edge_th = 8.0
    edges = [
        (corners[0], corners[1], HitType.EDGE_N),
        (corners[1], corners[2], HitType.EDGE_E),
        (corners[2], corners[3], HitType.EDGE_S),
        (corners[3], corners[0], HitType.EDGE_W),
    ]
    for a, c, ht in edges:
        if _segment_distance(pt, a, c) <= edge_th:
            return ht

    # inside
    inside = _point_poly_test(pt, pts)
    if inside >= 0:
        return HitType.MOVE
    return HitType.NONE


def _hit_is_rotate(hit: HitType) -> bool:
    return hit in {
        HitType.CORNER_TL,
        HitType.CORNER_TR,
        HitType.CORNER_BR,
        HitType.CORNER_BL,
        HitType.ROTATE,
    }


def _hit_is_resize(hit: HitType) -> bool:
    return hit in {HitType.EDGE_N, HitType.EDGE_S, HitType.EDGE_E, HitType.EDGE_W}


def _hit_is_move(hit: HitType) -> bool:
    return hit == HitType.MOVE


def _local_axes(b: RotBox) -> tuple[np.ndarray, np.ndarray]:
    # unit vectors in image px space for current angle
    ca = float(np.cos(b.angle))
    sa = float(np.sin(b.angle))
    ux = np.array([ca, sa], dtype=np.float32)
    uy = np.array([-sa, ca], dtype=np.float32)
    return ux, uy


def _apply_move(b: RotBox, dx_px: float, dy_px: float, img_w: int, img_h: int) -> RotBox:
    return RotBox(
        cls=b.cls,
        cx=_clamp01(b.cx + dx_px / img_w),
        cy=_clamp01(b.cy + dy_px / img_h),
        w=b.w,
        h=b.h,
        angle=b.angle,
    )


def _apply_resize(b: RotBox, hit: HitType, dx_px: float, dy_px: float, img_w: int, img_h: int) -> RotBox:
    # project mouse delta onto local axes to scale width/height
    ux, uy = _local_axes(b)
    d = np.array([dx_px, dy_px], dtype=np.float32)
    du = float(d @ ux)  # along width axis
    dv = float(d @ uy)  # along height axis

    w_px = b.w * img_w
    h_px = b.h * img_h

    if hit == HitType.EDGE_E:
        w_px += du
    elif hit == HitType.EDGE_W:
        w_px -= du
    elif hit == HitType.EDGE_S:
        h_px += dv
    elif hit == HitType.EDGE_N:
        h_px -= dv

    w_n, h_n = _ensure_min_size(w_px / img_w, h_px / img_h)
    return RotBox(cls=b.cls, cx=b.cx, cy=b.cy, w=w_n, h=h_n, angle=b.angle)


def _apply_rotate(b: RotBox, start_mouse: tuple[int, int], cur_mouse: tuple[int, int], img_w: int, img_h: int, start_angle: float) -> RotBox:
    cx, cy = _box_center_px(b, img_w, img_h)
    a0 = _angle_between((cx, cy), start_mouse)
    a1 = _angle_between((cx, cy), cur_mouse)
    da = _normalize_angle(a1 - a0)
    return RotBox(cls=b.cls, cx=b.cx, cy=b.cy, w=b.w, h=b.h, angle=_normalize_angle(start_angle + da))


def _select_topmost(boxes: list[RotBox], pt: tuple[int, int], img_w: int, img_h: int) -> int | None:
    # select first box where inside test passes, from last to first (topmost)
    for i in range(len(boxes) - 1, -1, -1):
        pts = boxes[i].points_px(img_w, img_h)
        if _point_poly_test(pt, pts) >= 0:
            return i
    return None


def _best_hit(boxes: list[RotBox], pt: tuple[int, int], img_w: int, img_h: int) -> tuple[int | None, HitType]:
    # prefer handles over inside
    best_i: int | None = None
    best_hit = HitType.NONE
    best_score = 1e9

    for i in range(len(boxes) - 1, -1, -1):
        hit = _rotbox_hit_test(boxes[i], pt, img_w, img_h)
        if hit == HitType.NONE:
            continue
        # score: corners highest priority, then edges, then move
        prio = 0
        if hit in {HitType.CORNER_TL, HitType.CORNER_TR, HitType.CORNER_BR, HitType.CORNER_BL}:
            prio = 0
        elif hit in {HitType.EDGE_N, HitType.EDGE_E, HitType.EDGE_S, HitType.EDGE_W}:
            prio = 1
        else:
            prio = 2
        if prio < best_score:
            best_score = prio
            best_i = i
            best_hit = hit

    return best_i, best_hit


@dataclass
class BoxAnno:
    box: RotBox
    text: str = ""
    index: int = 0


class AssistantLabeler:
    """简化版“标定助手”：一张图一次完成定位框 + 识别内容。"""

    def __init__(
        self,
        images_dir: Path,
        det_labels_dir: Path,
        rec_labels_dir: Path,
        vocab_path: Path,
        window_name: str = "PythonOcr 标定助手",
    ) -> None:
        self.images_dir = images_dir
        self.vocab_path = vocab_path

        split_name = images_dir.name.lower()
        if split_name in {"train", "val"}:
            if det_labels_dir.name.lower() not in {"train", "val"}:
                det_labels_dir = det_labels_dir / split_name
            if rec_labels_dir.name.lower() not in {"train", "val"}:
                rec_labels_dir = rec_labels_dir / split_name

        self.det_labels_dir = det_labels_dir
        self.rec_labels_dir = rec_labels_dir
        self.window_name = window_name

        self.images = [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}
        ]
        self.images.sort()
        if not self.images:
            raise FileNotFoundError(f"未找到图片: {images_dir}")

        self.vocab = load_vocab(vocab_path)
        self.char_to_cls = {ch: i for i, ch in enumerate(self.vocab)}

        self.idx = 0
        self.annos: list[BoxAnno] = []
        self.active_idx: int | None = None
        self.hover_hit: HitType = HitType.NONE

        self.dragging = False
        self._new_box_mode = False
        self.p1: tuple[int, int] | None = None
        self.p2: tuple[int, int] | None = None
        self._edit_start_mouse: tuple[int, int] | None = None
        self._edit_start_box: RotBox | None = None
        self._cur_w = 0
        self._cur_h = 0

        self._input_mode: str = ""
        self._input_buffer: str = ""
        self._input_prompt: str = ""
        self._input_target_idx: int | None = None

        self._auto_prompt_on_new_box: bool = True

        self._last_save_rec_boxes: int = 0
        self._last_save_skipped_chars: int = 0
        self._last_auto_added_chars: int = 0

        self._labeled_count: int = 0
        self._unlabeled_count: int = 0
        self._update_label_stats()

    def _img_path(self) -> Path:
        return self.images[self.idx]

    def _det_label_path(self) -> Path:
        return self.det_labels_dir / (self._img_path().stem + ".txt")

    def _rec_label_path(self) -> Path:
        return self.rec_labels_dir / (self._img_path().stem + ".txt")

    def _update_label_stats(self) -> None:
        """统计已标定/未标定数量（以 det 标签文件是否非空为准）。"""
        labeled = 0
        for p in self.images:
            lp = self.det_labels_dir / (p.stem + ".txt")
            if lp.exists() and lp.stat().st_size > 0:
                labeled += 1
        self._labeled_count = labeled
        self._unlabeled_count = max(0, len(self.images) - labeled)

    def _register_new_chars(self, text: str) -> int:
        chars = [c for c in text if c.strip()]
        if not chars:
            return 0
        missing = []
        known = set(self.vocab)
        for ch in chars:
            if ch not in known:
                missing.append(ch)
                known.add(ch)
        if not missing:
            return 0
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with self.vocab_path.open("a", encoding="utf-8") as f:
            if self.vocab_path.stat().st_size > 0:
                f.write("\n")
            f.write("\n".join(missing))
            f.write("\n")
        self.vocab = load_vocab(self.vocab_path)
        self.char_to_cls = {ch: i for i, ch in enumerate(self.vocab)}
        return len(missing)

    def _assistant_meta_path(self) -> Path:
        """侧写 meta 文件，用于稳定回显文本/序号。"""
        return self.det_labels_dir / (self._img_path().stem + ".assistant.json")

    def _save_meta(self) -> None:
        payload = {
            "version": 1,
            "image": self._img_path().name,
            "items": [
                {
                    "index": int(a.index),
                    "text": str(a.text),
                    "cls": int(a.box.cls),
                    "cx": float(a.box.cx),
                    "cy": float(a.box.cy),
                    "w": float(a.box.w),
                    "h": float(a.box.h),
                    "angle": float(a.box.angle),
                }
                for a in self.annos
            ],
        }
        mp = self._assistant_meta_path()
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_meta(self) -> bool:
        mp = self._assistant_meta_path()
        if not mp.exists():
            return False
        try:
            payload = json.loads(mp.read_text(encoding="utf-8", errors="ignore"))
            items = payload.get("items", [])
            annos: list[BoxAnno] = []
            for it in items:
                rb = RotBox(
                    cls=int(it.get("cls", 0)),
                    cx=_clamp01(float(it.get("cx", 0.5))),
                    cy=_clamp01(float(it.get("cy", 0.5))),
                    w=max(float(it.get("w", 0.1)), 1e-6),
                    h=max(float(it.get("h", 0.1)), 1e-6),
                    angle=_normalize_angle(float(it.get("angle", 0.0))),
                )
                annos.append(
                    BoxAnno(
                        box=rb,
                        text=str(it.get("text", "")),
                        index=int(it.get("index", len(annos) + 1)),
                    )
                )
            self.annos = annos
            self.active_idx = 0 if self.annos else None
            return True
        except Exception:
            return False

    def _load_existing(self) -> None:
        # 优先从 meta 恢复，确保文本与角度稳定。
        if self._load_meta():
            return

        self.annos = []
        det_boxes: list[RotBox] = []
        dp = self._det_label_path()
        if dp.exists():
            for line in dp.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) == 9:
                    cls = int(float(parts[0]))
                    pts = [float(v) for v in parts[1:]]
                    # clamp to [0,1] to avoid out-of-range display glitches
                    pts = [max(0.0, min(1.0, v)) for v in pts]
                    rb = RotBox.from_yolo_obb_points(cls, pts)
                    det_boxes.append(rb)

        rec_boxes: list[RotBox] = []
        rp = self._rec_label_path()
        if rp.exists():
            for line in rp.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) == 9:
                    cls = int(float(parts[0]))
                    pts = [float(v) for v in parts[1:]]
                    pts = [max(0.0, min(1.0, v)) for v in pts]
                    rec_boxes.append(RotBox.from_yolo_obb_points(cls, pts))

        # fallback: 仅 det
        if not rec_boxes:
            self.annos = [BoxAnno(box=b, text="", index=i + 1) for i, b in enumerate(det_boxes)]
            self.active_idx = 0 if self.annos else None
            return

        # 根据 rec 框中心是否落在 det 框内进行分组，恢复文本
        annos: list[BoxAnno] = []
        for i, b in enumerate(det_boxes):
            poly = b.points_norm().astype(np.float32)
            ca = float(np.cos(b.angle))
            sa = float(np.sin(b.angle))
            ux = np.array([ca, sa], dtype=np.float32)
            hits: list[tuple[float, RotBox]] = []
            for rb in rec_boxes:
                cx, cy = rb.cx, rb.cy
                if _point_poly_test((cx, cy), poly) >= 0:
                    proj = float((np.array([cx - b.cx, cy - b.cy], dtype=np.float32) @ ux))
                    hits.append((proj, rb))
            hits.sort(key=lambda t: t[0])
            text = ""
            for _, rb in hits:
                if 0 <= rb.cls < len(self.vocab):
                    text += self.vocab[int(rb.cls)]
            annos.append(BoxAnno(box=b, text=text, index=i + 1))

        self.annos = annos
        self.active_idx = 0 if self.annos else None

    def _save(self) -> None:
        det_boxes = [RotBox(cls=0, cx=a.box.cx, cy=a.box.cy, w=a.box.w, h=a.box.h, angle=a.box.angle) for a in self.annos]
        save_yolo_txt_rot(self._det_label_path(), det_boxes)

        rec_boxes: list[RotBox] = []
        skipped = 0
        auto_added = 0
        for a in sorted(self.annos, key=lambda t: t.index):
            s = a.text.strip()
            if not s:
                continue
            auto_added += self._register_new_chars(s)
            chars = [c for c in s if c.strip()]
            if not chars:
                continue
            n = len(chars)
            ca = float(np.cos(a.box.angle))
            sa = float(np.sin(a.box.angle))
            ux = np.array([ca, sa], dtype=np.float32)
            char_w = a.box.w / n
            for i, ch in enumerate(chars):
                if ch not in self.char_to_cls:
                    skipped += 1
                    continue
                offset = (-a.box.w / 2.0) + (i + 0.5) * char_w
                cx = float(a.box.cx + offset * ux[0])
                cy = float(a.box.cy + offset * ux[1])
                rec_boxes.append(
                    RotBox(
                        cls=int(self.char_to_cls[ch]),
                        cx=_clamp01(cx),
                        cy=_clamp01(cy),
                        w=max(char_w, 1e-6),
                        h=a.box.h,
                        angle=a.box.angle,
                    )
                )
        save_yolo_txt_rot(self._rec_label_path(), rec_boxes)

        if not rec_boxes:
            rp = self._rec_label_path()
            if rp.exists() and rp.stat().st_size == 0:
                try:
                    rp.unlink()
                except Exception:
                    pass

        self._last_save_rec_boxes = len(rec_boxes)
        self._last_save_skipped_chars = skipped
        self._last_auto_added_chars = auto_added
        self._update_label_stats()
        self._save_meta()

    def _current_boxes(self) -> list[RotBox]:
        return [a.box for a in self.annos]

    def _draw(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        self._cur_h, self._cur_w = h, w

        for i, a in enumerate(self.annos):
            pts = a.box.points_px(w, h)
            color = (0, 165, 255) if self.active_idx == i else (0, 255, 0)
            cv2.polylines(out, [pts], True, color, 2)
            for (cx, cy) in pts:
                cv2.circle(out, (int(cx), int(cy)), 4, color, -1)

            x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
            info = f"{a.index}:{a.text}" if a.text else f"{a.index}:<空>"
            draw_text_unicode(out, info, (x1, max(15, y1 - 22)), color, size=18)

        if self._new_box_mode and self.dragging and self.p1 and self.p2:
            cv2.rectangle(out, self.p1, self.p2, (0, 0, 255), 2)

        header = f"[助手] {self.idx+1}/{len(self.images)}  {self._img_path().name}"
        draw_text_unicode(out, header, (10, 8), (255, 0, 0), size=22)
        draw_text_unicode(
            out,
            "TAB选择框  E编辑识别  I编辑序号  n/p翻页(自动保存)  q退出",
            (10, 36),
            (255, 0, 0),
            size=20,
        )

        split_hint = f"labels 输出: det={self.det_labels_dir.name} rec={self.rec_labels_dir.name}"
        if self._last_auto_added_chars > 0:
            split_hint += f"  (自动新增类别 {self._last_auto_added_chars} 个)"
        if self._last_save_skipped_chars > 0:
            split_hint += f"  (跳过未收录字符 {self._last_save_skipped_chars} 个)"
        draw_text_unicode(out, split_hint, (10, 58), (255, 0, 0), size=18)

        stats = f"已标定: {self._labeled_count}  未标定: {self._unlabeled_count}"
        draw_text_unicode(out, stats, (10, 80), (255, 0, 0), size=18)

        if self._input_mode and self._input_target_idx is not None:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            prompt = self._input_prompt
            buf = self._input_buffer
            draw_text_unicode(out, "【输入模式】", (10, 60), (0, 255, 255), size=22)
            draw_text_unicode(out, prompt, (10, 84), (0, 255, 255), size=20)
            draw_text_unicode(
                out,
                f"> {buf}_  (Enter确认 / Esc取消 / Backspace删除)",
                (10, 110),
                (0, 255, 255),
                size=20,
            )
        return out

    def _begin_input(self, mode: str) -> None:
        if self.active_idx is None:
            return
        self._input_mode = mode
        self._input_target_idx = int(self.active_idx)
        if mode == "text":
            cur = self.annos[self._input_target_idx].text
            self._input_prompt = f"输入识别内容（可多个字符，例如 AB12），当前={cur!r}"
            self._input_buffer = cur
        elif mode == "index":
            cur_i = self.annos[self._input_target_idx].index
            self._input_prompt = f"输入序号 index（整数，用于排序），当前={cur_i}"
            self._input_buffer = str(cur_i)
        else:
            self._input_prompt = ""
            self._input_buffer = ""

    def _cancel_input(self) -> None:
        self._input_mode = ""
        self._input_buffer = ""
        self._input_prompt = ""
        self._input_target_idx = None

    def _commit_input(self) -> None:
        if not self._input_mode or self._input_target_idx is None:
            return
        ti = self._input_target_idx
        if ti < 0 or ti >= len(self.annos):
            self._cancel_input()
            return
        if self._input_mode == "text":
            self.annos[ti].text = self._input_buffer.strip()
        elif self._input_mode == "index":
            s = self._input_buffer.strip()
            try:
                self.annos[ti].index = int(s)
            except Exception:
                pass
        self._cancel_input()
        self._save()

    def _handle_input_key(self, k: int) -> bool:
        if not self._input_mode:
            return False
        kk = int(k)
        if kk == 27:
            self._cancel_input()
            return True
        if kk in (13, 10):
            self._commit_input()
            return True
        if kk in (8, 127):
            if self._input_buffer:
                self._input_buffer = self._input_buffer[:-1]
            return True
        if 32 <= kk <= 126:
            ch = chr(kk)
            if self._input_mode == "index":
                if ch.isdigit() or (ch in "+-" and not self._input_buffer):
                    self._input_buffer += ch
            else:
                self._input_buffer += ch.upper()
            return True
        return True

    def _update_hover(self, pt: tuple[int, int]) -> None:
        w, h = self._cur_w, self._cur_h
        if w <= 0 or h <= 0:
            self.active_idx = None
            self.hover_hit = HitType.NONE
            return
        boxes = self._current_boxes()
        idx, hit = _best_hit(boxes, pt, w, h)
        self.active_idx = idx
        self.hover_hit = hit

    def _apply_edit(self, pt: tuple[int, int]) -> None:
        if self.active_idx is None or self._edit_start_mouse is None or self._edit_start_box is None:
            return
        w, h = self._cur_w, self._cur_h
        if w <= 0 or h <= 0:
            return
        sx, sy = self._edit_start_mouse
        cx, cy = pt
        dx = float(cx - sx)
        dy = float(cy - sy)
        b0 = self._edit_start_box
        hit = self.hover_hit
        if _hit_is_move(hit):
            self.annos[self.active_idx].box = _apply_move(b0, dx, dy, w, h)
        elif _hit_is_resize(hit):
            self.annos[self.active_idx].box = _apply_resize(b0, hit, dx, dy, w, h)
        elif _hit_is_rotate(hit):
            self.annos[self.active_idx].box = _apply_rotate(b0, (sx, sy), (cx, cy), w, h, start_angle=b0.angle)

    def _finalize_box(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        w, h = self._cur_w, self._cur_h
        if w <= 0 or h <= 0:
            return
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            return
        rb = RotBox.from_xyxy_px(cls=0, x1=x1, y1=y1, x2=x2, y2=y2, img_w=w, img_h=h, angle=0.0)
        next_index = max([a.index for a in self.annos], default=0) + 1
        self.annos.append(BoxAnno(box=rb, text="", index=next_index))
        self.active_idx = len(self.annos) - 1

        if self._auto_prompt_on_new_box:
            self._begin_input("text")

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:  # noqa: ANN001
        if event == cv2.EVENT_MOUSEMOVE and not self.dragging:
            self._update_hover((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            # 右键删除鼠标所在 ROI
            w, h = self._cur_w, self._cur_h
            if w > 0 and h > 0:
                idx, hit = _best_hit(self._current_boxes(), (x, y), w, h)
                if idx is not None and hit != HitType.NONE:
                    self.annos.pop(idx)
                    self.active_idx = min(self.active_idx or 0, len(self.annos) - 1) if self.annos else None
                    self._save()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self._edit_start_mouse = (x, y)

            # if hovering an existing box handle -> edit, else create new box
            if self.active_idx is not None and self.hover_hit != HitType.NONE:
                self._edit_start_box = self.annos[self.active_idx].box.copy()
                self._new_box_mode = False
            else:
                self._new_box_mode = True
                self.p1 = (x, y)
                self.p2 = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self._new_box_mode:
                self.p2 = (x, y)
            else:
                self._apply_edit((x, y))

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self._new_box_mode:
                self.p2 = (x, y)
                if self.p1 and self.p2:
                    self._finalize_box(self.p1, self.p2)
                self.p1 = None
                self.p2 = None
                self._new_box_mode = False
            else:
                self._edit_start_mouse = None
                self._edit_start_box = None

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            img_path = self._img_path()
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {img_path}")

            self._load_existing()
            while True:
                frame = self._draw(img)
                cv2.imshow(self.window_name, frame)
                # waitKeyEx is more reliable on Windows for key codes
                k = cv2.waitKeyEx(20)

                # input mode consumes keys first
                if self._handle_input_key(k):
                    continue

                if k in (27, ord("q"), ord("Q")):
                    self._save()
                    cv2.destroyAllWindows()
                    return
                if k in (ord("n"), ord("N")):
                    self._save()
                    self.idx = min(len(self.images) - 1, self.idx + 1)
                    break
                if k in (ord("p"), ord("P")):
                    self._save()
                    self.idx = max(0, self.idx - 1)
                    break
                if k in (ord("s"), ord("S")):
                    self._save()
                if k in (ord("z"), ord("Z")):
                    if self.annos:
                        self.annos.pop()
                        self.active_idx = min(self.active_idx or 0, len(self.annos) - 1) if self.annos else None
                        self._save()
                if k in (ord("r"), ord("R")):
                    self.annos = []
                    self.active_idx = None
                    self._save()
                if k in (ord("e"), ord("E")):
                    self._begin_input("text")
                if k in (ord("i"), ord("I")):
                    self._begin_input("index")
                if k == 9:  # TAB
                    if self.annos:
                        if self.active_idx is None:
                            self.active_idx = 0
                        else:
                            self.active_idx = (self.active_idx + 1) % len(self.annos)
                if k == 8:  # BACKSPACE used as shift-tab alternative
                    if self.annos:
                        if self.active_idx is None:
                            self.active_idx = 0
                        else:
                            self.active_idx = (self.active_idx - 1) % len(self.annos)


def write_labeler_config(out_path: Path, mode: str, vocab: str | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": mode, "vocab": vocab}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DualLabeler(AssistantLabeler):
    """兼容旧命令的 det+rec 标注器（当前与 AssistantLabeler 共用实现）。"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        super().__init__(*args, **kwargs)

