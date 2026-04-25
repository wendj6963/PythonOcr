from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .paths import ensure_dir


IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SplitResult:
    train: list[Path]
    val: list[Path]


def list_images(src_dir: Path) -> list[Path]:
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files


def split_files(files: list[Path], val_ratio: float, seed: int = 0) -> SplitResult:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 (0, 1) 之间")
    rng = random.Random(seed)
    files2 = files[:]
    rng.shuffle(files2)
    n_val = max(1, int(round(len(files2) * val_ratio))) if len(files2) > 1 else 0
    val = files2[:n_val]
    train = files2[n_val:]
    return SplitResult(train=train, val=val)


def _copy_images(files: list[Path], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for p in files:
        shutil.copy2(p, out_dir / p.name)


def prepare_det_dataset(src: Path, out_root: Path, val_ratio: float, seed: int = 0) -> None:
    """Create YOLO-style detection dataset directory (images/labels train/val).

    Note: labels are not generated here because repo currently has no annotations.
    """

    images = list_images(src)
    if not images:
        raise FileNotFoundError(f"未找到图片: {src}")

    split = split_files(images, val_ratio=val_ratio, seed=seed)

    train_img = out_root / "images" / "train"
    val_img = out_root / "images" / "val"
    train_lbl = out_root / "labels" / "train"
    val_lbl = out_root / "labels" / "val"

    _copy_images(split.train, train_img)
    _copy_images(split.val, val_img)

    ensure_dir(train_lbl)
    ensure_dir(val_lbl)


def prepare_rec_dataset(src: Path, out_root: Path, val_ratio: float, seed: int = 0) -> None:
    """Prepare recognition dataset for detection+class.

    This creates YOLO-style directory structure (images/labels train/val).
    Labels are not generated here because annotations are not included in the repo.
    """

    images = list_images(src)
    if not images:
        raise FileNotFoundError(f"未找到图片: {src}")

    split = split_files(images, val_ratio=val_ratio, seed=seed)

    train_img = out_root / "images" / "train"
    val_img = out_root / "images" / "val"
    train_lbl = out_root / "labels" / "train"
    val_lbl = out_root / "labels" / "val"

    _copy_images(split.train, train_img)
    _copy_images(split.val, val_img)

    ensure_dir(train_lbl)
    ensure_dir(val_lbl)

