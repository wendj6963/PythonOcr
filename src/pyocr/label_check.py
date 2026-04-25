from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class CheckReport:
    images: int
    labels: int
    missing_labels: int
    empty_labels: int
    bad_format: int
    out_of_range: int
    bad_class: int
    out_of_range_files: tuple[str, ...]


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _parse_line(parts: list[str]) -> tuple[int, list[float]] | None:
    if not parts:
        return None
    try:
        cls = int(float(parts[0]))
    except Exception:
        return None

    try:
        nums = [float(x) for x in parts[1:]]
    except Exception:
        return None

    return cls, nums


def check_obb_dataset(images_dir: Path, labels_dir: Path, num_classes: int | None = None) -> CheckReport:
    """Check YOLO-OBB label integrity.

    Expected per-image label file: <stem>.txt.
    Supported line formats:
      - OBB: cls x1 y1 x2 y2 x3 y3 x4 y4  (8 floats)
      - Classic: cls x y w h              (4 floats)  (allowed but discouraged)

    We validate:
      - label exists for each image
      - line has correct number of columns
      - coordinates within [0,1] (for OBB and classic)
      - class id within range (if num_classes provided)
    """

    images = sorted([p for p in images_dir.iterdir() if _is_image(p)])

    labels = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    label_set = {p.name for p in labels}

    missing_labels = 0
    empty_labels = 0
    bad_format = 0
    out_of_range = 0
    bad_class = 0
    out_of_range_files: set[str] = set()

    for img in images:
        lp = labels_dir / (img.stem + ".txt")
        if lp.name not in label_set:
            missing_labels += 1
            continue

        content = lp.read_text(encoding="utf-8", errors="ignore").splitlines()
        content = [ln.strip() for ln in content if ln.strip()]
        if not content:
            empty_labels += 1
            continue

        has_out_of_range = False
        for ln in content:
            parts = ln.split()
            parsed = _parse_line(parts)
            if parsed is None:
                bad_format += 1
                continue
            cls, nums = parsed

            if num_classes is not None and (cls < 0 or cls >= num_classes):
                bad_class += 1

            if len(nums) == 8:
                # OBB points
                for v in nums:
                    if not (0.0 <= v <= 1.0):
                        out_of_range += 1
                        has_out_of_range = True
                        break
            elif len(nums) == 4:
                # classic x y w h
                x, y, w, h = nums
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    out_of_range += 1
                    has_out_of_range = True
            else:
                bad_format += 1

        if has_out_of_range:
            out_of_range_files.add(img.name)

    return CheckReport(
        images=len(images),
        labels=len(labels),
        missing_labels=missing_labels,
        empty_labels=empty_labels,
        bad_format=bad_format,
        out_of_range=out_of_range,
        bad_class=bad_class,
        out_of_range_files=tuple(sorted(out_of_range_files)),
    )
