from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainArgs:
    data: Path
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    project: Path
    name: str


def _require_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未安装或无法导入 ultralytics。请先 pip install -e . 或 pip install ultralytics"
        ) from e


def train(args: TrainArgs) -> Any:
    YOLO = _require_ultralytics()
    model = YOLO(args.model)
    return model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )


def predict(
    weights: Path,
    image: Any,
    imgsz: int = 640,
    device: str = "cpu",
    conf: float | None = None,
    iou: float | None = None,
    max_det: int | None = None,
) -> Any:
    YOLO = _require_ultralytics()
    model = YOLO(str(weights))

    source: Any
    if isinstance(image, (str, Path)):
        source = str(image)
    else:
        source = image

    kwargs: dict[str, Any] = {"source": source, "imgsz": imgsz, "device": device}
    if conf is not None:
        kwargs["conf"] = float(conf)
    if iou is not None:
        kwargs["iou"] = float(iou)
    if max_det is not None:
        kwargs["max_det"] = int(max_det)

    return model.predict(**kwargs)
