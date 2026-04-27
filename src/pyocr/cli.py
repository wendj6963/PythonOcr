from __future__ import annotations

# 本项目要求 Python 3.9+（pyproject.toml 已声明）。

import argparse
import shutil

from .dataset_prep import prepare_det_dataset, prepare_rec_dataset
from .paths import ensure_dir, norm_path
from .label_tool import SimpleLabeler, DualLabeler, AssistantLabeler
from .label_check import check_obb_dataset
from .pipeline import infer_ocr, save_json
from .yolo_backend import TrainArgs, train
from .rec_ctc_prep import prepare_rec_ctc_dataset
from .rec_ctc_backend import CtcTrainArgs, CtcInferArgs, train_ctc, infer_ctc


def _add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--batch", type=int, default=8, help="batch 大小")
    p.add_argument("--device", type=str, default="cpu", help="设备，例如 cpu / 0")
    p.add_argument("--project", type=str, default="models", help="输出目录")
    p.add_argument("--name", type=str, required=True, help="实验名称")
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n-obb.pt",
        help="预训练权重或模型配置（需支持 OBB；后续可替换为 YOLOv13-OBB）",
    )


def cmd_prepare_det(args: argparse.Namespace) -> None:
    prepare_det_dataset(norm_path(args.src), norm_path(args.out), args.val_ratio, seed=args.seed)
    print(f"定位数据集已生成: {args.out}")


def cmd_prepare_rec(args: argparse.Namespace) -> None:
    prepare_rec_dataset(norm_path(args.src), norm_path(args.out), args.val_ratio, seed=args.seed)
    print(f"识别数据集(检测+类别)目录已生成: {args.out}（请导入 labels 标注）")


def cmd_import_yolo(args: argparse.Namespace) -> None:
    images_dir = norm_path(args.images)
    labels_src = norm_path(args.labels)
    labels_out = ensure_dir(norm_path(args.out_labels))

    if not images_dir.exists():
        raise SystemExit(f"images 目录不存在: {images_dir}")
    if not labels_src.exists():
        raise SystemExit(f"labels 来源目录不存在: {labels_src}")

    n = 0
    for img in images_dir.iterdir():
        if not img.is_file():
            continue
        txt = labels_src / (img.stem + ".txt")
        if txt.exists():
            shutil.copy2(txt, labels_out / txt.name)
            n += 1

    print(f"已导入 YOLO labels: {n} 个 -> {labels_out}")


def cmd_sync_split_labels(args: argparse.Namespace) -> None:
    """Sync labels to train/val according to images files.

    Use case:
      - You labeled everything into a single pool directory (e.g. datasets/det/labels_all)
      - You already split images into images/train and images/val
      - Then you need to copy matching <stem>.txt into labels/train and labels/val
    """

    images_root = norm_path(args.images_root)
    labels_src = norm_path(args.labels_src)
    labels_root = ensure_dir(norm_path(args.labels_root))

    if not images_root.exists():
        raise SystemExit(f"images_root 不存在: {images_root}")
    if not labels_src.exists():
        raise SystemExit(f"labels_src 不存在: {labels_src}")

    total_missing = 0
    total_copied = 0

    for split in ("train", "val"):
        img_dir = images_root / split
        if not img_dir.exists():
            continue
        out_dir = ensure_dir(labels_root / split)

        for img in img_dir.iterdir():
            if not img.is_file():
                continue
            src_txt = labels_src / (img.stem + ".txt")
            dst_txt = out_dir / (img.stem + ".txt")
            if src_txt.exists():
                shutil.copy2(src_txt, dst_txt)
                total_copied += 1
            else:
                total_missing += 1
                if args.create_empty:
                    dst_txt.write_text("", encoding="utf-8")

    print(f"同步完成: copied={total_copied}, missing={total_missing}, out={labels_root}")


def cmd_train_det(args: argparse.Namespace) -> None:
    ta = TrainArgs(
        data=norm_path(args.data),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=norm_path(args.project),
        name=args.name,
    )
    train(ta)


def cmd_train_rec(args: argparse.Namespace) -> None:
    ta = TrainArgs(
        data=norm_path(args.data),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=norm_path(args.project),
        name=args.name,
    )
    train(ta)


def cmd_prepare_rec_ctc(args: argparse.Namespace) -> None:
    rep = prepare_rec_ctc_dataset(
        det_root=norm_path(args.det_root),
        rec_root=norm_path(args.rec_root),
        out_root=norm_path(args.out),
        vocab_path=norm_path(args.vocab),
    )
    print(f"CTC 数据集已生成: {args.out}")
    print(f"images={rep.images}, roi_samples={rep.rois}, empty_skipped={rep.skipped_empty}")


def cmd_train_rec_ctc(args: argparse.Namespace) -> None:
    ta = CtcTrainArgs(
        train_manifest=norm_path(args.train_manifest),
        val_manifest=norm_path(args.val_manifest),
        vocab=norm_path(args.vocab),
        project=norm_path(args.project),
        name=args.name,
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        img_h=int(args.img_h),
        img_w=int(args.img_w),
        workers=int(args.workers),
        device=str(args.device),
    )
    best = train_ctc(ta)
    print(f"CTC 训练完成，best 权重: {best}")


def cmd_infer_ctc(args: argparse.Namespace) -> None:
    ia = CtcInferArgs(
        weights=norm_path(args.weights),
        image=norm_path(args.image),
        img_h=int(args.img_h),
        img_w=int(args.img_w),
        device=str(args.device),
    )
    result = infer_ctc(ia)
    if args.out:
        save_json(result, norm_path(args.out))
        print(f"结果已保存: {args.out}")
    else:
        print(result["text"])


def cmd_infer(args: argparse.Namespace) -> None:
    result = infer_ocr(
        det_weights=norm_path(args.det_weights),
        rec_weights=norm_path(args.rec_weights),
        image=norm_path(args.image),
        device=args.device,
        vocab=norm_path(args.vocab) if args.vocab else None,
    )
    if args.out:
        save_json(result, norm_path(args.out))
        print(f"结果已保存: {args.out}")
    else:
        print(result["text"])


def cmd_infer_sample(args: argparse.Namespace) -> None:
    """Randomly sample images and run OCR inference, saving outputs to a folder."""

    import random

    images_dir = norm_path(args.images)
    outdir = ensure_dir(norm_path(args.outdir))
    vocab = norm_path(args.vocab) if args.vocab else None

    if not images_dir.exists():
        raise SystemExit(f"images 目录不存在: {images_dir}")

    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}]
    imgs.sort()
    if not imgs:
        raise SystemExit(f"未找到图片: {images_dir}")

    rng = random.Random(int(args.seed))
    k = min(int(args.num), len(imgs))
    picks = rng.sample(imgs, k=k) if k < len(imgs) else imgs

    # lazy imports
    import cv2
    import numpy as np

    # PIL text drawing (unicode-safe)
    try:
        from .label_tool import draw_text_unicode  # reuse existing helper
    except Exception:
        draw_text_unicode = None  # type: ignore[assignment]

    for img_path in picks:
        res = infer_ocr(
            det_weights=norm_path(args.det_weights),
            rec_weights=norm_path(args.rec_weights),
            image=img_path,
            device=args.device,
            vocab=vocab,
        )

        stem = img_path.stem
        save_json(res, outdir / f"{stem}.json")

        # visualization (det poly + per-region text)
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]

                for b in res.get("boxes", []):
                    pts = b.get("poly") if isinstance(b, dict) else None
                    if pts and len(pts) == 8:
                        p = np.array(
                            [(int(pts[i] * img.shape[1]), int(pts[i + 1] * img.shape[0])) for i in range(0, 8, 2)],
                            dtype=np.int32,
                        )
                        cv2.polylines(img, [p], True, (0, 255, 0), 2)

                # add per-region recognized string
                regions = res.get("regions", [])
                if isinstance(regions, list):
                    for r in regions:
                        if not isinstance(r, dict):
                            continue
                        db = r.get("det_box") or {}
                        x1 = int(db.get("x1", 0))
                        y1 = int(db.get("y1", 0))

                        roi_chars = r.get("roi_chars", [])
                        if isinstance(roi_chars, list):
                            s = "".join([str(c.get("char", "")) for c in roi_chars if isinstance(c, dict)])
                        else:
                            s = ""

                        if not s:
                            continue

                        # draw text background
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        cv2.rectangle(img, (x1, max(0, y1 - 24)), (min(w - 1, x1 + 120), y1), (0, 0, 0), -1)

                        if draw_text_unicode is not None:
                            draw_text_unicode(img, s, (x1 + 2, max(0, y1 - 22)), (0, 255, 255), size=18)
                        else:
                            cv2.putText(img, s, (x1 + 2, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imwrite(str(outdir / f"{stem}.jpg"), img)
        except Exception:
            pass

    (outdir / "picks.txt").write_text("\n".join([p.name for p in picks]) + "\n", encoding="utf-8")
    print(f"抽查完成: {len(picks)} 张 -> {outdir}")


def cmd_check_obb_labels(args: argparse.Namespace) -> None:
    images_dir = norm_path(args.images)
    labels_dir = norm_path(args.labels)
    nc = int(args.num_classes) if args.num_classes else None

    if not images_dir.exists():
        raise SystemExit(f"images 目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"labels 目录不存在: {labels_dir}")

    rep = check_obb_dataset(images_dir=images_dir, labels_dir=labels_dir, num_classes=nc)

    print("================ OBB 标注检查报告 ================")
    print(f"images 数量        : {rep.images}")
    print(f"labels 文件数量    : {rep.labels}")
    print(f"缺失 labels        : {rep.missing_labels}")
    print(f"空 labels（无行）  : {rep.empty_labels}")
    print(f"格式错误行数       : {rep.bad_format}")
    print(f"坐标越界行数       : {rep.out_of_range}")
    print(f"类别越界行数       : {rep.bad_class}")
    if rep.out_of_range_files:
        print("坐标越界图片       : " + ", ".join(rep.out_of_range_files))

    bad = (
        rep.missing_labels
        + rep.empty_labels
        + rep.bad_format
        + rep.out_of_range
        + rep.bad_class
    )
    if bad > 0:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyocr", description="PythonOcr 命令行工具（中文提示）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare-det", help="准备定位(det)数据集目录")
    sp.add_argument("--src", required=True, help="原始图片目录，例如 Images/trains")
    sp.add_argument("--out", required=True, help="输出目录，例如 datasets/det")
    sp.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    sp.add_argument("--seed", type=int, default=0, help="随机种子")
    sp.set_defaults(func=cmd_prepare_det)

    sp = sub.add_parser("prepare-rec", help="准备识别(rec)数据集占位目录")
    sp.add_argument("--src", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--val-ratio", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_prepare_rec)

    sp = sub.add_parser("import-yolo", help="导入外部 YOLO labels（同名 .txt）")
    sp.add_argument("--images", required=True, help="images 目录")
    sp.add_argument("--labels", required=True, help="外部 labels 目录")
    sp.add_argument("--out-labels", required=True, help="输出 labels 目录")
    sp.set_defaults(func=cmd_import_yolo)

    sp = sub.add_parser(
        "sync-split-labels",
        help="按 images/train|val 同步拆分 labels（从一个 labels 池复制到 labels/train|val）",
    )
    sp.add_argument("--images-root", required=True, help="images 根目录（包含 train/val 子目录）")
    sp.add_argument("--labels-src", required=True, help="labels 池目录（含全量 <stem>.txt）")
    sp.add_argument("--labels-root", required=True, help="输出 labels 根目录（将生成 train/val）")
    sp.add_argument(
        "--create-empty",
        action="store_true",
        help="如果某图片缺少 label，则创建空 txt（谨慎使用：rec 通常不建议空标注）",
    )
    sp.set_defaults(func=cmd_sync_split_labels)

    sp = sub.add_parser("label-det", help="打开简易标注工具：定位(det)框标注")
    sp.add_argument("--images", required=True, help="图片目录，例如 datasets/det/images/train")
    sp.add_argument("--labels", required=True, help="labels 输出目录，例如 datasets/det/labels/train")
    sp.set_defaults(
        func=lambda a: SimpleLabeler(
            images_dir=norm_path(a.images),
            labels_dir=norm_path(a.labels),
            mode="det",
        ).run()
    )

    sp = sub.add_parser("label-rec", help="打开简易标注工具：识别(rec)字符框+类别")
    sp.add_argument("--images", required=True, help="图片目录，例如 datasets/rec/images/train")
    sp.add_argument("--labels", required=True, help="labels 输出目录，例如 datasets/rec/labels/train")
    sp.add_argument(
        "--vocab",
        required=True,
        help="类别到字符映射文件（UTF-8 每行一个字符，行号=class id），例如 vocab_digits.txt",
    )
    sp.set_defaults(
        func=lambda a: SimpleLabeler(
            images_dir=norm_path(a.images),
            labels_dir=norm_path(a.labels),
            mode="rec",
            vocab_path=norm_path(a.vocab),
        ).run()
    )

    sp = sub.add_parser("label-both", help="同时标注定位(det)与识别(rec)并自动生成两套 labels")
    sp.add_argument("--images", required=True, help="图片目录，例如 datasets/det/images/train")
    sp.add_argument("--det-labels", required=True, help="det labels 输出目录，例如 datasets/det/labels/train")
    sp.add_argument("--rec-labels", required=True, help="rec labels 输出目录，例如 datasets/rec/labels/train")
    sp.add_argument("--vocab", required=True, help="vocab 文件（UTF-8 每行一个字符）")
    sp.set_defaults(
        func=lambda a: DualLabeler(
            images_dir=norm_path(a.images),
            det_labels_dir=norm_path(a.det_labels),
            rec_labels_dir=norm_path(a.rec_labels),
            vocab_path=norm_path(a.vocab),
        ).run()
    )

    sp = sub.add_parser("label-assistant", help="标定助手：一张图完成定位框+识别内容，自动生成 det/rec labels")
    sp.add_argument("--images", required=True, help="图片目录，例如 Images/trains")
    sp.add_argument("--det-labels", required=True, help="det labels 输出目录")
    sp.add_argument("--rec-labels", required=True, help="rec labels 输出目录")
    sp.add_argument("--vocab", required=True, help="vocab 文件（UTF-8 每行一个字符）")
    sp.set_defaults(
        func=lambda a: AssistantLabeler(
            images_dir=norm_path(a.images),
            det_labels_dir=norm_path(a.det_labels),
            rec_labels_dir=norm_path(a.rec_labels),
            vocab_path=norm_path(a.vocab),
        ).run()
    )

    sp = sub.add_parser("train-det", help="训练定位(det)模型")
    sp.add_argument("--data", required=True, help="det.yaml 路径")
    _add_common_train_args(sp)
    sp.set_defaults(func=cmd_train_det)

    sp = sub.add_parser("train-rec", help="训练识别(rec)模型（检测+类别，支持旋转框 OBB）")
    sp.add_argument("--data", required=True, help="rec.yaml 路径")
    _add_common_train_args(sp)
    sp.set_defaults(func=cmd_train_rec)

    sp = sub.add_parser("prepare-rec-ctc", help="基于已标注 det/rec 自动生成 CTC 训练数据（无需重标）")
    sp.add_argument("--det-root", required=True, help="det 数据根目录，例如 datasets/det")
    sp.add_argument("--rec-root", required=True, help="rec 数据根目录，例如 datasets/rec")
    sp.add_argument("--out", required=True, help="输出目录，例如 datasets/rec_ctc")
    sp.add_argument("--vocab", required=True, help="vocab 文件，UTF-8 每行一个字符")
    sp.set_defaults(func=cmd_prepare_rec_ctc)

    sp = sub.add_parser("train-rec-ctc", help="训练 CRNN+CTC 识别模型")
    sp.add_argument("--train-manifest", required=True, help="训练清单，例如 datasets/rec_ctc/manifest_train.txt")
    sp.add_argument("--val-manifest", required=True, help="验证清单，例如 datasets/rec_ctc/manifest_val.txt")
    sp.add_argument("--vocab", required=True, help="vocab 文件")
    sp.add_argument("--project", type=str, default="models", help="输出目录")
    sp.add_argument("--name", type=str, default="rec_ctc_exp", help="实验名称")
    sp.add_argument("--epochs", type=int, default=80, help="训练轮数")
    sp.add_argument("--batch", type=int, default=32, help="batch")
    sp.add_argument("--lr", type=float, default=1e-3, help="学习率")
    sp.add_argument("--img-h", type=int, default=48, help="输入高")
    sp.add_argument("--img-w", type=int, default=320, help="输入宽")
    sp.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    sp.add_argument("--device", type=str, default="cpu", help="cpu 或 0/cuda")
    sp.set_defaults(func=cmd_train_rec_ctc)

    sp = sub.add_parser("infer-ctc", help="单张图 CTC 推理")
    sp.add_argument("--weights", required=True, help="CTC 权重 .pt")
    sp.add_argument("--image", required=True, help="输入图片")
    sp.add_argument("--img-h", type=int, default=0, help="输入高（0=跟随模型）")
    sp.add_argument("--img-w", type=int, default=0, help="输入宽（0=跟随模型）")
    sp.add_argument("--device", type=str, default="cpu", help="cpu 或 0/cuda")
    sp.add_argument("--out", type=str, default="", help="可选：输出 json")
    sp.set_defaults(func=cmd_infer_ctc)

    sp = sub.add_parser("infer", help="OCR 推理（定位→裁剪→识别）")
    sp.add_argument("--det-weights", required=True, help="定位权重 .pt")
    sp.add_argument("--rec-weights", required=True, help="识别权重 .pt")
    sp.add_argument("--image", required=True, help="输入图片")
    sp.add_argument("--device", type=str, default="cpu")
    sp.add_argument(
        "--vocab",
        type=str,
        default="",
        help="可选：类别到字符映射文件（UTF-8，每行一个字符，行号=class id）",
    )
    sp.add_argument("--out", type=str, default="", help="输出 json 路径（可选）")
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser("infer-sample", help="随机抽查多张图片推理并保存结果（json+可视化）")
    sp.add_argument("--det-weights", required=True, help="定位权重 .pt")
    sp.add_argument("--rec-weights", required=True, help="识别权重 .pt")
    sp.add_argument("--images", required=True, help="图片目录，例如 datasets/det/images/train")
    sp.add_argument("--num", type=int, default=5, help="抽样数量")
    sp.add_argument("--seed", type=int, default=0, help="随机种子")
    sp.add_argument("--device", type=str, default="cpu")
    sp.add_argument("--vocab", type=str, default="", help="可选 vocab 文件")
    sp.add_argument("--outdir", type=str, required=True, help="输出目录")
    sp.set_defaults(func=cmd_infer_sample)

    sp = sub.add_parser("check-obb-labels", help="检查 OBB 标注是否完整/格式正确")
    sp.add_argument("--images", required=True, help="图片目录，例如 datasets/det/images/train")
    sp.add_argument("--labels", required=True, help="labels 目录，例如 datasets/det/labels/train")
    sp.add_argument(
        "--num-classes",
        type=str,
        default="",
        help="可选：类别总数，用于检查 class id 是否越界",
    )
    sp.set_defaults(func=cmd_check_obb_labels)

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)

