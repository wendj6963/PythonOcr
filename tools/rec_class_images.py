from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Hit:
    image: str
    label_file: str


def _load_vocab(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _iter_label_files(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        if r.is_file() and r.suffix.lower() == ".txt":
            out.append(r)
            continue
        if r.is_dir():
            out.extend(p for p in r.rglob("*.txt") if p.is_file())
    return sorted(set(out))


def _resolve_class(target: str, vocab: list[str]) -> int | None:
    if target.isdigit():
        idx = int(target)
        return idx if 0 <= idx < len(vocab) else None
    target = target.strip()
    if not target:
        return None
    if target in vocab:
        return vocab.index(target)
    return None


def _scan_file(p: Path, class_id: int) -> bool:
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 9:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if cls == class_id:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="统计指定类别对应的图片名称")
    parser.add_argument("--target", required=True, help="类别 ID 或字符，例如 13 或 G")
    parser.add_argument("--vocab", default="vocab_rec.txt", help="词表路径")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["datasets/rec/labels_all"],
        help="labels 目录或文件（可多个）",
    )
    parser.add_argument("--out", default="datasets/rec/class_images.txt", help="输出路径")
    args = parser.parse_args()

    vocab = _load_vocab(Path(args.vocab))
    class_id = _resolve_class(args.target, vocab)
    if class_id is None:
        print("无法解析目标类别，请输入有效的类别 ID 或字符。")
        return 1

    label_files = _iter_label_files([Path(p) for p in args.labels])
    hits: list[Hit] = []
    for p in label_files:
        if _scan_file(p, class_id):
            hits.append(Hit(image=p.stem, label_file=str(p)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("=== 类别图片统计 ===")
    lines.append(f"target={args.target} (class_id={class_id})")
    lines.append(f"labels_files={len(label_files)}")
    lines.append(f"hits={len(hits)}")
    lines.append("")
    for h in hits:
        lines.append(f"{h.image}\t{h.label_file}")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"已输出: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

