from __future__ import annotations

import argparse
from pathlib import Path


def _count_vocab(vocab_path: Path) -> int:
    lines = [ln.strip() for ln in vocab_path.read_text(encoding="utf-8").splitlines()]
    return len([ln for ln in lines if ln])


def _scan_labels(labels_dirs: list[Path], max_cls: int) -> list[str]:
    rows: list[str] = []
    for d in labels_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.txt")):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                parts = s.split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    continue
                if cls >= max_cls:
                    rows.append(f"{p.name}: {s}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="List rec label rows with class id out of range.")
    parser.add_argument("--vocab", required=True, help="vocab file path")
    parser.add_argument(
        "--labels-root",
        default="datasets/rec/labels",
        help="labels root (expects train/val under it)",
    )
    parser.add_argument(
        "--out",
        default="datasets/rec/out_of_range.txt",
        help="output file path",
    )
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    labels_root = Path(args.labels_root)
    out_path = Path(args.out)

    max_cls = _count_vocab(vocab_path)
    rows = _scan_labels(
        [labels_root / "train", labels_root / "val"],
        max_cls=max_cls,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    else:
        out_path.write_text("OK\n", encoding="utf-8")

    print(f"vocab_classes={max_cls}")
    print(f"out={out_path}")
    print(f"out_of_range={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

