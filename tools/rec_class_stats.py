from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Stats:
    files: int = 0
    lines: int = 0


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


def _count_labels(files: list[Path]) -> tuple[dict[int, int], Stats, list[str]]:
    stats = Stats()
    counts: dict[int, int] = {}
    bad_lines: list[str] = []

    for p in files:
        stats.files += 1
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 9:
                bad_lines.append(f"{p}: {s}")
                continue
            try:
                cls = int(float(parts[0]))
            except Exception:
                bad_lines.append(f"{p}: {s}")
                continue
            counts[cls] = counts.get(cls, 0) + 1
            stats.lines += 1
    return counts, stats, bad_lines


def main() -> int:
    parser = argparse.ArgumentParser(description="统计 rec 标签各类别数量")
    parser.add_argument("--vocab", default="vocab_rec.txt", help="词表路径")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["datasets/rec/labels_all"],
        help="labels 目录或文件（可多个）",
    )
    parser.add_argument("--out", default="datasets/rec/class_report.txt", help="报告输出路径")
    args = parser.parse_args()

    vocab = _load_vocab(Path(args.vocab))
    files = _iter_label_files([Path(p) for p in args.labels])
    counts, stats, bad_lines = _count_labels(files)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=== rec 类别统计报告 ===")
    lines.append(f"labels 文件数: {stats.files}")
    lines.append(f"标注行数: {stats.lines}")
    lines.append("")
    lines.append("class_id\tchar\tcount")
    for idx, ch in enumerate(vocab):
        lines.append(f"{idx}\t{ch}\t{counts.get(idx, 0)}")

    if bad_lines:
        lines.append("")
        lines.append("格式异常行:")
        lines.extend(bad_lines)

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"已输出: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

