from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BadItem:
    image: str
    text: str
    reason: str
    source: str


def _has_space(text: str) -> bool:
    return any(ch.isspace() for ch in text)


def _has_invalid_char(text: str, allowed: set[str]) -> bool:
    for ch in text:
        if ch.isspace():
            continue
        if ch not in allowed:
            return True
    return False


def _scan_meta_file(p: Path, allowed: set[str]) -> list[BadItem]:
    out: list[BadItem] = []
    try:
        payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return out

    image = str(payload.get("image") or p.stem.replace(".assistant", ""))
    items = payload.get("items", [])
    if not isinstance(items, list):
        return out

    for it in items:
        text = str(it.get("text", ""))
        if not text:
            continue
        if _has_space(text):
            out.append(BadItem(image=image, text=text, reason="包含空格", source=str(p)))
            continue
        if _has_invalid_char(text, allowed):
            out.append(BadItem(image=image, text=text, reason="包含非法字符", source=str(p)))
    return out


def _collect_meta_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.endswith(".assistant.json"):
            files.append(root)
            continue
        if root.is_dir():
            files.extend(root.rglob("*.assistant.json"))
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description="检查标定输入内容是否含空格/非法字符")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[],
        help="标注目录（包含 *.assistant.json）。可传多个路径。",
    )
    parser.add_argument(
        "--allowed",
        default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="允许的字符集（默认 0-9 + A-Z）。",
    )
    parser.add_argument(
        "--out",
        default="datasets/det/bad_text.txt",
        help="输出报告路径。",
    )
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots] if args.roots else []
    if not roots:
        roots = [
            Path("datasets/det/labels_all"),
            Path("datasets/det/labels/train"),
            Path("datasets/det/labels/val"),
        ]

    allowed = set(args.allowed)

    meta_files = _collect_meta_files(roots)
    bad: list[BadItem] = []
    for p in meta_files:
        bad.extend(_scan_meta_file(p, allowed))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=== 标定输入内容检查报告 ===")
    lines.append(f"扫描文件数: {len(meta_files)}")
    lines.append(f"问题条目数: {len(bad)}")
    lines.append("")

    for b in bad:
        lines.append(f"图片: {b.image}")
        lines.append(f"原因: {b.reason}")
        lines.append(f"内容: {b.text}")
        lines.append(f"来源: {b.source}")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"已输出: {out_path}")
    if bad:
        print(f"发现问题条目: {len(bad)}")
    else:
        print("未发现问题内容。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
