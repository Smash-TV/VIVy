# -*- coding: utf-8 -*-
"""为冻结工程的不可变部分生成 SHA-256 清单。"""

from __future__ import annotations

import hashlib
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "provenance" / "frozen_manifest.sha256"
EXCLUDED_PREFIXES = (
    ".git/",
    "data/",
    "tmp/",
    "results/",
    ".venv/",
    "venv/",
    ".pytest_cache/",
    ".ruff_cache/",
    "build/",
    "dist/",
    "output/",
    "experiments/caps_situation_behavior_profile/results/",
    "experiments/caps_situation_behavior_profile/baselines/",
)
EXCLUDED_FILES = {"provenance/frozen_manifest.sha256"}


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if (
            relative in EXCLUDED_FILES
            or relative.startswith(EXCLUDED_PREFIXES)
            or "__pycache__" in path.parts
            or any(part.endswith(".egg-info") for part in path.parts)
            or path.name.startswith(".env")
            or path.suffix == ".pyc"
        ):
            continue
        rows.append(f"{_hash_file(path)}  {relative}")
    MANIFEST_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"已写入 {len(rows)} 个冻结文件哈希：{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
