# -*- coding: utf-8 -*-
"""验证冻结文件 SHA-256、SQLite 完整性和核心代码可导入性。"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "provenance" / "frozen_manifest.sha256"
DATABASE_INVENTORY_PATH = PROJECT_ROOT / "provenance" / "database_inventory.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest() -> list[str]:
    errors = []
    for line_number, line in enumerate(
        MANIFEST_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative_path = line.split("  ", 1)
        except ValueError:
            errors.append(f"清单第 {line_number} 行格式错误")
            continue
        path = PROJECT_ROOT / pathlib.PurePosixPath(relative_path)
        if not path.is_file():
            errors.append(f"缺少文件：{relative_path}")
            continue
        actual = _hash_file(path)
        if actual != expected:
            errors.append(f"哈希不一致：{relative_path}")
    return errors


def _verify_databases() -> list[str]:
    errors = []
    inventory = json.loads(DATABASE_INVENTORY_PATH.read_text(encoding="utf-8"))
    for item in inventory["databases"]:
        path = PROJECT_ROOT / pathlib.PurePosixPath(item["path"])
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                errors.append(f"数据库完整性失败：{item['path']} -> {integrity}")
            for table, expected_count in item["row_counts"].items():
                actual_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
                if actual_count != expected_count:
                    errors.append(
                        f"数据库行数不一致：{item['path']} {table} "
                        f"expected={expected_count} actual={actual_count}"
                    )
        finally:
            connection.close()
    return errors


def main() -> None:
    errors = _verify_manifest() + _verify_databases()
    from gwen_agent.modules.personality_core import PersonalityCore

    if PersonalityCore is None:
        errors.append("人格内核导入失败")
    if errors:
        print("冻结验证失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(
        "公开副本文件哈希与核心导入验证通过；数据库仅按清单验证，空清单不代表原始数据库已验证。"
    )


if __name__ == "__main__":
    main()
