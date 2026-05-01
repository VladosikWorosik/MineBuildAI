from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCHEMATIC_EXTENSIONS = (".schem", ".schematic", ".litematic", ".nbt", ".zip")


@dataclass(frozen=True)
class ManifestRecord:
    prompt: str
    path: str
    sha256: str
    bytes: int
    source_url: str | None = None
    source_page: str | None = None


def clean_prompt(value: str) -> str:
    value = re.sub(r"[_\-+.]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or "minecraft build"


def prompt_from_path(path: Path) -> str:
    name = path.name
    for suffix in sorted(SCHEMATIC_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return clean_prompt(name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_schematic_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SCHEMATIC_EXTENSIONS:
            yield path


def write_jsonl(path: Path, records: Iterable[ManifestRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, record: ManifestRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def build_manifest(schematics_dir: Path, out_path: Path) -> int:
    records = []
    for path in iter_schematic_files(schematics_dir):
        stat = path.stat()
        records.append(
            ManifestRecord(
                prompt=prompt_from_path(path),
                path=str(path),
                sha256=file_sha256(path),
                bytes=stat.st_size,
            )
        )
    return write_jsonl(out_path, records)
