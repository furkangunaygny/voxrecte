"""Helpers for writing the CSV and JSON data outputs."""

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path


def write_json(path: Path, data) -> None:
    """Pretty-printed UTF-8 JSON (non-ASCII characters kept)."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence]) -> None:
    """CSV with a header row."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_dict_csv(path: Path, rows: Sequence[dict]) -> None:
    """CSV with the keys of the first row as header."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
