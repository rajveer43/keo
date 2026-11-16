"""Resumable result store: one small parquet file per cell, consolidated on demand.

Files are `<out>/rows/<cell_id>.ok.parquet` or `.failed.parquet` (status in the name makes resume checks free). Writes are
atomic (temp file + rename), so an interrupted run never leaves a half-written row. A successful re-run of a failed cell
removes its `.failed` file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from keo.bench.schema import SCHEMA, row_to_table


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.rows_dir = self.root / "rows"
        self.rows_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cell_id: str, status: str) -> Path:
        return self.rows_dir / f"{cell_id}.{status}.parquet"

    def has_ok(self, cell_id: str) -> bool:
        return self._path(cell_id, "ok").exists()

    def has_failed(self, cell_id: str) -> bool:
        return self._path(cell_id, "failed").exists()

    def done(self, cell_id: str, retry_failed: bool = False) -> bool:
        """A cell is done if it has an ok row, or a failed row and we are not retrying failures."""
        return self.has_ok(cell_id) or (self.has_failed(cell_id) and not retry_failed)

    def write(self, row: dict[str, Any]) -> Path:
        status = row["status"]
        if status not in {"ok", "failed"}:
            raise ValueError(f"bad status {status!r}")
        table = row_to_table(row)
        final = self._path(row["cell_id"], status)
        tmp = final.with_suffix(".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, final)
        if status == "ok":
            self._path(row["cell_id"], "failed").unlink(missing_ok=True)
        return final

    def read_row(self, cell_id: str) -> dict[str, Any] | None:
        for status in ("ok", "failed"):
            p = self._path(cell_id, status)
            if p.exists():
                return pq.read_table(p).to_pylist()[0]
        return None

    def read_all(self) -> pd.DataFrame:
        files = sorted(self.rows_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame({name: pd.Series(dtype="object") for name in SCHEMA.names})
        table = pa.concat_tables([pq.read_table(f) for f in files])
        return table.to_pandas()

    def consolidate(self) -> Path:
        out = self.root / "results.parquet"
        files = sorted(self.rows_dir.glob("*.parquet"))
        if files:
            pq.write_table(pa.concat_tables([pq.read_table(f) for f in files]), out)
        return out

    def counts(self) -> dict[str, int]:
        return {
            "ok": len(list(self.rows_dir.glob("*.ok.parquet"))),
            "failed": len(list(self.rows_dir.glob("*.failed.parquet"))),
        }
