"""The result-row schema. One row per (cell); columns are fixed and typed; absent values are nulls.

Every number in a report must trace back to a row with these columns (provenance included: backend, hardware, versions,
seed, git sha when available).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

SCHEMA_VERSION = 1

_F = pa.float64()
_I = pa.int64()
_S = pa.string()

FIELDS: list[tuple[str, pa.DataType, str]] = [
    # identity
    ("schema_version", pa.int32(), "row schema version"),
    ("experiment", _S, "experiment name"),
    ("cell_id", _S, "content hash of the cell; resume key"),
    ("config_hash", _S, "fingerprint of the experiment config"),
    ("dataset", _S, ""), ("normalize", pa.bool_(), "L2-normalised vectors"),
    ("n", _I, "training vectors"), ("k", _I, "clusters"), ("d", pa.int32(), "dimension"),
    ("algo", _S, "lloyd | emeans | omeans"), ("backend", _S, "reference | official (never mixed within a row)"),
    ("adapter", _S, "official adapter key"), ("variant", _S, "implementation string"), ("recipe", _S, "e/o-means recipe"),
    ("params_json", _S, "algorithm kwargs"), ("budget_kind", _S, "iterations | epochs"), ("budget", _F, ""),
    ("seed", pa.int32(), ""), ("role", _S, "baseline | run"),
    # quality (exact assignment, float64)
    ("train_eval_n", _I, "size of the train subset used for train MSE / cluster stats"),
    ("train_mse", _F, ""), ("val_mse", _F, ""), ("test_mse", _F, ""),
    ("rel_test_vs_lloyd25_ref", _F, "(test_mse - L25) / L25, L25 = reference Lloyd@25 of the SAME seed"),
    ("rel_train_vs_lloyd25_ref", _F, ""),
    ("rel_test_vs_lloyd25_official", _F, "same, vs the official (Faiss) Lloyd@25, when present"),
    ("rel_train_vs_lloyd25_official", _F, ""),
    ("dead_clusters", _I, "< 10 points in exact train assignment"), ("empty_clusters", _I, ""),
    ("imbalance_factor", _F, "k*sum(size^2)/n^2"), ("max_over_mean", _F, ""),
    # time (seconds)
    ("t_init", _F, ""), ("t_setup", _F, "o-means: initial cell build"), ("t_fit", _F, ""), ("t_finalize", _F, ""),
    # operation counts (hardware independent)
    ("distance_computations", _F, "assignment + index maintenance"), ("assign_passes", _F, "distance_computations / (n*k)"),
    ("assign_distance_computations", _F, ""), ("maintenance_distance_computations", _F, ""),
    # algorithm diagnostics
    ("batch_size", _I, ""), ("alpha0", _F, ""), ("n_updates", _I, ""), ("n_revived_total", _I, ""),
    ("nlist", _I, ""), ("nprobe", _I, ""), ("missed_nearest_mean", _F, ""),
    ("assign_fraction_actual", _F, "o-means: actual comparisons / exact"), ("assign_fraction_nominal", _F, ""),
    ("finish", _S, "last_epoch_mean | recency | none | exact_pass"),
    # provenance
    ("status", _S, "ok | failed"), ("error", _S, ""), ("keo_version", _S, ""), ("git_sha", _S, ""),
    ("torch_threads", pa.int32(), ""), ("device", _S, ""), ("peak_rss_mb", _F, "process high-water mark so far"),
    ("hardware_json", _S, ""), ("timestamp_utc", _S, ""),
]

COLUMNS: list[str] = [f[0] for f in FIELDS]
SCHEMA = pa.schema([pa.field(name, typ) for name, typ, _ in FIELDS])


def new_row() -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    row["schema_version"] = SCHEMA_VERSION
    return row


def validate_row(row: dict[str, Any]) -> None:
    """Raises ValueError on unknown/missing columns or type errors."""
    extra, missing = set(row) - set(COLUMNS), set(COLUMNS) - set(row)
    if extra or missing:
        raise ValueError(f"row columns mismatch: unknown={sorted(extra)} missing={sorted(missing)}")
    try:
        pa.Table.from_pylist([row], schema=SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as e:
        raise ValueError(f"row does not match schema: {e}") from e


def row_to_table(row: dict[str, Any]) -> pa.Table:
    validate_row(row)
    return pa.Table.from_pylist([row], schema=SCHEMA)
