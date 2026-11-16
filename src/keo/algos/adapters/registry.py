"""Which official implementations can run on THIS machine, and why not. Never raises."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from keo.algos.adapters import faiss_cpu, faiss_official, flash_kmeans


@dataclass(frozen=True)
class AdapterStatus:
    key: str  # stable id used in configs
    algo: str  # lloyd | emeans | omeans
    description: str
    available: bool
    reason: str
    verified: bool  # True = API read from an installed package on this machine; False = written from docs only

    def as_dict(self) -> dict:
        return asdict(self)


def _safe(fn, *a) -> tuple[bool, str]:
    try:
        return fn(*a)
    except Exception as e:  # availability probing must never take the benchmark down
        return False, f"probe failed: {type(e).__name__}: {e}"


def adapter_status() -> list[AdapterStatus]:
    rows = [
        ("faiss_lloyd_cpu", "lloyd", "Faiss CPU Lloyd k-means", _safe(faiss_cpu.is_available, False), True),
        ("faiss_lloyd_gpu", "lloyd", "Faiss GPU Lloyd k-means", _safe(faiss_cpu.is_available, True), False),
        ("flash_kmeans", "lloyd", "Flash-KMeans (GPU Lloyd)", _safe(flash_kmeans.is_available), False),
        ("official_emeans_cpu", "emeans", "faiss.contrib.e_means.NumpyEmeans (CPU)", _safe(faiss_official.emeans_available), True),
        ("official_omeans", "omeans", "faiss-fork contrib.o_means.Omeans (CUDA)", _safe(faiss_official.omeans_available), False),
    ]
    return [AdapterStatus(k, a, desc, ok, why, ver) for k, a, desc, (ok, why), ver in rows]


def get_adapter(key: str, **kwargs):
    """Instantiate an adapter by key, or raise AdapterUnavailable with the reason."""
    from keo.algos.adapters._worker_client import AdapterUnavailable

    table = {
        "faiss_lloyd_cpu": lambda: faiss_cpu.FaissLloyd(gpu=False, **kwargs),
        "faiss_lloyd_gpu": lambda: faiss_cpu.FaissLloyd(gpu=True, **kwargs),
        "flash_kmeans": lambda: flash_kmeans.FlashKMeans(**kwargs),
        "official_emeans_cpu": lambda: faiss_official.OfficialEmeans(**kwargs),
        "official_omeans": lambda: faiss_official.OfficialOmeans(**kwargs),
    }
    if key not in table:
        raise KeyError(f"unknown adapter {key!r}; known: {sorted(table)}")
    st = {s.key: s for s in adapter_status()}[key]
    if not st.available:
        raise AdapterUnavailable(f"{key}: {st.reason}")
    return table[key]()
