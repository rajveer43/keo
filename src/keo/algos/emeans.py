"""[E] e-means, reference backend (PyTorch). Spec: docs/SPEC.md §4; source [E §3, Alg. 1, App. H].

Mini-batch k-means whose centroids are EMA ratios c_j = S_j / N_j:
    N_j <- gamma N_j + (1-gamma) n_j ;  S_j <- gamma S_j + (1-gamma) s_j        (Eq. 2-4)
with a cosine-annealed step alpha = 1 - gamma, scaling-law batch size / peak step, dead-cluster revival every 5
batches, and a last-epoch-mean finalisation.

The training loop is written once and parametrised by three hooks so that o-means (which "keeps everything from
e-means" [O §03]) reuses it verbatim and cannot drift from it:
  * `_new_assigner`   -> object doing the nearest-centroid search (exact here; two-level IVF in o-means)
  * `_use_recency`    -> whether the closing mean is recency-weighted (never for e-means)
  * `_recency_tau`    -> time constant of that weighting (epochs)

Ablation switches (all default to the paper's recipe): cosine, revival, finalize, init.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from keo.algos import laws
from keo.algos.base import Budget, FitResult
from keo.algos.init import kmeanspp_init, random_row_init
from keo.algos.revival import BF16_EPS, revive_resample, revive_split
from keo.metrics.core import DEFAULT_CHUNK_ELEMS, as_tensor, assign

N_INIT = 1e-10  # [E §3.1 "Initialization"] (the paper's value; the shipped code uses 2^-16, see `recipe`)
_FINAL_CHUNK = 65536
_TINY = torch.finfo(torch.float32).tiny


class ExactAssigner:
    """Brute-force nearest centroid: B*k distance computations per batch."""

    def __init__(self, k: int, chunk_elems: int) -> None:
        self.k, self.chunk_elems = k, chunk_elems
        self.assign_ops = 0.0
        self.maintenance_ops = 0.0

    def start(self, C: torch.Tensor) -> None:
        pass

    def assign(self, xb: torch.Tensor, C: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.assign_ops += float(xb.shape[0]) * self.k
        return assign(xb, C, self.chunk_elems)

    def on_revival(self, pairs: list[tuple[int, int]]) -> None:
        pass

    def after_batch(self, t: int, nb: int, get_C: Callable[[], torch.Tensor]) -> None:
        pass

    def diagnostics(self) -> dict[str, Any]:
        return {}


class RecencyAccumulator:
    """Recency-weighted per-cluster sums/counts for the o-means short-run closing mean (A18).

    Every batch: A <- A*decay + s_batch ; W <- W*decay + n_batch with decay = exp(-1/(tau*batches_per_epoch)).
    On a split (dead, donor): half of the donor's accumulated sums go to the child [O §04.4].
    """

    def __init__(self, k: int, d: int, batches_per_epoch: int, tau_epochs: float) -> None:
        self.decay = math.exp(-1.0 / (tau_epochs * batches_per_epoch))
        self.A = torch.zeros(k, d, dtype=torch.float64)
        self.W = torch.zeros(k, dtype=torch.float64)

    def update(self, labels: torch.Tensor, xb: torch.Tensor) -> None:
        self.A.mul_(self.decay)
        self.W.mul_(self.decay)
        self.A.index_add_(0, labels, xb.double())
        self.W += torch.bincount(labels, minlength=self.W.shape[0]).double()

    def split(self, pairs: list[tuple[int, int]]) -> None:
        for dead, donor in pairs:
            self.A[donor] = self.A[donor] / 2
            self.W[donor] = self.W[donor] / 2
            self.A[dead] = self.A[donor]
            self.W[dead] = self.W[donor]

    def centroids(self, fallback: torch.Tensor) -> torch.Tensor:
        out = fallback.clone()
        has = self.W > 0
        out[has] = (self.A[has] / self.W[has, None]).float()
        return out


class EMeans:
    name = "emeans"
    backend = "reference"

    def __init__(
        self,
        init: str = "random",
        revival: str = "split",  # "split" | "resample" | "off"        [E App. H.2]
        cosine: bool = True,  # False -> constant step alpha0           (ablation, A40)
        finalize: str = "last_epoch_mean",  # | "none" | "exact_pass"  [E §3.2, A2]
        tau: float = 0.1,
        eps: float = laws.COSINE_EPS,
        delta: float | None = None,  # None -> by recipe: paper 2^-7 (BF16 eps), official 1e-4
        n_init: float | None = None,  # None -> by recipe: paper 1e-10, official 2^-16
        recipe: str = "official",  # "official" (default, study decision A52): constants shipped in faiss.contrib.e_means; "paper": as printed in the paper
        revival_every: int = 5,
        batch_size: int | None = None,  # None -> law (Eq. 7)
        alpha0: float | None = None,  # None -> law (Eq. 6)
        chunk_elems: int = DEFAULT_CHUNK_ELEMS,
    ) -> None:
        if init not in {"random", "kmeanspp"}:
            raise ValueError(f"init must be 'random' or 'kmeanspp', got {init!r}")
        if revival not in {"split", "resample", "off"}:
            raise ValueError(f"revival must be split|resample|off, got {revival!r}")
        if finalize not in {"last_epoch_mean", "none", "exact_pass"}:
            raise ValueError(f"finalize must be last_epoch_mean|none|exact_pass, got {finalize!r}")
        if recipe not in {"paper", "official"}:
            raise ValueError(f"recipe must be 'paper' or 'official', got {recipe!r}")
        self.recipe = recipe
        official = recipe == "official"
        self.n_init = float(n_init) if n_init is not None else (laws.OFFICIAL_N_INIT if official else N_INIT)
        delta = delta if delta is not None else (laws.OFFICIAL_SPLIT_DELTA if official else BF16_EPS)
        self.init, self.revival, self.cosine, self.finalize = init, revival, cosine, finalize
        self.tau, self.eps, self.delta, self.revival_every = tau, eps, delta, revival_every
        self.batch_size, self.alpha0, self.chunk_elems = batch_size, alpha0, chunk_elems

    # ------------------------------------------------------------------ hooks (overridden by OMeans)
    def _new_assigner(self, k: int, d: int, nb: int, seed_seq: np.random.SeedSequence) -> ExactAssigner:
        return ExactAssigner(k, self.chunk_elems)

    def _use_recency(self, T: float) -> bool:
        return False

    def _recency_tau(self) -> float:
        return 1.0

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: np.ndarray,
        k: int,
        budget: Budget,
        seed: int,
        init_centroids: np.ndarray | None = None,
    ) -> FitResult:
        if budget.epochs is None:
            raise ValueError(f"{type(self).__name__} needs an epochs budget")
        T = float(budget.epochs)
        n, d = X.shape
        Xt = as_tensor(X)

        revival_mode = self.revival
        if self.recipe == "official":
            B_law, a0_law, law_name = laws.official_recipe(n, k, T, d)
            if d <= laws.OFFICIAL_NO_REVIVAL_MAX_D:
                revival_mode = "off"  # the shipped code turns revival off for d <= 2
        else:
            B_law, law_name = laws.batch_size(n, k, T), "paper"
            a0_law = None
        B = int(self.batch_size) if self.batch_size is not None else B_law
        B = min(max(B, 1), n)
        if self.alpha0 is not None:
            a0 = float(self.alpha0)
        elif self.recipe == "official":
            ca, ak, ab, an, at = laws.OFFICIAL_ALPHA  # re-evaluate at the (possibly overridden) B
            a0 = min(max(ca * k**ak * B**ab * n**an * T**at, laws.OFFICIAL_ALPHA_CLAMP[0]), laws.OFFICIAL_ALPHA_CLAMP[1])
        else:
            a0 = laws.alpha0(n, k, T, B)
        nb = n // B  # batches per epoch (remainder points dropped each epoch, A2)
        M = laws.n_updates(n, B, T)
        if M < 1:
            raise ValueError(f"budget too small: T={T}, n={n}, B={B} gives zero updates")
        L = min(M, nb)  # "last epoch" = last L updates (A4)

        t0 = time.perf_counter()
        if init_centroids is not None:
            C0, init_name = np.ascontiguousarray(init_centroids, dtype=np.float32), "given"
        elif self.init == "kmeanspp":
            C0, init_name = kmeanspp_init(X, k, seed)[0], "kmeanspp"
        else:
            C0, init_name = random_row_init(X, k, seed)[0], "random"
        t_init = time.perf_counter() - t0

        ss_shuffle, ss_revive, ss_extra = np.random.SeedSequence(seed).spawn(3)
        rng_shuffle, rng_revive = np.random.default_rng(ss_shuffle), np.random.default_rng(ss_revive)

        C_valid = torch.from_numpy(C0.copy())  # last well-defined centroid per cluster (A39)
        N = torch.full((k,), self.n_init, dtype=torch.float32)
        S = C_valid * N[:, None]

        def current_centroids() -> torch.Tensor:
            return torch.where((N > 0)[:, None], S / N.clamp_min(_TINY)[:, None], C_valid)

        use_recency = self._use_recency(T)
        record = self.finalize == "last_epoch_mean"
        fsum = torch.zeros(k, d, dtype=torch.float64) if record else None
        fcnt = torch.zeros(k, dtype=torch.float64) if record else None
        rec = RecencyAccumulator(k, d, nb, self._recency_tau()) if use_recency else None
        train_labels = np.full(n, -1, dtype=np.int64)
        batch_mse: list[float] = []
        dead_per_check: list[int] = []
        n_revived = n_skipped = 0
        perm = np.empty(0, dtype=np.int64)

        assigner = self._new_assigner(k, d, nb, ss_extra)
        t_setup0 = time.perf_counter()
        assigner.start(C_valid)
        t_setup = time.perf_counter() - t_setup0

        ts = time.perf_counter()
        for t in range(M):
            if t % nb == 0:
                perm = rng_shuffle.permutation(n)
            b = t % nb
            idx = perm[b * B : (b + 1) * B]
            xb = Xt[torch.from_numpy(idx)]

            C = current_centroids()
            C_valid = C
            labels, d2 = assigner.assign(xb, C, t)
            batch_mse.append(float(d2.double().mean()))

            nj = torch.bincount(labels, minlength=k).to(torch.float32)
            sj = torch.zeros(k, d, dtype=torch.float32).index_add_(0, labels, xb)
            alpha = laws.alpha_at(t, M, a0, self.eps) if self.cosine else a0
            gamma = 1.0 - alpha
            N.mul_(gamma).add_(nj, alpha=alpha)
            S.mul_(gamma).add_(sj, alpha=alpha)

            if rec is not None:
                rec.update(labels, xb)
            if record and t >= M - L:
                fsum.index_add_(0, labels, xb.double())  # type: ignore[union-attr]
                fcnt += torch.bincount(labels, minlength=k).double()  # type: ignore[operator]
                train_labels[idx] = labels.numpy()

            if revival_mode != "off" and (t + 1) % self.revival_every == 0:
                if revival_mode == "split":
                    st = revive_split(S, N, self.tau, rng_revive, self.delta)
                    if st.pairs:
                        assigner.on_revival(st.pairs)
                        if rec is not None:
                            rec.split(st.pairs)
                else:
                    st = revive_resample(S, N, self.tau, xb, rng_revive)
                dead_per_check.append(st.n_dead)
                n_revived += st.n_revived
                n_skipped += st.n_skipped

            assigner.after_batch(t, nb, current_centroids)
        t_fit = time.perf_counter() - ts

        tf = time.perf_counter()
        C_ema = current_centroids()
        extra_dist = 0.0
        finish = "none"
        if use_recency and rec is not None:
            C_final = rec.centroids(C_ema)
            finish = "recency"
        elif self.finalize == "last_epoch_mean":
            has = fcnt > 0  # type: ignore[operator]
            C_final = C_ema.clone()
            C_final[has] = (fsum[has] / fcnt[has, None]).float()  # type: ignore[index]
            finish = "last_epoch_mean"
        elif self.finalize == "exact_pass":
            lab_all, _ = assign(Xt, C_ema, self.chunk_elems)
            sums = torch.zeros(k, d, dtype=torch.float64)
            for s in range(0, n, _FINAL_CHUNK):
                sums.index_add_(0, lab_all[s : s + _FINAL_CHUNK], Xt[s : s + _FINAL_CHUNK].double())
            cnt = torch.bincount(lab_all, minlength=k).double()
            has = cnt > 0
            C_final = C_ema.clone()
            C_final[has] = (sums[has] / cnt[has, None]).float()
            train_labels = lab_all.numpy().copy()
            extra_dist = float(n) * k
            finish = "exact_pass"
        else:
            C_final = C_ema
        t_final = time.perf_counter() - tf

        total_dist = assigner.assign_ops + assigner.maintenance_ops + extra_dist
        op_counts = {
            "distance_computations": total_dist,
            "assign_passes": total_dist / (float(n) * k),
            "assign_distance_computations": assigner.assign_ops,
            "maintenance_distance_computations": assigner.maintenance_ops,
            "exact_equivalent_distance_computations": float(M) * B * k,
        }
        return FitResult(
            centroids=C_final.numpy().copy(),
            train_labels=train_labels,  # labels seen during the last epoch (-1 = not visited); approximate
            timings={"init": t_init, "setup": t_setup, "fit": t_fit, "finalize": t_final},
            op_counts=op_counts,
            diagnostics={
                "init": init_name,
                "epochs": T,
                "n": n,
                "k": k,
                "batch_size": B,
                "alpha0": a0,
                "batches_per_epoch": nb,
                "n_updates": M,
                "last_epoch_updates": L,
                "revival": revival_mode,
                "recipe": self.recipe,
                "batch_law": law_name,
                "n_init": self.n_init,
                "split_delta": self.delta,
                "cosine": self.cosine,
                "finalize": self.finalize,
                "finish": finish,
                "batch_mse": batch_mse,
                "dead_per_revival_check": dead_per_check,
                "n_revived_total": n_revived,
                "n_revival_skipped": n_skipped,
                "n_revival_checks": len(dead_per_check),
                "centroids_before_finalize": C_ema.numpy().copy(),
                **assigner.diagnostics(),
            },
        )
