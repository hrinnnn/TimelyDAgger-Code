"""Training-free Bridge-PCA monitor from the TimelyDAgger method.

The caller supplies frozen VLA bridge features. No particular VLA, robot,
environment, or experiment is required by this module.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BridgePCA:
    """Fit an ID principal subspace and score its reconstruction residual.

    The score is ||(I - V V^T)(z - mu)||_2, as in the paper. ``rank`` is fixed
    before evaluation and counts retained principal directions.
    """

    def __init__(self, rank: int) -> None:
        if rank < 1:
            raise ValueError("rank must be at least one")
        self.rank = rank
        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.threshold_: float | None = None

    def fit(self, id_features: ArrayLike) -> BridgePCA:
        """Fit PCA to one feature vector per ID observation, shape (N, d)."""
        features = np.asarray(id_features, dtype=np.float64)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("id_features must be a finite (N, d) array")
        n, d = features.shape
        if n < 2 or not 1 <= self.rank < min(n, d):
            raise ValueError("rank must be smaller than both N and d")

        self.mean_ = features.mean(axis=0)
        centered = features - self.mean_
        # Centered SVD and covariance eigendecomposition span the same PCA
        # directions; SVD avoids constructing a potentially large d-by-d matrix.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.rank]
        self.threshold_ = None
        return self

    def score(self, bridge_feature: ArrayLike) -> float:
        """Return the residual norm for one bridge feature of shape (d,)."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("call fit before score")
        feature = np.asarray(bridge_feature, dtype=np.float64)
        if feature.shape != self.mean_.shape or not np.isfinite(feature).all():
            raise ValueError("bridge_feature must be one finite d-vector")
        centered = feature - self.mean_
        residual = centered - self.components_.T @ (self.components_ @ centered)
        return float(np.linalg.norm(residual))

    def calibrate(
        self, id_rollouts: Sequence[ArrayLike], *, alpha: float = 0.05
    ) -> float:
        """Set the upper-tail ID threshold from per-rollout maximum scores.

        ``id_rollouts`` is a separate collection of successful ID rollouts.
        Each rollout is an array of bridge features with shape (T_i, d).
        ``method='higher'`` makes the finite-sample empirical quantile
        conservative relative to interpolating between calibration maxima.
        This is an implementation choice, not a formal coverage guarantee.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if len(id_rollouts) == 0:
            raise ValueError("at least one calibration rollout is required")

        maxima: list[float] = []
        for rollout in id_rollouts:
            features = np.asarray(rollout, dtype=np.float64)
            if features.ndim != 2 or features.shape[0] == 0:
                raise ValueError("each rollout must contain at least one feature")
            maxima.append(max(self.score(z) for z in features))
        self.threshold_ = float(np.quantile(maxima, 1 - alpha, method="higher"))
        return self.threshold_

    def requests_help(self, bridge_feature: ArrayLike, threshold: float | None = None) -> bool:
        """Request takeover only for a strict threshold exceedance."""
        gamma = self.threshold_ if threshold is None else threshold
        if gamma is None:
            raise RuntimeError("calibrate first or supply a threshold")
        return self.score(bridge_feature) > gamma
