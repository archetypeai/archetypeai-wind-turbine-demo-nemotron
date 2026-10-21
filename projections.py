"""2D projection of the Omega embedding space for the embedding panel.

PCA-2 is fit once on the n-shot library (the joint [4 channels × 768] vectors KNN
votes against), then every live window is projected through the same axes, so
the live trail lands in the library's coordinates. PCA rather than t-SNE: t-SNE
has no transform for new points, so the live cursor would force a refit per tick.

Leave-one-out KNN accuracy is computed in the full embedding space, not 2D: it
says whether the classifier separates the classes, however the 2D picture looks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Projection:
    mean: np.ndarray
    axes: np.ndarray            # (2, d)
    explained: list[float]      # variance ratio of PC1, PC2
    library: list[dict]         # [{x, y, label}]
    loo_accuracy: float

    def project(self, vec: list[float]) -> list[float]:
        x, y = self.axes @ (np.asarray(vec) - self.mean)
        return [round(float(x), 4), round(float(y), 4)]


def fit(library: list[tuple[list[float], str]], k: int) -> Projection:
    X = np.asarray([vec for vec, _ in library], dtype=np.float64)
    labels = [label for _, label in library]
    mean = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    var = s ** 2
    proj = Projection(
        mean=mean,
        axes=vt[:2],
        explained=[round(float(v), 3) for v in var[:2] / var.sum()],
        library=[],
        loo_accuracy=_loo_knn(X, labels, k),
    )
    proj.library = [{"x": x, "y": y, "label": label}
                    for (x, y), label in zip((proj.project(v) for v in X), labels)]
    return proj


def _loo_knn(X: np.ndarray, labels: list[str], k: int) -> float:
    """Predict each library window from the other windows' k nearest neighbours."""
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d, np.inf)
    correct = 0
    for i, row in enumerate(d):
        votes: dict[str, int] = {}
        for j in np.argsort(row)[:k]:
            votes[labels[j]] = votes.get(labels[j], 0) + 1
        correct += max(votes, key=votes.get) == labels[i]
    return round(correct / len(labels), 3)
