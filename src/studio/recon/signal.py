"""Smoothing shared by grasp detection, the interaction graph and the
object trajectory. In one place because detection thresholds are tuned
against these exact window widths."""

import numpy as np


def smooth(x: np.ndarray, k: int = 7) -> np.ndarray:
    """Centred box filter with edge padding, so the output keeps length."""
    kern = np.ones(k) / k
    return np.convolve(np.pad(x, k // 2, mode="edge"), kern, mode="valid")


def smooth3(p: np.ndarray, k: int = 7) -> np.ndarray:
    """`smooth` applied per column of an (T, 3) track."""
    return np.stack([smooth(p[:, i], k) for i in range(3)], axis=1)
