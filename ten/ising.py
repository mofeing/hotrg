import numpy as np


def W(temp: float, dtype=np.float32) -> np.ndarray:
    return np.array(
        [
            [np.sqrt(np.cosh(1 / temp)), np.sqrt(np.sinh(1 / temp))],
            [np.sqrt(np.cosh(1 / temp)), -np.sqrt(np.sinh(1 / temp))],
        ],
        dtype=dtype,
    )


def partition_tensor(temp: float, dtype=np.float32) -> np.ndarray:
    w = W(temp, dtype)
    return np.einsum("al,ar,au,ad->lrud", w, w, w, w)


def magnetization(temp: float, dtype=np.float32) -> np.ndarray:
    """Magnetization tensor."""
    w = W(temp, dtype)
    m = np.array([1, -1])
    return np.einsum("al,ar,au,ad,a->lrud", w, w, w, w, m)
