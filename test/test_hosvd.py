import pytest
import numpy as np
from ten import hosvd


def test_hosvd_cx():
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.float32)
    X = np.einsum("ia,ja,ka->ijk", H, H, H)
    S, U = hosvd(X)

    assert np.allclose(S, 1)
    assert all(np.allclose(u, H) for u in U)
