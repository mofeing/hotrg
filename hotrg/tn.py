import numpy as np
from numpy.typing import ArrayLike
import opt_einsum as oe


def contract_tensors(top: np.ndarray, bottom: np.ndarray, **kwargs) -> ArrayLike:
    m = oe.contract("abci,deif->adbecf", top, bottom, **kwargs)
    return m.reshape((m.shape[0] * m.shape[1], m.shape[2] * m.shape[3], m.shape[4], m.shape[5]))


def update_tensor(U: ArrayLike, M: ArrayLike, **kwargs) -> ArrayLike:
    return oe.contract("il,ijud,jr->lrud", U, M, U, **kwargs)


def error_left(S: ArrayLike, max_bond: int) -> float:
    return np.sum(np.square(np.abs(S[max_bond:, :, :, :])))


def error_right(S: ArrayLike, max_bond: int) -> float:
    return np.sum(np.square(np.abs(S[:, max_bond:, :, :])))


def trace(t: ArrayLike) -> float:
    return oe.contract("iijj->", t)
