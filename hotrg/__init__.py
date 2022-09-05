from .hosvd import hosvd, hosvd_sides
from . import ising
from .tn import *


def iterate(H: ArrayLike, T: ArrayLike, max_bond: int, **kwargs) -> tuple[ArrayLike, ArrayLike]:
    # contract with environment
    Mh = contract_tensors(H, T, **kwargs)
    Mt = contract_tensors(T, T, **kwargs)

    # select projector
    S, Ul, Ur = hosvd_sides(Mt)
    epsl = error_left(S, max_bond)
    epsr = error_right(S, max_bond)
    U = Ul if epsl < epsr else Ur

    # truncate projector
    if U.shape[1] > max_bond:
        U = U[:, 0:max_bond]

    # normalize tensors
    U /= np.sqrt(np.max(S))

    # update tensors
    H = update_tensor(U, Mh, **kwargs)
    T = update_tensor(U, Mt, **kwargs)

    return H, T
