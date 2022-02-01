from multiprocessing import Pool
import numpy as np
from pathlib import Path

from rcoeffs_cy import R_tensor


def compute_R(N: int, tau: int) -> None:
    """computation protocol. Followed by each worker of the multiprocessing pool
    (one worker per value of tau).

    Parameters
    ----------
    N : int
        max number of particles for each input channel
    tau : int
        transmission coefficient
    """
    outname = f"R_N{N}_T{tau}.npy"
    p = Path(".") / "data" / outname

    R = R_tensor(N, tau / 100.0)
    with p.open(mode="wb") as file:
        np.save(file, R)


"""
PARAMETERS
----------
"""
# single channel, max number of particles
N = 300
# beamsplitter's transmission coefficients to compute (in percent without decimals)
tau_array = np.linspace(10, 50, 20, dtype=int)


"""
ACTUAL COMPUTATION (WITH MULTIPROCESSING)
-----------------------------------------
"""
with Pool(processes=len(tau_array)) as pool:
    for tau in tau_array:
        pool.apply_async(
            compute_R,
            args=(N, tau),
        )
    pool.close()
    pool.join()
