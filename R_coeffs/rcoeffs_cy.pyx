cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

cimport scipy.special.cython_special as csc

np.import_array()

cdef double R_value_cy(int n1, int n2, int N1, float tau):
    """Returns the value of R_{N1,N2}^{n1,n2} assuming the following conditions to be verified:
        - n1 + n2 = N1 + N2 -> fixes the value of N2
        - n1, n2 <= N1 -> other cases can be computed thanks to symmetries relations

    for more info:
    [Campos, Saleh, and Teich, Physical Review A 40, no. 3 (1 August 1989): 1371–84, https://doi.org/10/bjcf48.]


    Parameters
    ----------
    n1 : int
        first input channel
    n2 : int
        second input channel
    N1 : int
        first output channel

    Returns
    -------
    float
        Output photon-number probability amplitude
    """

    cdef int N2 = n1 + n2 - N1

    # computation of the combinatory prefactor
    # computation in log for precision of very large numbers
    cdef double log_combi_prefector = 0.5 * (
            (N1 - n2) * np.log(tau)
            + (N1 - n1) * np.log(1 - tau)
            + csc.loggamma(<double>N1 + 1.0)
            + csc.loggamma(<double>N2 + 1.0)
            - csc.loggamma(<double>n1 + 1.0)
            - csc.loggamma(<double>n2 + 1.0)
        )

    return np.exp(log_combi_prefector) * csc.eval_jacobi(N2, N1 - n1, N1 - n2, 2 * tau - 1.0)

def R_tensor(int N_atoms, float tau):
    """Returns the values of the tensor R_{N1,N2}^{n1,n2}(tau). It is comupted in a "minimal" way:
        - max(n1, n2) <= N1 <= n1 + n2 -> tuple of values for a fixed couple (n1, n2)
        - N2 = n1 + n2 - N1
        - tau is the transmission coefficient of the beamsplitter.

    Any other indices (n1, n2, N2, N2) will lead to a value that can be retrieved with the
    value computed by this function, and symetry relations. For more info:
    [Campos, Saleh, and Teich, Physical Review A 40, no. 3 (1 August 1989): 1371–84, https://doi.org/10/bjcf48.]


    Parameters
    ----------
    N_atoms : int
        maximum number of particles per input channel. The total maximum number of particles is therefore 2*N_atoms
    tau : float
        beamsplitter's transmission coefficient

    Returns
    -------
    np.ndarray[tuple[double]]
        2D array of output photon-number probability amplitudes. Each cell refers to the
        (n1, n2) input configuration, and contains a tuple: the "minimal" possible output probabilities.
    """
    
    cdef np.ndarray[object, ndim=2] R_tensor = np.empty((N_atoms+1, N_atoms+1), dtype=object)
    cdef int n1, n2, N1, N2

    for n1 in range(N_atoms + 1):
        for n2 in range(N_atoms + 1):
                R_tensor[n1, n2] = tuple(R_value_cy(n1, n2, N1, tau) for N1 in range(max(n1, n2), n1 + n2 + 1))

    return R_tensor