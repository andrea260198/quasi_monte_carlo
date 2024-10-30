from scipy.stats import norm, qmc
import numpy as np
import numpy.typing as npt
from typing import Callable
import matplotlib.pyplot as plt
import scipy


def get_low_discr_sample(N: int, M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=N, scramble=False)
    sobol_sequences = sampler.random(M)
    return sobol_sequences.transpose()


def get_pseudo_rnd_sample(N: int, M: int) -> npt.NDArray[np.float64]:
    np.random.seed(0)
    return np.random.uniform(size=((N, M)))


def integrate(U: npt.NDArray[np.float64], dt: float):
    # We want to integrate the following SDE leading to a geometric Brownian motion:
    # dS = mu * S * dt + sigma * S * dW
    mu = 0.00
    sigma = 0.20
    S_0 = 1
    dW = np.sqrt(12 ) * (U - 0.5) * np.sqrt(dt)

    (N, M) = U.shape

    T = N * dt
    S_t = S_0 * np.ones((1, M))
    for k, t in enumerate(np.arange(0, T, dt)):
        S_t += mu * S_t * dt + sigma * S_t * dW[k, :]

    #S_T = S_0 * np.prod(1 + sigma * dW, axis=0)
    S_T = S_t
    return S_T

"""
def integrate(U: npt.NDArray[np.float64], dt: float):
    # We want to integrate the Vasicek SDE:
    # dr = a * (b - r) * dt + sigma * r * dW
    mu = 0.03
    sigma = 0.20
    S_0 = 0.01
    dW = np.sqrt(12 ) * (U - 0.5) * np.sqrt(dt)

    (N, M) = U.shape

    T = N * dt
    S_t = S_0 * np.ones((1, M))
    for k, t in enumerate(np.arange(0, T, dt)):
        S_t += mu * (0.05 - S_t) * dt + sigma * S_t * dW[k, :]
        #S_t += mu * S_t * dt + sigma * S_t * dW[k, :]

    #S_T = S_0 * np.prod(1 + sigma * dW, axis=0)
    S_T = S_t
    return S_T
"""

def calculate_random_walks(get_sample: Callable[[int], npt.NDArray[np.float64]], M: int) -> npt.NDArray[np.float64]:
    T = 1
    dt = 0.01
    N = int(T // dt)
    U = get_sample(N, M)
    S_T = integrate(U, dt)
    return S_T


if __name__ == '__main__':
    M_min = 100
    M_max = 20_000
    MM = range(M_min, M_max, 100)

    low_discr_result = np.array([np.mean(calculate_random_walks(get_low_discr_sample, M)) for M in MM])
    pseudo_rnd_result = np.array([np.mean(calculate_random_walks(get_pseudo_rnd_sample, M)) for M in MM])

    exact_mean = np.mean(calculate_random_walks(get_low_discr_sample, 10*M_max))

    plt.figure(1)
    ax1 = plt.subplot(221)

    # Verify convergence when M goes to inf
    ax1.plot(MM, low_discr_result, "r")
    ax1.plot(MM, pseudo_rnd_result, "k")
    ax1.plot([MM[0], MM[-1]], 2 * [exact_mean], "k--")
    ax1.legend(["QMC", "MC"])
    ax1.set_xlabel("M")
    ax1.set_ylabel("mean")
    ax1.set_title("mean convergence")
    ax1.margins(0.05)

    ax2 = plt.subplot(222)

    # Error convergence is O(1/M) for QMC
    ax2.loglog(MM, np.abs(low_discr_result-exact_mean), "r")
    ax2.loglog(MM, 1/np.array(MM), "r:")
    ax2.loglog(MM, np.abs(pseudo_rnd_result-exact_mean), "k")
    ax2.loglog(MM, 1/np.array(MM)**00.5, "k:")
    ax2.legend(["QMC", "$O(1/M)$", "MC", "$O(1/\sqrt{M})$"])
    ax2.set_xlabel("M")
    ax2.set_ylabel("error")
    ax2.set_title("error convergence")
    ax2.margins(0.05)

    ax3 = plt.subplot(223)

    # Verify that QMC and MC yield the same distribution with Kolmogorov-Smirnov test
    pseudo_random_sample = calculate_random_walks(get_pseudo_rnd_sample, M_max)
    low_discr_sample = calculate_random_walks(get_low_discr_sample, M_max)

    print(scipy.stats.ks_2samp(low_discr_sample, pseudo_random_sample))

    # Compare the cdf of both distribution
    ax3.ecdf(pseudo_random_sample[0, :], linestyle="-", color="k")
    ax3.ecdf(low_discr_sample[0, :], linestyle="--", color="r")
    ax3.legend(["MC", "QMC"])
    ax3.set_title("Empirical CDF")
    ax3.margins(0.05)

    plt.show()




