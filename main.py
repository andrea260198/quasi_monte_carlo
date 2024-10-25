import numpy.random
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from typing import Callable
import numpy.typing as npt


def get_low_discr_sample(M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=1, scramble=False)
    sobol_sample = sampler.random(M)
    normal_sample = norm.ppf(sobol_sample)
    normal_sample = normal_sample[~np.isinf(normal_sample)]
    return normal_sample


def get_pseudo_rnd_sample(M: int) -> npt.NDArray[np.float64]:
    normal_sample = numpy.random.normal(size=M)
    return normal_sample


def calc_cutoff(get_sample: Callable[[int], npt.NDArray[np.float64]], M: int, thresold: float) -> float:
    sample = get_sample(M)
    p = sum(sample > thresold) / M
    return p


if __name__ == '__main__':
    threshold = 2.0
    MM = np.arange(10, 10_000, 100, dtype=int)

    plt.plot(MM, list(map(lambda M: calc_cutoff(get_pseudo_rnd_sample, M, threshold), MM)), "k")
    plt.plot(MM, list(map(lambda M: calc_cutoff(get_low_discr_sample, M, threshold), MM)), "r")
    plt.plot(MM, np.ones(MM.shape) * (1 - norm.cdf(threshold)), "k-")
    plt.legend(["Monte Carlo", "Quasi-Monte Carlo"])
    plt.show()
