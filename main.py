import numpy.random
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import numpy.typing as npt
# TODO: use 'pipe' package


def get_low_discr_sample(M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=1, scramble=False)
    sobol_sample = sampler.random(M)
    normal_sample = norm.ppf(sobol_sample)
    normal_sample = normal_sample[~np.isinf(normal_sample)]
    return normal_sample


def get_pseudo_rnd_sample(M: int) -> npt.NDArray[np.float64]:
    normal_sample = numpy.random.normal(size=M)
    return normal_sample


def calc_cutoff(sample: npt.NDArray, threshold: float) -> float:
    p = sum(sample > threshold) / len(sample)
    return p


if __name__ == '__main__':
    threshold = 2.0
    MM = np.arange(10, 10_000, 100, dtype=int)

    plt.plot(MM, list(map(lambda M: calc_cutoff(get_pseudo_rnd_sample(M), threshold), MM)), "k")
    plt.plot(MM, list(map(lambda M: calc_cutoff(get_low_discr_sample(M), threshold), MM)), "r")
    plt.plot(MM, np.ones(MM.shape) * (1 - norm.cdf(threshold)), "k-")
    plt.legend(["Monte Carlo", "Quasi-Monte Carlo"])
    plt.show()
