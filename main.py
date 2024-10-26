import numpy.random
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import numpy.typing as npt
from pipe import select


def get_low_discr_sample(M: int) -> npt.NDArray[np.float64]:
    sampler = qmc.Sobol(d=1, scramble=False)
    sobol_sample = sampler.random(M)
    normal_sample = norm.ppf(sobol_sample)
    normal_sample = normal_sample[~np.isinf(normal_sample)]
    return normal_sample


def get_pseudo_rnd_sample(M: int) -> npt.NDArray[np.float64]:
    normal_sample = numpy.random.normal(size=M)
    return normal_sample


def calc_cutoff(sample: npt.NDArray[np.float64], threshold: float) -> float:
    p = sum(sample > threshold) / len(sample)
    return p


if __name__ == '__main__':
    threshold = 2.0
    MM = range(10, 10_000, 100)

    plt.plot(MM, list(MM | select(get_pseudo_rnd_sample) | select(lambda sample: calc_cutoff(sample, threshold))), "k")
    plt.plot(MM, list(MM | select(get_low_discr_sample) | select(lambda sample: calc_cutoff(sample, threshold))), "r")
    plt.plot([MM[0], MM[-1]], 2 * [(1 - norm.cdf(threshold))], "k--")
    plt.legend(["Monte Carlo", "Quasi-Monte Carlo"])
    plt.show()
