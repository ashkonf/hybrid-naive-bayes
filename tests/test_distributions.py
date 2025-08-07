import math

import pytest

from src import distributions


def test_exponential_mle_estimate_errors():
    # No data points should raise an estimation error
    with pytest.raises(distributions.EstimationError):
        distributions.Exponential.mle_estimate([])

    # Negative data points are not allowed for exponential distribution
    with pytest.raises(distributions.EstimationError):
        distributions.Exponential.mle_estimate([-1.0, 2.0])

    # Mean of zero leads to a parametrization error
    with pytest.raises(distributions.ParametrizationError):
        distributions.Exponential.mle_estimate([0.0, 0.0])


def test_gaussian_mle_estimate():
    dist = distributions.Gaussian.mle_estimate([1.0, 2.0, 3.0, 4.0])
    assert dist.mean == pytest.approx(2.5)
    assert dist.stdev == pytest.approx(math.sqrt(5.0 / 3.0))


def test_poisson_mle_estimate():
    dist = distributions.Poisson.mle_estimate([2, 3, 4])
    assert dist.lambdaa == pytest.approx(3.0)
    expected = (3.0**3) / math.factorial(3) * math.exp(-3.0)
    assert dist.probability(3) == pytest.approx(expected)


def test_continuous_distributions() -> None:
    uniform = distributions.Uniform(0.0, 1.0)
    assert math.isclose(uniform.pdf(0.5), 1.0)
    assert math.isclose(uniform.cdf(0.5), 0.5)
    assert math.isclose(uniform.pdf(-1.0), 0.0)

    gaussian = distributions.Gaussian(0.0, 1.0)
    assert gaussian.pdf(0.0) > 0
    assert math.isclose(gaussian.cdf(0.0), 0.5)

    tg = distributions.TruncatedGaussian.mle_estimate([0.0, 1.0, 0.5])
    assert tg.pdf(0.5) > 0

    ln = distributions.LogNormal(0.0, 1.0)
    assert ln.pdf(1.0) > 0
    assert ln.cdf(1.0) > 0

    ex = distributions.Exponential(1.0)
    assert ex.pdf(1.0) > 0
    assert ex.cdf(1.0) > 0

    kde = distributions.KernelDensityEstimate([0.0, 1.0])
    assert kde.pdf(0.5) > 0


def test_discrete_distributions() -> None:
    du = distributions.DiscreteUniform(0, 2)
    assert du.probability(1) > 0
    assert du.probability(3) == 0.0

    pois = distributions.Poisson(1.0)
    assert pois.probability(0) > 0

    m = distributions.Multinomial({"a": 2, "b": 1})
    assert m.probability("a") > m.probability("c")

    bin_est = distributions.Binary.mle_estimate([True, False, True])
    assert isinstance(bin_est, distributions.Binary)
