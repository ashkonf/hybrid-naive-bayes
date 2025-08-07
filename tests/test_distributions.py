import sys
from pathlib import Path

# Ensure src directory is on sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import pytest
import distributions
import math


def test_exponential_mle_estimate_errors():
    # No data points should raise an estimation error
    with pytest.raises(distributions.EstimationError):
        distributions.Exponential.mleEstimate([])

    # Negative data points are not allowed for exponential distribution
    with pytest.raises(distributions.EstimationError):
        distributions.Exponential.mleEstimate([-1.0, 2.0])

    # Mean of zero leads to a parametrization error
    with pytest.raises(distributions.ParametrizationError):
        distributions.Exponential.mleEstimate([0.0, 0.0])


def test_gaussian_mle_estimate():
    dist = distributions.Gaussian.mleEstimate([1.0, 2.0, 3.0, 4.0])
    assert dist.mean == pytest.approx(2.5)
    assert dist.stdev == pytest.approx(math.sqrt(5.0 / 3.0))


def test_poisson_mle_estimate():
    dist = distributions.Poisson.mleEstimate([2, 3, 4])
    assert dist.lambdaa == pytest.approx(3.0)
    expected = (3.0 ** 3) / math.factorial(3) * math.exp(-3.0)
    assert dist.probability(3) == pytest.approx(expected)
