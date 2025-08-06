import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import distributions


def test_continuous_distributions() -> None:
    uniform = distributions.Uniform(0.0, 1.0)
    assert math.isclose(uniform.pdf(0.5), 1.0)
    assert math.isclose(uniform.cdf(0.5), 0.5)
    assert math.isclose(uniform.pdf(-1.0), 0.0)

    gaussian = distributions.Gaussian(0.0, 1.0)
    assert gaussian.pdf(0.0) > 0
    assert math.isclose(gaussian.cdf(0.0), 0.5)

    tg = distributions.TruncatedGaussian.mleEstimate([0.0, 1.0, 0.5])
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

    bin_est = distributions.Binary.mleEstimate([True, False, True])
    assert isinstance(bin_est, distributions.Binary)
