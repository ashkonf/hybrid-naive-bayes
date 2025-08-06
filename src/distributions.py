from __future__ import annotations

import math
import collections
from typing import Any, Dict, Iterable, List

## Distribution ########################################################################################


class Distribution(object):
    def __init__(self) -> None:
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "Distribution":
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def momEstimate(cls, points: Iterable[float]) -> "Distribution":
        raise NotImplementedError("Subclasses should override.")


## ContinuousDistribution ##############################################################################


class ContinuousDistribution(Distribution):
    def pdf(self, value: float) -> float:
        raise NotImplementedError("Subclasses should override.")

    def cdf(self, value: float) -> float:
        raise NotImplementedError("Subclasses should override.")


## Uniform #############################################################################################


class Uniform(ContinuousDistribution):
    def __init__(self, alpha: float, beta: float) -> None:
        if alpha == beta:
            raise ParametrizationError("alpha and beta cannot be equivalent")
        self.alpha: float = alpha
        self.beta: float = beta
        self.range: float = beta - alpha

    def pdf(self, value: float) -> float:
        if value < self.alpha or value > self.beta:
            return 0.0
        else:
            return 1.0 / self.range

    def cdf(self, value: float) -> float:
        if value < self.alpha:
            return 0.0
        elif value >= self.beta:
            return 1.0
        else:
            return (value - self.alpha) / self.range

    def __str__(self) -> str:
        return "Continuous Uniform distribution: alpha = %s, beta = %s" % (
            self.alpha,
            self.beta,
        )

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "Uniform":
        return cls(min(points), max(points))


## Gaussian ############################################################################################


class Gaussian(ContinuousDistribution):
    def __init__(self, mean: float, stdev: float) -> None:
        self.mean: float = mean
        self.stdev: float = stdev
        if stdev == 0.0:
            raise ParametrizationError("standard deviation must be non-zero")
        if stdev < 0.0:
            raise ParametrizationError("standard deviation must be positive")
        self.variance: float = math.pow(stdev, 2.0)

    def pdf(self, value: float) -> float:
        numerator = math.exp(
            -math.pow(float(value - self.mean) / self.stdev, 2.0) / 2.0
        )
        denominator = math.sqrt(2 * math.pi * self.variance)
        return numerator / denominator

    def cdf(self, value: float) -> float:
        return 0.5 * (
            1.0 + math.erf((value - self.mean) / math.sqrt(2.0 * self.variance))
        )

    def __str__(self) -> str:
        return (
            "Continuous Gaussian (Normal) distribution: mean = %s, standard deviation = %s"
            % (self.mean, self.stdev)
        )

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "Gaussian":
        points_list = list(points)
        numPoints = float(len(points_list))
        if numPoints <= 1:
            raise EstimationError("must provide at least 2 training points")

        mean = sum(points_list) / numPoints

        variance = 0.0
        for point in points_list:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= numPoints - 1.0
        stdev = math.sqrt(variance)

        return cls(mean, stdev)


## TruncatedGaussian ##################################################################################


class TruncatedGaussian(ContinuousDistribution):
    def __init__(self, mean: float, stdev: float, alpha: float, beta: float) -> None:
        self.mean: float = mean
        if stdev == 0.0:
            raise ParametrizationError("standard deviation must be non-zero")
        if stdev < 0.0:
            raise ParametrizationError("standard deviation must be positive")
        self.stdev: float = stdev
        self.variance: float = math.pow(stdev, 2.0)
        self.alpha: float = alpha
        self.beta: float = beta

    def pdf(self, value: float) -> float:
        if self.alpha == self.beta or self.__phi(self.alpha) == self.__phi(self.beta):
            if value == self.alpha:
                return 1.0
            else:
                return 0.0
        else:
            numerator = math.exp(-math.pow((value - self.mean) / self.stdev, 2.0) / 2.0)
            denominator = (
                math.sqrt(2 * math.pi)
                * self.stdev
                * (self.__phi(self.beta) - self.__phi(self.alpha))
            )
            return numerator / denominator

    def cdf(self, value: float) -> float:
        if value < self.alpha or value > self.beta:
            return 0.0
        else:
            numerator = self.__phi((value - self.mean) / self.stdev) - self.__phi(
                self.alpha
            )
            denominator = self.__phi(self.beta) - self.__phi(self.alpha)
            return numerator / denominator

    def __str__(self) -> str:
        return (
            "Continuous Truncated Gaussian (Normal) distribution: mean = %s, standard deviation = %s, lower bound = %s, upper bound = %s"
            % (self.mean, self.stdev, self.alpha, self.beta)
        )

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "TruncatedGaussian":
        points_list = list(points)
        numPoints = float(len(points_list))

        if numPoints <= 1:
            raise EstimationError("must provide at least 2 training points")

        mean = sum(points_list) / numPoints

        variance = 0.0
        for point in points_list:
            variance += math.pow(float(point) - mean, 2.0)
        variance /= numPoints - 1.0
        stdev = math.sqrt(variance)

        return cls(mean, stdev, min(points_list), max(points_list))

    def __phi(self, value: float) -> float:
        return 0.5 * (
            1.0 + math.erf((value - self.mean) / (self.stdev * math.sqrt(2.0)))
        )


## LogNormal ###########################################################################################


class LogNormal(ContinuousDistribution):
    def __init__(self, mean: float, stdev: float) -> None:
        self.mean: float = mean
        self.stdev: float = stdev
        if stdev == 0.0:
            raise ParametrizationError("standard deviation must be non-zero")
        if stdev < 0.0:
            raise ParametrizationError("standard deviation must be positive")
        self.variance: float = math.pow(stdev, 2.0)

    def pdf(self, value: float) -> float:
        if value <= 0:
            return 0.0
        else:
            return math.exp(
                -math.pow(float(math.log(value) - self.mean) / self.stdev, 2.0) / 2.0
            ) / (value * math.sqrt(2 * math.pi * self.variance))

    def cdf(self, value: float) -> float:
        return 0.5 + 0.5 * math.erf(
            (math.log(value) - self.mean) / math.sqrt(2.0 * self.variance)
        )

    def __str__(self) -> str:
        return (
            "Continuous Log Normal distribution: mean = %s, standard deviation = %s"
            % (self.mean, self.stdev)
        )

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "LogNormal":
        points_list = list(points)
        numPoints = float(len(points_list))

        if numPoints <= 1:
            raise EstimationError("must provide at least 2 training points")

        mean = sum(math.log(float(point)) for point in points_list) / numPoints

        variance = 0.0
        for point in points_list:
            variance += math.pow(math.log(float(point)) - mean, 2.0)
        variance /= numPoints - 1.0
        stdev = math.sqrt(variance)

        return cls(mean, stdev)


## Exponential ########################################################################################


class Exponential(ContinuousDistribution):
    def __init__(self, lambdaa: float) -> None:
        # 2 "a"s to avoid confusion with "lambda" keyword
        self.lambdaa: float = lambdaa

    def mean(self) -> float:
        return 1.0 / self.lambdaa

    def variance(self) -> float:
        return 1.0 / pow(self.lambdaa, 2.0)

    def pdf(self, value: float) -> float:
        return self.lambdaa * math.exp(-self.lambdaa * value)

    def cdf(self, value: float) -> float:
        return 1.0 - math.exp(-self.lambdaa * value)

    def __str__(self) -> str:
        return "Continuous Exponential distribution: lamda = %s" % self.lambdaa

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "Exponential":
        points_list = list(points)
        if len(points_list) == 0:
            raise EstimationError("Must provide at least one point.")
        if min(points_list) < 0.0:
            raise EstimationError(
                "Exponential distribution only supports non-negative values."
            )

        mean = float(sum(points_list)) / float(len(points_list))

        if mean == 0.0:
            raise ParametrizationError("Mean of points must be positive.")

        return cls(1.0 / mean)


## KernelDensityEstimate ##############################################################################


class KernelDensityEstimate(ContinuousDistribution):
    """
    See this paper for more information about using Gaussian
    Kernal Density Estimation with the Naive Bayes Classifier:
    http://www.cs.iastate.edu/~honavar/bayes-continuous.pdf
    """

    def __init__(self, observedPoints: Iterable[float]) -> None:
        self.observedPoints: List[float] = list(observedPoints)
        self.numObservedPoints: float = float(len(self.observedPoints))
        self.stdev: float = 1.0 / math.sqrt(self.numObservedPoints)

    def pdf(self, value: float) -> float:
        pdfValues = [
            self.__normalPdf(point, self.stdev, value) for point in self.observedPoints
        ]
        return sum(pdfValues) / self.numObservedPoints

    def __normalPdf(self, mean: float, stdev: float, value: float) -> float:
        numerator = math.exp(-math.pow(float(value - mean) / stdev, 2.0) / 2.0)
        denominator = math.sqrt(2 * math.pi * math.pow(stdev, 2.0))
        return numerator / denominator

    def cdf(self, value: float) -> float:  # pragma: no cover - not implemented
        raise NotImplementedError("Not implemented")

    def __str__(self) -> str:
        return "Continuous Gaussian Kernel Density Estimate distribution"

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "KernelDensityEstimate":
        return cls(points)


## DiscreteDistribution ###############################################################################


class DiscreteDistribution(Distribution):
    def probability(self, value: Any) -> float:
        raise NotImplementedError("Subclasses should override.")


## Uniform ############################################################################################


class DiscreteUniform(DiscreteDistribution):
    def __init__(self, alpha: float, beta: float) -> None:
        if alpha == beta:
            raise Exception("alpha and beta cannot be equivalent")
        self.alpha: float = float(alpha)
        self.beta: float = float(beta)
        self.prob: float = 1.0 / (self.beta - self.alpha)

    def probability(self, value: float) -> float:
        if value < self.alpha or value > self.beta:
            return 0.0
        else:
            return self.prob

    def __str__(self) -> str:
        return "Discrete Uniform distribution: alpha = %s, beta = %s" % (
            self.alpha,
            self.beta,
        )

    @classmethod
    def mleEstimate(cls, points: Iterable[float]) -> "DiscreteUniform":
        return cls(min(points), max(points))


## Poissoin ###########################################################################################


class Poisson(DiscreteDistribution):
    def __init__(self, lambdaa: float) -> None:
        # 2 "a"s to avoid confusion with "lambda" keyword
        self.lambdaa: float = lambdaa

    def probability(self, value: int) -> float:
        try:
            first = float(math.pow(self.lambdaa, value)) / float(math.factorial(value))
            second = float(math.exp(-float(self.lambdaa)))
            return first * second
        except OverflowError:
            # this is an approximation to the probability of very unlikely events
            return 0.0

    def __str__(self) -> str:
        return "Discrete Poisson distribution: lamda = %s" % self.lambdaa

    @classmethod
    def mleEstimate(cls, points: Iterable[int]) -> "Poisson":
        points_list = list(points)
        mean = float(sum(points_list)) / float(len(points_list))
        return cls(mean)


## Multinomial #######################################################################################


class Multinomial(DiscreteDistribution):
    def __init__(
        self, categoryCounts: Dict[Any, int], smoothingFactor: float = 1.0
    ) -> None:
        self.categoryCounts: Dict[Any, int] = categoryCounts
        self.numPoints: float = float(sum(categoryCounts.values()))
        self.numCategories: float = float(len(categoryCounts))
        self.smoothingFactor: float = float(smoothingFactor)

    def probability(self, value: Any) -> float:
        if value not in self.categoryCounts:
            return 0.0
        numerator = float(self.categoryCounts[value]) + self.smoothingFactor
        denominator = self.numPoints + self.numCategories * self.smoothingFactor
        return numerator / denominator

    def __str__(self) -> str:
        return "Discrete Multinomial distribution: buckets = %s" % self.categoryCounts

    @classmethod
    def mleEstimate(cls, points: Iterable[Any]) -> "Multinomial":
        categoryCounts: Dict[Any, int] = collections.Counter()
        for point in points:
            categoryCounts[point] += 1
        return cls(categoryCounts)


## Binary ############################################################################################


class Binary(Multinomial):
    def __init__(
        self, trueCount: int, falseCount: int, smoothingFactor: float = 1.0
    ) -> None:
        categoryCounts = {True: trueCount, False: falseCount}
        super().__init__(categoryCounts, smoothingFactor)

    def __str__(self) -> str:
        return "Discrete Binary distribution: true count = %s, false count = %s" % (
            self.categoryCounts[True],
            self.categoryCounts[False],
        )

    @classmethod
    def mleEstimate(
        cls, points: Iterable[bool], smoothingFactor: float = 1.0
    ) -> "Binary":
        points_list = list(points)
        trueCount = 0
        for point in points_list:
            if point:
                trueCount += 1
        falseCount = len(points_list) - trueCount
        return cls(trueCount, falseCount, smoothingFactor)


## Errors ############################################################################################


class EstimationError(Exception):
    def __init__(self, value: str) -> None:
        self.value: str = value

    def __str__(self) -> str:
        return repr(self.value)


class ParametrizationError(Exception):
    def __init__(self, value: str) -> None:
        self.value: str = value

    def __str__(self) -> str:
        return repr(self.value)
