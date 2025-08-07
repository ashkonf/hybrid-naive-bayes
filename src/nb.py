from __future__ import annotations

import sys
import collections
import math
import operator
import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import distributions


## Feature ##############################################################################


class Feature(object):
    def __init__(
        self, name: str, distribution: Type[distributions.Distribution], value: Any
    ) -> None:
        self.name: str = name
        self.distribution: Type[distributions.Distribution] = distribution
        self.value: Any = value

    def __repr__(self) -> str:
        return self.name + " => " + str(self.value)

    def hashable(self) -> tuple[str, Any]:
        return (self.name, self.value)

    @classmethod
    def binary(cls, name: str) -> "Feature":
        return cls(name, distributions.Binary, True)


## ExtractedFeature #####################################################################


class ExtractedFeature(Feature):
    def __init__(self, object: Any) -> None:
        name = self.__class__.__name__
        distribution = self.distribution()
        value = self.extract(object)
        super(ExtractedFeature, self).__init__(name, distribution, value)

    def extract(self, object: Any) -> Any:
        # returns feature value corresponding to |object|
        raise NotImplementedError("Subclasses should override.")

    @classmethod
    def distribution(cls) -> Type[distributions.Distribution]:
        # returns the distribution this feature conforms to
        raise NotImplementedError("Subclasses should override.")


## NaiveBayesClassifier #################################################################


class NaiveBayesClassifier(object):
    def __init__(
        self, featurizer: Optional[Callable[[Any], List[Feature]]] = None
    ) -> None:
        self.featurizer: Optional[Callable[[Any], List[Feature]]] = featurizer
        self.priors: Optional[Dict[Any, float]] = None
        self.distributions: Optional[
            Dict[Any, Dict[str, distributions.Distribution]]
        ] = None

    def featurize(self, object: Any) -> List[Feature]:
        if self.featurizer is None:
            raise Exception(
                "If no featurizer is provided upon initialization, self.featurize must be overridden."
            )
        return self.featurizer(object)

    def train(self, objects: Sequence[Any], labels: Sequence[Any]) -> None:
        featureValues: Dict[Any, Dict[str, List[Any]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        distributionTypes: Dict[str, Type[distributions.Distribution]] = {}

        labelCounts: collections.Counter[Any] = collections.Counter()

        for index, object in enumerate(objects):
            label = labels[index]
            labelCounts[label] += 1
            for feature in self.featurize(object):
                featureValues[label][feature.name].append(feature.value)
                distributionTypes[feature.name] = feature.distribution

        self.distributions = collections.defaultdict(dict)
        for label in featureValues:
            for featureName in featureValues[label]:
                try:
                    values = featureValues[label][featureName]
                    if issubclass(distributionTypes[featureName], distributions.Binary):
                        trueCount = len([value for value in values if value])
                        # the absence of binary feature is treated as it having been present with a False value
                        falseCount = labelCounts[label] - trueCount
                        distribution: distributions.Distribution = distributions.Binary(
                            trueCount, falseCount
                        )
                    else:
                        distribution = distributionTypes[featureName].mleEstimate(
                            values
                        )
                except (
                    distributions.EstimationError,
                    distributions.ParametrizationError,
                ):
                    if issubclass(distributionTypes[featureName], distributions.Binary):
                        distribution = distributions.Binary(0, labelCounts[label])
                    elif issubclass(
                        distributionTypes[featureName],
                        distributions.DiscreteDistribution,
                    ):
                        distribution = distributions.DiscreteUniform(
                            -sys.maxsize, sys.maxsize
                        )
                    else:
                        distribution = distributions.Uniform(
                            -sys.float_info.max, sys.float_info.max
                        )
                self.distributions[label][featureName] = distribution

        self.priors = {}
        for label in labelCounts:
            # A label count can never be 0 because we only generate
            # a label count upon observing the first data point that
            # belongs to it. As a result, we don't worrying about
            # the argument to log being 0 here.
            self.priors[label] = math.log(labelCounts[label])

    def __labelWeights(self, object: Any) -> Dict[Any, float]:
        if self.priors is None or self.distributions is None:
            raise Exception("Classifier has not been trained")
        features = self.featurize(object)

        labelWeights: Dict[Any, float] = copy.deepcopy(self.priors)

        for feature in features:
            for label in self.priors:
                if feature.name in self.distributions[label]:
                    distribution = self.distributions[label][feature.name]
                    if isinstance(distribution, distributions.DiscreteDistribution):
                        probability = distribution.probability(feature.value)
                    elif isinstance(distribution, distributions.ContinuousDistribution):
                        probability = distribution.pdf(feature.value)
                    else:
                        raise Exception(
                            "Naive Bayes Training Error: Invalid probability distribution"
                        )
                else:
                    if issubclass(feature.distribution, distributions.Binary):
                        distribution = distributions.Binary(
                            0, int(math.exp(self.priors[label]))
                        )
                        probability = distribution.probability(feature.value)
                    else:
                        raise Exception(
                            "Naive Bayes Training Error: Non-binary features must be present for all training examples"
                        )

                if probability == 0.0:
                    labelWeights[label] = float("-inf")
                else:
                    labelWeights[label] += math.log(probability)

        return labelWeights

    def probability(self, object: Any, label: Any) -> float:
        labelWeights = self.__labelWeights(object)

        numerator = labelWeights[label]
        if numerator == float("-inf"):
            return 0.0

        denominator = 0.0
        minWeight = min(labelWeights.items(), key=operator.itemgetter(1))[1]
        for label in labelWeights:
            weight = labelWeights[label]
            if minWeight < 0.0:
                weight /= -minWeight
            denominator += math.exp(weight)
        denominator = math.log(denominator)

        return math.exp(numerator - denominator)

    def probabilities(self, object: Any) -> Dict[Any, float]:
        labelProbabilities: Dict[Any, float] = {}
        if self.priors is None:
            raise Exception("Classifier has not been trained")
        for label in self.priors:
            labelProbabilities[label] = self.probability(object, label)
        return labelProbabilities

    def classify(
        self, object: Any, costMatrix: Optional[Dict[Any, Dict[Any, float]]] = None
    ) -> Any:
        if costMatrix is None:
            labelWeights = self.__labelWeights(object)
            return max(labelWeights.items(), key=operator.itemgetter(1))[0]

        else:
            labelCosts: Dict[Any, float] = {}
            labelProbabilities = self.probabilities(object)
            for predictedLabel in labelProbabilities:
                if predictedLabel not in costMatrix:
                    raise Exception(
                        "Naive Bayes Prediction Error: Cost matrix does not include all labels."
                    )
                cost = 0.0
                for actualLabel in labelProbabilities:
                    if actualLabel not in costMatrix:
                        raise Exception(
                            "Naive Bayes Prediction Error: Cost matrix does not include all labels."
                        )
                    cost += (
                        labelProbabilities[predictedLabel]
                        * costMatrix[predictedLabel][actualLabel]
                    )
                labelCosts[predictedLabel] = cost
            return min(labelCosts.items(), key=operator.itemgetter(1))[0]

    def accuracy(self, objects: Sequence[Any], goldLabels: Sequence[Any]) -> float:
        if len(objects) == 0 or len(objects) != len(goldLabels):
            raise ValueError("Malformed data")

        numCorrect = 0
        for index, object in enumerate(objects):
            if self.classify(object) == goldLabels[index]:
                numCorrect += 1
        return float(numCorrect) / float(len(objects))
