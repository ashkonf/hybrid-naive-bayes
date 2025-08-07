import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import distributions
import nb


def simple_featurizer(x: float) -> List[nb.Feature]:
    return [nb.Feature("pos", distributions.Binary, x > 0)]


def test_classifier_basic() -> None:
    clf = nb.NaiveBayesClassifier(simple_featurizer)
    objects = [1, -1, 2, -2]
    labels = ["pos", "neg", "pos", "neg"]
    clf.train(objects, labels)
    assert clf.classify(3) == "pos"
    probs = clf.probabilities(3)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    cost = {"pos": {"pos": 0.0, "neg": 1.0}, "neg": {"pos": 1.0, "neg": 0.0}}
    assert clf.classify(3, cost) in {"pos", "neg"}
    assert clf.accuracy(objects, labels) == 1.0


def test_zero_probability_branch() -> None:
    clf = nb.NaiveBayesClassifier(
        lambda x: [nb.Feature("val", distributions.Uniform, float(x))]
    )
    clf.train([0.0, 1.0], ["a", "b"])
    assert clf.probability(10.0, "a") == 0.0


def test_exception_branch() -> None:
    clf = nb.NaiveBayesClassifier(
        lambda x: [nb.Feature("exp", distributions.Exponential, x)]
    )
    clf.train([-1.0, -2.0], ["a", "b"])
