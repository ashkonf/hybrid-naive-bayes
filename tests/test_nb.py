import sys
from pathlib import Path
from typing import List

# Ensure src directory is on sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import nb
import distributions
import pytest


def simple_featurizer(value):
    """Return a single binary feature for the given value."""
    return [nb.Feature('flag', distributions.Binary, value)]


def test_classify_and_accuracy():
    classifier = nb.NaiveBayesClassifier(simple_featurizer)
    objects = [True, True, False, False]
    labels = ['spam', 'spam', 'ham', 'ham']
    classifier.train(objects, labels)
    assert classifier.classify(True) == 'spam'
    assert classifier.classify(False) == 'ham'
    assert classifier.accuracy(objects, labels) == 1.0


def test_probability_single_label():
    classifier = nb.NaiveBayesClassifier(simple_featurizer)
    objects = [True, False]
    labels = ['yes', 'yes']
    classifier.train(objects, labels)
    # With only one label observed, probability should be 1
    assert classifier.probability(True, 'yes') == 1.0


def test_probabilities_reflect_training():
    classifier = nb.NaiveBayesClassifier(simple_featurizer)
    objects = [True, False]
    labels = ['spam', 'ham']
    classifier.train(objects, labels)
    probs = classifier.probabilities(True)
    assert probs['spam'] > probs['ham']


def test_accuracy_mismatched_lengths_raises():
    classifier = nb.NaiveBayesClassifier(simple_featurizer)
    classifier.train([True], ['spam'])
    with pytest.raises(ValueError):
        classifier.accuracy([True, False], ['spam'])


def test_classify_with_cost_matrix():
    classifier = nb.NaiveBayesClassifier(simple_featurizer)
    objects = [True, False]
    labels = ['spam', 'ham']
    classifier.train(objects, labels)
    cost_matrix = {
        'spam': {'spam': 0, 'ham': 0},
        'ham': {'spam': 10, 'ham': 1},
    }
    assert classifier.classify(False, cost_matrix) == 'spam'


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

