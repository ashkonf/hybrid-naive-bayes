import sys
from pathlib import Path

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
