# Hybrid Naive Bayes

[![PyPI version](https://img.shields.io/pypi/v/hybrid-naive-bayes.svg)](https://pypi.org/project/hybrid-naive-bayes/)
[![CI](https://github.com/ashkonfarhangi/hybrid-naive-bayes/actions/workflows/ci.yml/badge.svg)](https://github.com/ashkonfarhangi/hybrid-naive-bayes/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ashkonfarhangi/hybrid-naive-bayes/branch/main/graph/badge.svg)](https://codecov.io/gh/ashkonfarhangi/hybrid-naive-bayes)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Pytest](https://img.shields.io/badge/tested%20with-pytest-blue)](https://docs.pytest.org/en/stable/)
[![Pyright](https://img.shields.io/badge/checked%20with-pyright-blue)](https://github.com/microsoft/pyright)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ashkonfarhangi/hybrid-naive-bayes/main.svg)](https://results.pre-commit.ci/latest/github/ashkonfarhangi/hybrid-naive-bayes/main)

A generalized implementation of the Naive Bayes classifier that supports features of arbitrary type.

## Features

- **Mixed Feature Types**: Simultaneously use categorical and ordered (both discrete and continuous) features in the same model.
- **Flexible Distribution Modeling**: Model ordered features with any probability distribution, not just Gaussian. This library includes several common distributions and allows you to define your own.
- **Cost-Sensitive Classification**: Minimize classification errors by providing a cost matrix that reflects the real-world costs of misclassification.
- **No External Dependencies**: The core logic in `src/nb.py` and `src/distributions.py` is written in pure Python 3 with no external libraries.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Code Quality](#code-quality)

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for package management. To create a virtual environment and install dependencies, run:

```bash
uv sync
```

This will create a `.venv` directory and install the packages listed in `pyproject.toml`.

If you prefer not to use the virtual environment, the core modules `src/nb.py` and `src/distributions.py` have no external dependencies and can be copied directly into your project.

## Usage

Here’s a simple example of how to train and use the Naive Bayes classifier.

First, define a `featurizer` function to convert your raw data into a list of `Feature` objects. Each feature specifies its name, distribution, and value.

```python
import nb
import distributions

def featurizer(data_point: list[str]) -> list[nb.Feature]:
    return [
        nb.Feature("Checking account status", distributions.Multinomial, data_point[0]),
        nb.Feature("Duration in months", distributions.Exponential, float(data_point[1])),
        nb.Feature("Credit history", distributions.Multinomial, data_point[2]),
        nb.Feature("Credit amount", distributions.Gaussian, float(data_point[4])),
    ]
```

Next, create an instance of the classifier, passing your `featurizer` to the constructor. Then, call the `train` method with your training data and labels.

```python
# Sample data
training_data = [
    ['A11', '6', 'A34', '1169'],
    ['A12', '48', 'A32', '5951'],
    ['A14', '12', 'A34', '2096'],
]
labels = ['good', 'bad', 'good']

# Initialize and train the classifier
classifier = nb.NaiveBayesClassifier(featurizer)
classifier.train(training_data, labels)
```

Finally, you can classify new data points using the `classify` method or get the probability distribution over all labels with the `probabilities` method.

```python
# Classify a new data point
new_data_point = ['A11', '24', 'A32', '3000']
prediction = classifier.classify(new_data_point)
print(f"Prediction: {prediction}")

# Get probabilities for the new data point
probabilities = classifier.probabilities(new_data_point)
print(f"Probabilities: {probabilities}")
```

For a more detailed and runnable example, see `src/test.py`. It demonstrates training the classifier on a real-world credit dataset from UCI.

## Development

To set up a development environment, install the project with its "dev" dependencies:

```bash
uv sync --dev
```

This will install development tools like `pre-commit`, `ruff`, and `pyright`.

### Pre-Commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality and run tests before each commit. To install the hooks, run:

```bash
uv run pre-commit install
```

To run all checks on your staged files:

```bash
uv run pre-commit run
```

Or to run them on all files:
```bash
uv run pre-commit run --all-files
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Code Quality

Static checks and formatting are handled with [ruff](https://github.com/astral-sh/ruff) and [pyright](https://github.com/microsoft/pyright). They can be executed via:

```bash
uv run ruff format
uv run ruff check
uv run pyright
```
