# Hybrid Naive Bayes

[![PyPI version](https://img.shields.io/pypi/v/your-package)](link-to-pypi-page)
[![codecov](https://codecov.io/github/ashkonf/hybrid-naive-bayes/graph/badge.svg?token=7Y596J8IYZ)](https://codecov.io/github/ashkonf/hybrid-naive-bayes)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pytest](https://img.shields.io/badge/pytest-✓-brightgreen)](https://docs.pytest.org)
[![Pyright](https://img.shields.io/badge/pyright-✓-green)](https://github.com/microsoft/pyright)
[![Ruff](https://img.shields.io/badge/ruff-✓-blue?logo=ruff)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ashkonf/hybrid-naive-bayes/ci.yml?branch=main)](https://github.com/ashkonf/hybrid-naive-bayes/actions/workflows/ci.yml?query=branch%3Amain)

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
