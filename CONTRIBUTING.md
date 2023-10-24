# Table of contents

- [Contributing to RePlay](#contributing-to-replay)
- [Codebase structure](#codebase-structure)
- [Developing RePlay](#developing-replay)
- [Writing Documentation](#writing-documentation)
- [Style Guide](#style-guide)
- [Versioning](#versioning)
- [Release Process](#release-process)

# Contributing to RePlay

We welcome community contributions to RePlay. You can:

- Submit your changes directly with a [pull request](https://github.com/sb-ai-lab/RePlay/pulls).
- Log a bug or make a feature request with an [issue](https://github.com/sb-ai-lab/RePlay/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [development](#developing-replay) before you proceed.

# Codebase structure

To be defined.

# Developing RePlay

Development of any feature is organized in separate branches with naming conventions:
- *feature/feature_name* - regular feature.
- *release/vX.Y.Z* - release branch (for details see [versioning][#versioning]).

## Installation 

### Basic

    ```bash
    pip install replay-rec
    ```

### Troubleshooting

### General

If you have an installation trouble, update the core packages:

    ```bash
    pip install --upgrade pip wheel
    ```

### RePlay dependencies compilation

RePlay depends on packages (e.g. LightFM, Implicit) that perform C/C++ extension compilation on installation. This requires C++ compiler, header files and other necessary components to be installed.

An example of error indicating header files absence is: Python.h: No such file or directory

To install the necessary packages run the following for Ubuntu:

    ```bash
    sudo apt-get install python3-dev
    sudo apt-get install build-essential
    ```

### Installing from the source

If you are installing from the source, you will need Python 3.8-3.10.

1. Install poetry using [the poetry installation guide](https://python-poetry.org/docs/#installation). 

2. Clone the project to your own local machine:

    ```bash
    git clone git@github.com:sb-ai-lab/RePlay.git
    cd RePlay
    ```
3. **Optional**: specify python for poetry

    ```bash
    poetry env use PYTHON_PATH
    ```

4. Install RePlay:

    ```bash
    pip install -U pip wheel
    pip install -U requests pypandoc cython optuna poetry
    poetry build
    pip install --force-reinstall dist/replay_rec-0.10.0-py3-none-any.whl
    ```

After that, there is virtual environment, where you can test and implement your own code.
So, you don't need to rebuild the full project every time.
Each change in the code will be reflected in the library inside the environment.

## Style Guide

We follow [the standard python PEP8](https://www.python.org/dev/peps/pep-0008/) conventions for style.

### Automated code checking

In order to automate checking of the code quality, please run:

    ```bash
    pycodestyle --ignore=E203,E231,E501,W503,W605 --max-doc-length=160 replay tests
    pylint --rcfile=.pylintrc replay
    ```

## How to add a new model
How to add a new model is described [here](https://sb-ai-lab.github.io/RePlay/pages/installation.html#adding-new-model).

When you're done with your feature development please create [pull request](#pull-requests).

## Testing

Before making a pull request (despite changing only the documentation or writing new code), please check your code on tests:

    ```bash
    pytest --cov=replay --cov-report=term-missing --doctest-modules replay --cov-fail-under=93 tests
    ```

Also if you develop new functionality, please add your own tests.

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Cover your code by [unit tests](https://github.com/sb-ai-lab/RePlay/tree/main/tests). 
- For a larger feature, provide a relevant [example](https://github.com/sb-ai-lab/RePlay/tree/main/examples).
- [Document](#documentation-guidelines) your code.
- [Submit](https://github.com/sb-ai-lab/RePlay/pulls) a pull request into the `main` branch.

Public CI is enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

# Writing Documentation

RePlay uses `Sphinx` for inline comments in public header files that are used to build the API reference and the Developer Guide. See [RePlay documentation](https://sb-ai-lab.github.io/RePlay/index.html) for reference.

# Versioning

We use the following versioning rules:
XX.YY.ZZ, where:
- XX = 0 until the framework is not mature enough (will provide the separate notice when we're ready to switch to XX = 1).
- YY is incrementing in case when backward compatibility is broken.
- ZZ is incrementing in case of minor changes or bug fixes which are not broken backward compatibility.

# Release Process

To release the new version of the product:
- Change version according to [versioning](#versioning) in [config](https://github.com/sb-ai-lab/RePlay/blob/main/pyproject.toml).
- Create the release branch according to [development](#development) conventions.
- Add tag with the appropriate version.
- Add the newly created release on the [releases](https://github.com/sb-ai-lab/RePlay/releases) tab. 

---
**Note:** RePlay is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---