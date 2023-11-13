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

# Developing RePlay

Development of any feature is organized in separate branches with naming conventions:
- *feature/feature_name* - regular feature.
- *release/X.Y.Z* - release branch (for details see [versioning](#versioning)).

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

RePlay depends on packages that perform C/C++ extension compilation on installation. This requires C++ runtime to be installed.

An example of error indicating header files absence is: Python.h: No such file or directory

To install the necessary packages run the following instructions:

    ```bash
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

3. Install RePlay:

    ```bash
    pip install poetry==1.5.1
    ./poetry_wrapper.sh install
    ```
    **If you need to install Replay with the experimental submodule**:
    ```bash
    pip install poetry==1.5.1 lightfm==1.17
    ./poetry_wrapper.sh --experimental install
    ```
    After that, there is an environment, where you can test and implement your own code.
    So, you don't need to rebuild the full project every time.
    Each change in the code will be reflected in the library inside the environment.


4. **optional**: Build wheel package:

    ```bash
    ./poetry_wrapper.sh build
    ```
    **If you need to build Replay package with the experimental submodule**:
    ```bash
    ./poetry_wrapper.sh --experimental build
    ```
    You can find the assembled package in the ``dist`` folder.


5. **optional**: Сhanging dependencies:
    - If you want to make changes in the pyproject.toml file then change the projects/pyproject.toml.template file. There may be inserts in it that relate only to the main part of the library or experimental. In this case, it is necessary to make inserts in Jinja2-like syntax. For example:
    ```bash
    {% if project == "default" %}
    {% endif %}
    ```
    or
    ```bash
    {% if project == "experimental" %}
    {% endif %}
    ```
    - After updating the pyproject.toml file, you need to make changes to the poetry.lock file.
    ```bash
    ./poetry_wrapper.sh lock
    ```
    For the experimental module.
    ```bash
    ./poetry_wrapper.sh --experimental lock
    ```
    Note that during this step, updated poetry.lock file do not need to be copied anywhere.


## Style Guide

We follow [the standard python PEP8](https://www.python.org/dev/peps/pep-0008/) conventions for style.

### Automated code checking

In order to automate checking of the code quality, please run:

    ```bash
    pycodestyle replay tests
    pylint replay
    ```

## How to add a new model
How to add a new model is described [here](https://sb-ai-lab.github.io/RePlay/pages/development.html#adding-new-model).

When you're done with your feature development please create [pull request](#pull-requests).

## Testing

Before making a pull request (despite changing only the documentation or writing new code), please check your code on tests:

    ```bash
    pytest
    ```

Also if you develop new functionality, please add your own tests.

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Cover your code by [unit tests](https://github.com/sb-ai-lab/RePlay/tree/main/tests). 
- For a larger feature, provide a relevant [example](https://github.com/sb-ai-lab/RePlay/tree/main/experiments).
- [Document](#documentation-guidelines) your code.
- [Submit](https://github.com/sb-ai-lab/RePlay/pulls) a pull request into the `main` branch.

Public CI is enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub repository.

# Writing Documentation

RePlay uses `Sphinx` for inline comments in public header files that are used to build the API reference and the Developer Guide. See [RePlay documentation](https://sb-ai-lab.github.io/RePlay/index.html) for reference.

# Versioning

We use the following versioning rules:
XX.YY.ZZ, where:
- XX = 0 until the framework is not mature enough (will provide the separate notice when we're ready to switch to XX = 1).
- YY is incrementing in case when backward compatibility is broken.
- ZZ is incrementing in case of minor changes or bug fixes which are not broken backward compatibility.

For the packages with the `experimental` submodule we use additional suffix `rc`. The default package and the package with `experimental` submodule have synchronous versions. For example:
- 0.13.0 (default package)
- 0.13.0rc0 (default package with `experimental` submodule)

# Release Process

To release the new version of the product:
- Change version according to [versioning](#versioning) in [config](https://github.com/sb-ai-lab/RePlay/blob/main/projects/pyproject.toml.template).
- Create the release branch according to [development](#development) conventions.
- Add tag with the appropriate version.
- Add the newly created release on the [releases](https://github.com/sb-ai-lab/RePlay/releases) tab. 

---
**Note:** RePlay is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---