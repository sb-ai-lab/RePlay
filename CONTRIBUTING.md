# How to Contribute
We welcome community contributions to RePlay. You can:

- Submit your changes directly with a [pull request](https://github.com/sb-ai-lab/RePlay/pulls).
- Log a bug or make a feature request with an [issue](https://github.com/sb-ai-lab/RePlay/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [development](#development) before you proceed.

## Development

Development of any feature is organized in separate branches with naming conventions:
- *feature/feature_name* - regular feature.
- *release/vX.Y.Z* - release branch (for details see [versioning][#versioning]).

How to add a new model is described [here](https://sb-ai-lab.github.io/RePlay/pages/installation.html#adding-new-model).

When you're done with your feature development please create [pull request](#pull-requests).

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Cover your code by [unit tests](https://github.com/sb-ai-lab/RePlay/tree/main/tests). 
- For a larger feature, provide a relevant [example](https://github.com/sb-ai-lab/RePlay/tree/main/experiments).
- [Document](#documentation-guidelines) your code.
- [Submit](https://github.com/sb-ai-lab/RePlay/pulls) a pull request into the `main` branch.

Public CI is enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

## Documentation Guidelines

RePlay uses `Sphinx` for inline comments in public header files that are used to build the API reference and the Developer Guide. See [RePlay documentation](https://sb-ai-lab.github.io/RePlay/index.html) for reference.

## Versioning

We use the following versioning rules:
XX.YY.ZZ, where:
- XX = 0 until the framework is not mature enough (will provide the separate notice when we're ready to switch to XX = 1).
- YY is incrementing in case when backward compatibility is broken.
- ZZ is incrementing in case of minor changes or bug fixes which are not broken backward compatibility.

## Release Process

To release the new version of the product:
- Change version according to [versioning](#versioning) in [config](https://github.com/sb-ai-lab/RePlay/blob/main/pyproject.toml).
- Create the release branch according to [development](#development) conventions.
- Add tag with the appropriate version.
- Add the newly created release on the [releases](https://github.com/sb-ai-lab/RePlay/releases) tab. 

---
**Note:** RePlay is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---