# Setup environment for developing ice.

We use `poetry` as the virtualenv as well as project manager (e.g. dependencies, packaging, publishing, etc.).

Please read about poetry (see REFERNCES section at the end of this page) and the `pyproject.toml` file before you run following commands to setup your local develop environment. Having another pip installed release version of ice will not cause a problem since poetry isolates the environment.

## Steps

1. Install poetry following the instruction [here](https://python-poetry.org/docs/#installation).
2. `git clone https://github.com/tjyuyao/ice-learn`
3. `cd ice-learn`
4. `poetry install -E pycuda`
5. `poetry shell`
6. Set tab-completion for poe-the-poet following the instruction [here](https://github.com/nat-n/poethepoet#enable-tab-completion-for-your-shell).

## References

- [Poetry First Impression](https://python-poetry.org/)
- [Poetry Tutorial](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f)
- [Poetry API Reference](https://python-poetry.org/docs/cli/)
