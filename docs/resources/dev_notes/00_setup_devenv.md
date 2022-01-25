# Setup Environment

We use `poetry` as the virtualenv as well as project manager (e.g. dependencies, packaging, publishing, etc.).

Please read about poetry (see REFERNCES section at the end of this page) and the `pyproject.toml` file before you run following commands to setup your local develop environment. Having another pip installed release version of ice will not cause a problem since poetry isolates the environment.


## Pypi Mirror for China (Optional)

```bash
mkdir -p ~/.pip
echo "[config]\nindex-url = https://pypi.douban.com/simple" > ~/.pip/pip.conf
```

## Steps

1. Install poetry following the instruction [here](https://python-poetry.org/docs/#installation).
1. Set tab-completion for poetry following the instruction [here](https://python-poetry.org/docs/master/#enable-tab-completion-for-bash-fish-or-zsh)
1. `git clone https://github.com/tjyuyao/ice-learn`
1. `cd ice-learn`
1. `poetry install -E pycuda`
1. `poetry shell`
1. Set tab-completion for poe-the-poet following the instruction [here](https://github.com/nat-n/poethepoet#enable-tab-completion-for-your-shell).
1. Install torch and torchvision manually for correct cuda version using pip in the poetry shell, e.g.:
    `pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

## References

- [Poetry First Impression](https://python-poetry.org/)
- [Poetry Tutorial](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f)
- [Poetry API Reference](https://python-poetry.org/docs/cli/)
