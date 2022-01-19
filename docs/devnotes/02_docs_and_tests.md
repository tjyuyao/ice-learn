# About documentation generation, unit test and documentation-based test

## `pytest`

We use `pytest` for unittesting. For example,

```bash
poetry run pytest tests/llutil/test_config.py
```

## `xdoctest`

We use `xdoctest` for testing demostration code in docstring. For example,

```bash
poetry run python -m xdoctest ice/llutil/config.py
```

## `lazydocs`

We integrated a modified `lazydocs` script for generate markdown documentations from docstring. For example,

```bash
poetry run python ./docs/build_references.py
```
