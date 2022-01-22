# About documentation generation, unit test and documentation-based test

## `pytest`

We use `pytest` for unittesting. For example,

```bash
poetry run pytest tests/llutil/test_config.py -vs
```

where `-s` will print outputs for failed tests.

Run specific tests with `test_mod.py::TestClass:test_method` or `test_mod.py::test_function`.

```bash
poetry run pytest tests/llutil/test_multiprocessing.py::test_reduce -v
```

Option `-m slow` will run tests decorated with the `@pytest.mark.slow` decorator.

```bash
poetry run pytest tests/llutil/test_multiprocessing.py -v -m slow
```

Option `-m "not slow"` will run tests not decorated with the `@pytest.mark.slow` decorator.

```bash
poetry run pytest tests/llutil/ -v -m "not slow and not cuda"
```

Configurations are in `pytest.ini`.

## `xdoctest`

We use `xdoctest` for testing demostration code in docstring. For example,

```bash
poetry run python -m xdoctest ice/llutil/config.py
```

or

```bash
poetry run pytest ice/llutil/ -v --xdoc
```


## `lazydocs`

We integrated a modified `lazydocs` script for generate markdown documentations from docstring. For example,

```bash
poetry run python ./docs/build_references.py
```
