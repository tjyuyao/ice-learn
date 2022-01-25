<!-- markdownlint-disable -->

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `llutil.dictprocess`







---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dictprocess`

```python
dictprocess(f)
```

``ice.dictprocess`` is a function decorator that convert any function into a callable DictProcessor class that would take a dict as input and update its content.
The input arguments and return values of the function are automatically mapped to source and destination the keywords of the state dict being modified.


The input arguments mapping rule is simpler. A decorated DictProcessor class can specify fixed parameters at instantiation time, and dynamic parameters as state dict content at runtime.


The output arguments mapping is controlled by an extra argument at instantiation time called `dst` and the return value of the original function, may vary in different scenarios as shown in the following table:


| dst \ ret                     | `value`                               | `dict`                                                                                                                                           | `list` / `tuple`                              | `None`                       |
| ----------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- | ---------------------------- |
| `None`                        | Do not update, return value directly. | Update state dict with returned dict.                                                                                                            | Do not update, return list / tuple directly.  | Do nothing.                  |
| `str`                         | Update with `dict(dst=ret)`           | If `len(ret) == 1`, update with `dict(dst=ret.values()[0])`; If `dst in ret`, update with `dict(dst=ret[dst])`; else update with `dict(dst=ret)` | Update with `dict(dst=ret)`                   | Update with `dict(dst=None)` |
| `list` / `tuple`              | Update with `{dst[0]:ret}`            | Update with `{k:ret[k] for k in dst}`                                                                                                            | Update with `{k:v for k, v in zip(dst, ret)}` | Update with `{dst[0]:None}`  |
| `dict(update_key=return_key)` | Raise TypeError                       | Update with `{k:ret[rk] for k, rk in dst.items()}`                                                                                               | Raise TypeError                               | Raise TypeError              |




**Example:**



```python
import ice

@ice.dictprocess
def Add(x, y): return x+y

@ice.dictprocess
def Power(x, n): return pow(x, n)

pipeline = [
    Add(x="a", y="b", dst="c"),
    Power(x="c", n=2, dst="c"),
]
state_dict = {"a": 1, "b":2 }
for f in pipeline:
    state_dict == f(state_dict)
assert state_dict == {"a": 1, "b": 2, "c": 9}
```

The definition of operations minimizes the boilerplate, and the configuration phase is simple and concise. All these features enables best reusability for complex data processing pipelines.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Compose`

```python
Compose(translist: List[Callable[[Dict], Dict]])
```

a predefined DictProcessor that composes a list of other DictProcessors together.





---

<a href="https://github.com/tjyuyao/ice-learn/blob/main/ice/llutil/dictprocess.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `Collect`

```python
Collect(*keys)
```

a predefined DictProcessor that keep only selected entries.





