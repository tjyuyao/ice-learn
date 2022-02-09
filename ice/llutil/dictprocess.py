from functools import wraps
from inspect import signature
from typing import Callable, Dict, List

from .argparser import as_list, isa


DictProcessor = Callable[[Dict], Dict]


def dictprocess(f):
    r"""a decorator that convert function into a DictProcessor (`Callable[[Dict], Dict]`).
    
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

    Example:

    >>> import ice
    >>>
    >>> @ice.dictprocess
    >>> def Add(x, y): return x+y
    >>>
    >>> @ice.dictprocess
    >>> def Power(x, n): return pow(x, n)
    >>>
    >>> pipeline = [
    >>>     Add(x="a", y="b", dst="c"),
    >>>     Power(x="c", n=2, dst="c"),
    >>> ]
    >>> state_dict = {"a": 1, "b":2 }
    >>> for f in pipeline:
    >>>     state_dict == f(state_dict)
    >>> assert state_dict == {"a": 1, "b": 2, "c": 9}

    The definition of operations minimizes the boilerplate, and the configuration phase is simple and concise. All these features enables best reusability for complex data processing pipelines.

    """
    @wraps(f)
    def configurator(*args, dst=None, **kwds) -> DictProcessor:

        parameters = [k for k in signature(f).parameters]

        def wrapper(data={}):
            # pre process: remapping input keywords
            mapped = dict()
            unmapped = dict()
            for k in parameters:
                if k in kwds:
                    v = kwds[k]
                    if isa(v, str) and v in data:
                        mapped[k] = data[v]
                    elif isa(v, str):
                        unmapped[k] = v
                    else:
                        mapped[k] = v
                elif k in data:
                    mapped[k] = data[k]
                else:
                    pass  # use default value

            # call implementation
            try:
                ret = f(*args, **mapped, **unmapped)
            except Exception as e:
                if len(unmapped):
                    eargs = list(e.args)
                    eargs.append(
                        f"This may be caused by unmapped transform keyword arguments: {unmapped}.")
                    raise e.__class__(*eargs)
                raise e

            # post process: remapping output keywords
            if dst is None:
                if isa(ret, dict):
                    data.update(ret)  # test 1
                elif ret is None:
                    pass  # test 2
                else:
                    return ret  # test 3
            elif isa(dst, str):
                if isa(ret, dict) and dst in ret:
                    data[dst] = ret[dst]  # test 4
                elif isa(ret, dict) and len(ret) == 1:
                    for v in ret.values(): data[dst] = v  # test 10
                else:
                    data[dst] = ret  # test 5
            elif isa(dst, (list, tuple)):
                if isa(ret, dict): # test 6
                    for k in dst:
                        data[k] = ret[k]
                else: # test 7
                    for k, v in zip(dst, as_list(ret)):
                        data[k] = v
            elif isa(dst, dict) and isa(ret, dict): # test 8
                for tgtk, srck in dst.items():
                    data[tgtk] = ret[srck]
            else: # test 9
                raise TypeError(
                    f"Unsupported \"dst\" or return value for tranform \"{f}\".")

            return data
        return wrapper
    return configurator


def Compose(translist: List[DictProcessor]):
    """a predefined DictProcessor that composes a list of other DictProcessors together."""
    def _transform(data: Dict={}):
        for trans in as_list(translist):
            data = trans(data)
        return data
    return _transform


def Collect(*keys):
    """a predefined DictProcessor that keep only selected entries."""
    def _transform(data: Dict={}):
        ret = {k:data[k] for k in keys}
        return ret
    return _transform