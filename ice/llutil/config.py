import copy
from functools import partial
from inspect import Parameter, signature
from multiprocessing import get_logger
import traceback
from typing import Any, Tuple

from ice.llutil.argparser import as_list, isa
from ice.llutil.utils import auto_freeze_enabled


def _inplace_surrogate(cls, funcname):
    backupname = "__backup" + funcname
    setattr(cls, backupname, getattr(cls, funcname, None))
    def newfunc(self, *a, **k):
        try:
            cfg = object.__getattribute__(self, "_builder")
            if not cfg._frozen:
                try:
                    value = getattr(cfg, funcname)(*a, **k)
                except AttributeError as e:
                    get_logger().info(f"Try to automatically freeze due to AttributeError:\n{e.args[0]}")
                    self.auto_freeze()
                    if not cfg._frozen: raise
            if cfg._frozen:
                fn = getattr(cls, backupname)  # will get None instead of raise AttributeError if not hasattr(original_class, funcname), so
                if fn is None: raise AttributeError(funcname) # we raise it manually.
                value = fn(self, *a, **k)
        except AttributeError:
            # TODO: rephrase the exception handling part
            if funcname == "__getattribute__":
                if len(a) and a[0] in ('freeze', 'clone', 'update_params'):
                    return object.__getattribute__(cfg, *a)
                else:
                    try:
                        value = object.__getattribute__(cls, *a)
                    except AttributeError:
                        value = object.__getattribute__(self, *a)
            else:
                # assert False, f"{cls.__name__}.{funcname}({a}, {k}) frozen={cfg._frozen}\nAttributeError:{e}"
                raise
        return value
    setattr(cls, funcname, newfunc)


def configurable(cls):
    """This decorator delays the initialization of ``cls`` until ``freeze()``.

    Returns:
        decorated class which is now configurable.

    Example:

    >>> import ice
    >>>
    >>> @ice.configurable
    >>> class AClass:
    >>>     def __init__(self, a, b, c, d):
    >>>         self.a = a
    >>>         self.b = b
    >>>         self.c = c
    >>>         self.d = d
    >>>
    >>> # partial initialization.
    >>> i = AClass(b=0)
    >>>
    >>> # alter positional and keyword arguments afterwards.
    >>> i[0] = 2
    >>> i['b'] = 1
    >>> i.update({'c': 3, 'd': 4})
    >>> i.update(d=5)
    >>>
    >>> # unfrozen configurable can be printed as a legal construction python statement.
    >>> assert repr(i) == "AClass(a=2, b=1, c=3, d=5)"
    >>>
    >>> # real initialization of original object.
    >>> i.freeze()
    >>> assert i.a == 2 and i.b == 1
    """


    # do not wrap the same cls more than once.
    if is_configurable(cls):
        return cls

    # mark the cls so that is becomes a configurable class.
    cls.configurable_class_id = id(cls)

    try:
        object.__getattribute__(cls, "__freeze__")
    except AttributeError:
        # use the original initialization function as __freeze__.
        cls.__freeze__ = getattr(cls, "__init__")

    # substitute the original initialization function.
    def __new_init__(obj, *args, **kwds):
        if obj.configurable_class_id == cls.configurable_class_id:
            _Builder(cls, obj, *args, **kwds)
        else:  # called from subclass instance
            cls.__freeze__(obj, *args, **kwds)

    cls.__init__ = __new_init__

    # surrogate other functions
    _inplace_surrogate(cls, "__getattribute__")
    _inplace_surrogate(cls, "__getattr__")
    _inplace_surrogate(cls, "__str__")
    _inplace_surrogate(cls, "__getitem__")
    _inplace_surrogate(cls, "__setitem__")
    _inplace_surrogate(cls, "__call__")
    _inplace_surrogate(cls, "__repr__")
    _inplace_surrogate(cls, "extra_repr")
    _inplace_surrogate(cls, "__contains__")

    # return the modified cls
    return cls


def is_configurable(cls) -> bool:
    """check if a class or an object is configurable.

    Returns:
        bool

    Example:

    >>> import ice
    >>> import torch.nn as nn
    >>> ice.make_configurable(nn.Conv2d, nn.Linear)
    >>> assert ice.is_configurable(nn.Conv2d)

    """
    return getattr(cls, "configurable_class_id", None) in (id(cls), -1)


def has_builder(obj) -> bool:
    try:
        object.__getattribute__(obj, "_builder")
        return True
    except:
        return False


def frozen(obj):
    if is_configurable(obj):
        if has_builder(obj):
            builder = object.__getattribute__(obj, "_builder")
            return builder._frozen
        else:
            return obj._frozen
    else:
        return True  # view normal objects as a frozen version


def make_configurable(*classes):
    """This function converts multiple existing classes to configurables.

    Note:
        This have exactly the same effects of decorate each class with `@configurable` when defining the class.
        Each class only need to be decorated once, extra calling of conversion is ignored and has no side effects.

    Example:

    >>> import ice
    >>> import torch.nn as nn
    >>> ice.make_configurable(nn.Conv2d, nn.Linear)
    >>> assert ice.is_configurable(nn.Conv2d)

    """
    for cls in classes:
        configurable(cls)


def clone(obj, deepcopy=True):
    """clone configurables, containers, and ordinary objects recursively.

    Args:
        obj (configurable or list/dict of configurables): the configurable object to be cloned.
        deepcopy (bool, optional): copy resources by value. Defaults to True.

    Returns:
        Unfrozen copy of the original configurable.

    >>> import ice
    >>> import torch.nn as nn
    >>> ice.make_configurable(nn.Conv2d, nn.Linear)
    >>>
    >>> convcfg = nn.Conv2d(16, 8)
    >>>
    >>> conv1x1 = convcfg.clone()  # or ice.clone(convcfg)
    >>> conv1x1['kernel_size'] = 1
    >>> conv1x1.freeze()  # or ice.freeze(conv1x1)
    >>> assert conv1x1.kernel_size == (1, 1)
    >>>
    >>> conv3x3 = convcfg.clone()
    >>> conv3x3['kernel_size'] = 3
    >>> conv3x3.freeze()
    >>> assert conv3x3.kernel_size == (3, 3)
    """
    import torch
    if isinstance(obj, list):
        obj = [clone(x, deepcopy=deepcopy) for x in obj]
    elif isinstance(obj, tuple):
        obj = tuple(clone(x, deepcopy=deepcopy) for x in obj)
    elif isinstance(obj, dict):
        obj = obj.__class__({k: clone(v, deepcopy=deepcopy) for k, v in obj.items()})
    elif has_builder(obj):
        obj = objattr(obj, "_builder").clone(deepcopy=deepcopy)
    elif isinstance(obj, Configurable):
        obj = obj.clone()
    elif isinstance(obj, torch.Tensor):
        obj = obj.clone()
    elif deepcopy:
        obj = copy.deepcopy(obj)
    else:
        obj = copy.copy(obj)
    return obj


def freeze(obj):
    """freeze configurables recursively.

    **Freezing** is the process of building the configuration into real objects.
    Original `__init__()` functions of configurable classes declared by ``configurable``
     or ``make_configurable`` now will be called recursively to initialize the real instance,
     also known as the frozen version of a configurable.

    Args:
        obj (configurable or list/dict of configurables): the configurable object to be freeze.

    Returns:
        Frozen version of the original configurable.

    Note:
        Freezing happens in-place, ignoring the returned value is safe.
        If a user wants to reuse the configuration feature, he can clone() the
        object before or after frozen with the same effect.

    Example:
        See examples for ``configurable`` and ``clone``.
    """
    import torch
    if isinstance(obj, list):
        obj = [freeze(x) for x in obj]
    elif isinstance(obj, tuple):
        obj = tuple(freeze(x) for x in obj)
    elif isinstance(obj, dict):
        obj = obj.__class__({k: freeze(v) for k, v in obj.items()})
    elif has_builder(obj):
        obj = objattr(obj, "_builder").freeze()
    elif isinstance(obj, Configurable):
        obj = obj.freeze()
    elif isinstance(obj, torch.Tensor):
        obj = obj.clone()
    return obj


def objattr(obj, attrname):
    return object.__getattribute__(obj, attrname)


class Configurable:

    configurable_class_id = -1

    def __init__(self, *args, **kwds) -> None:
        try:
            objattr(self, "_cls")
        except AttributeError:
            self._obj = self
            self._cls = self.__class__

        param_signs = list(signature(self._cls.__freeze__).parameters.items())[1:]
        self._args_only = []
        self._args_only_count = 0
        self._args_only_names = {}
        self._args_or_kwds = {}
        self._argnames = []
        self._var_args = []
        self._var_args_name = None
        self._kwds_only = {}
        self._var_kwds = {}
        self._var_kwds_name = None
        for name, param in param_signs:
            if param.kind == Parameter.POSITIONAL_ONLY:
                self._args_only.append(param.default)
                self._args_only_names[name] = self._args_only_count
                self._args_only_count += 1
            elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                self._args_or_kwds[name] = param.default
                self._argnames.append(name)
            elif param.kind == Parameter.VAR_POSITIONAL:
                self._var_args_name = name
            elif param.kind == Parameter.KEYWORD_ONLY:
                self._kwds_only[name] = param.default
            elif param.kind == Parameter.VAR_KEYWORD:
                self._var_kwds_name = name
            else:
                assert False, "unknown param kind"

        self._all_args_count = self._args_only_count + len(self._args_or_kwds)
        self._all_kwds_names = list(self._args_only_names.keys()) + self._argnames + [self._var_args_name, self._var_kwds_name] + list(self._kwds_only.keys())

        self._frozen = False
        self.update_params(*args, **kwds)

    def update_params(self, *args, **kwds):
        # if provided args for position-only params, all of them should be provided.
        if self._args_only_count and len(args):
            self._args_only = [None] * self._args_only_count
            for idx in range(self._args_only_count):
                try:
                    self._args_only[idx] = args[idx]
                except IndexError:
                    raise TypeError(f" {self._cls.__name__} missing 1 required position-only argument '{self._args_only_names[idx]}'.")
            args = args[idx+1:]
        # args or kwds
        if len(self._args_or_kwds) and len(args):
            for idx, arg in enumerate(args):
                self[self._args_only_count + idx] = arg
            args = args[idx+1:]
        # var_args
        if self._var_args_name is not None and len(args):
            self._var_args = args
            args = []
        if len(args):
            raise TypeError(f"{self._cls.__name__} doesn't accept variable position-only arguments.")
        # kwds
        for name, arg in kwds.items():
            self[name] = arg
        return self._obj

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                raise KeyError(key)
            elif key < self._args_only_count:
                return self._args_only[key]
            elif key < self._all_args_count:
                key = self._argnames[key - self._args_only_count]
                return self._args_or_kwds[key]
            else:
                raise KeyError(key)
        else:
            if key in self._args_or_kwds:
                return self._args_or_kwds[key]
            elif key in self._kwds_only:
                return self._kwds_only[key]
            elif key == self._var_args_name:
                return self._var_args
            elif key == self._var_kwds_name:
                return self._var_kwds
            elif key in self._args_only_names:
                idx = self._args_only_names[key]
                return self._args_only[idx]
            elif key in self._var_kwds:
                return self._var_kwds[key]
            else:
                raise KeyError(key)

    def __setitem__(self, key, value):
        if self._frozen:
            # traceback.print_stack()
            raise RuntimeError(f"Frozen configuration {self._cls.__name__} can not be altered, please use clone() at proper time.")
        if isa(key, tuple):
            for i in key:
                if i in self:
                    self.__setitem__(i, value)
                    break
            else:
                raise KeyError(key)
        else:
            if isinstance(key, int):
                if key < 0:
                    raise KeyError(key, value)
                elif key < self._args_only_count:
                    self._args_only[key] = value
                elif key < self._all_args_count:
                    key = self._argnames[key - self._args_only_count]
                    self._args_or_kwds[key] = value
                else:
                    raise KeyError(key, value)
            else:
                if key in self._args_or_kwds:
                    self._args_or_kwds[key] = value
                elif key in self._kwds_only:
                    self._kwds_only[key] = value
                elif key == self._var_args_name:
                    self._var_args = as_list(value)
                elif key == self._var_kwds_name:
                    if not isa(value, dict):
                        raise TypeError(f"direct assignment of '**{key}' argument of '{self._cls.__name__}' type should be a dict.")
                    self._var_kwds = value
                elif key in self._args_only_names:
                    idx = self._args_only_names[key]
                    self._args_only[idx] = value
                elif self._var_kwds_name is not None:
                    self._var_kwds[key] = value
                else:
                    raise KeyError(key, f"'{self._cls.__name__}' does not accept variable keywords arguments.")

    def __contains__(self, key):
        if isa(key, int):
            return key >= 0 and key < self._all_args_count
        elif isa(key, str):
            return key in self._all_kwds_names
        else:
            raise TypeError(key)

    def __str__(self):
        args = []
        if self._args_only_count:
            args.append(', '.join([str(arg) for arg in self._args_only]))
        if len(self._args_or_kwds):
            args.append(', '.join([f"{k}={v}" for k, v in self._args_or_kwds.items() if v != Parameter.empty]))
        if len(self._var_args):
            args.append(', '.join([str(arg) for arg in self._var_args]))
        if len(self._kwds_only):
            args.append(', '.join([f"{k}={v}" for k, v in self._kwds_only.items() if v != Parameter.empty]))
        if len(self._var_kwds):
            args.append(', '.join([f"{k}={v}" for k, v in self._var_kwds.items() if v != Parameter.empty]))
        args = ', '.join(args)
        return f"{objattr(self, '_cls').__name__}({args})"

    def extra_repr(self):
        return ''

    def __getattr__(self, attrname):
        if objattr(self, "_frozen"):
            raise AttributeError(f"\"{self._cls.__name__}\" does not have attribute \"{attrname}\".")
        else:
            raise AttributeError(f"Configurable \"{self._cls.__name__}\" is not frozen, which may be the reason of not having attribute \"{attrname}\".")

    def clone(self, deepcopy=True):
        return self._cls(
            *clone(self._args_only, deepcopy=deepcopy),
            *clone(self._args_or_kwds, deepcopy=deepcopy).values(),
            *clone(self._var_args, deepcopy=deepcopy),
            **clone(self._kwds_only, deepcopy=deepcopy),
            **clone(self._var_kwds, deepcopy=deepcopy)
            )

    def freeze(self):
        # make twice freezing legal.
        if self._frozen: return self._obj
        # mark as frozen so that orig__init__ will be able to manipulate the original instance via __getattribute__.
        self._frozen = True
        # initialize the class instance
        self._cls.__freeze__(self._obj, *self._args_only, *self._args_or_kwds.values(), *self._var_args, **self._kwds_only, **self._var_kwds)
        return self._obj

    def auto_freeze(self):

        if not auto_freeze_enabled(): return self._obj

        def all_args():
            for arg in self._args_only: yield arg
            for arg in self._args_or_kwds.values(): yield arg
            for arg in self._var_args: yield arg
            for arg in self._kwds_only.values(): yield arg
            for arg in self._var_kwds.values(): yield arg

        for arg in all_args():
            if arg is Parameter.empty:
                break
        else:
            self.freeze()

        return self._obj

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.clone().update_params(*args, **kwds).auto_freeze()

class _Builder(Configurable):
    """This class stores the arguments and meta informations that is needed by ``configurable`` decorator.

    This class is not intended to be visited directly by end user. See ``configurable`` for usage guide instead.
    """

    def __init__(self, cls, obj, *args, **kwds) -> None:
        self._cls = cls
        self._obj = obj
        super().__init__(*args, **kwds)
        object.__setattr__(obj, "_builder", self)
        self.auto_freeze()

    def __reduce__(self) -> Tuple[Any, ...]:
        assert not self._frozen, f"{self}"
        return (partial(self._cls, *self._args_only, *self._args_or_kwds.values(), *self._var_args, **self._kwds_only, **self._var_kwds), ())
    
    def __repr__(self):
        return str(self)
