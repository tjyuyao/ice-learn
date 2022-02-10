import copy
from functools import partial
from inspect import Parameter, signature
from typing import Any

import torch
from ice.llutil.multiprocessing import in_main_process


def _inplace_surrogate(cls, funcname):
    backupname = "__backup" + funcname
    setattr(cls, backupname, getattr(cls, funcname, None))
    def newfunc(self, *a, **k):
        try:
            cfg = object.__getattribute__(self, "_builder")
            if cfg._frozen:
                fn = getattr(cls, backupname)  # will get None instead of raise AttributeError if not hasattr(original_class, funcname), so
                if fn is None: raise AttributeError(funcname) # we raise it manually.
                value = fn(self, *a, **k)
            else:
                value = getattr(cfg, funcname)(*a, **k)
        except AttributeError:
            if funcname == "__getattribute__":
                if len(a) and a[0] == 'freeze':
                    return object.__getattribute__(cfg, *a)
                else:
                    value = object.__getattribute__(cls, *a)
            elif funcname == "__getattr__":
                raise AttributeError(*a)
            else:
                assert False
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
    cls.is_configurable = True

    if not hasattr(cls, "__freeze__"):
        # use the original initialization function as __freeze__.
        cls.__freeze__ = getattr(cls, "__init__")

    # substitute the original initialization function.
    def make_config(obj, *args, **kwds):
        object.__setattr__(obj, "_builder", _Builder(cls, obj, *args, **kwds))
    cls.__init__ = make_config

    # surrogate other functions
    _inplace_surrogate(cls, "__getattribute__")
    _inplace_surrogate(cls, "__getattr__")
    _inplace_surrogate(cls, "__str__")
    _inplace_surrogate(cls, "__getitem__")
    _inplace_surrogate(cls, "__setitem__")

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
    return hasattr(cls, "is_configurable")


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

    is_configurable = True

    def __init__(self, *args, **kwds) -> None:
        try:
            objattr(self, "_cls")
        except AttributeError:
            self._obj = self
            self._cls = self.__class__
        self._argnames = self._cls.__freeze__.__code__.co_varnames[1:]
        self._kwds = kwds
        self._frozen = False
        for i, v in enumerate(args):
            self[i] = v

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self._argnames[key]
        return self._kwds[key]

    def __setitem__(self, key, value):
        assert not self._frozen, "Frozen configuration can not be altered, please use clone() at proper time."
        if isinstance(key, int):
            key = self._argnames[key]
        self._kwds[key] = value

    def __contains__(self, key):
        return key in self._kwds

    def update(self, explicit={}, **implicit):
        explicit.update(implicit)
        for k, v in explicit.items():
            self[k] = v

    def __str__(self):
        kwds = ', '.join([f"{k}={str(objattr(self, '_kwds')[k])}" for k in objattr(self, '_argnames') if k in objattr(self, '_kwds')])
        return f"{objattr(self, '_cls').__name__}({kwds})"

    def __getattr__(self, attrname):
        if objattr(self, "_frozen"):
            raise AttributeError(f"\"{self._cls.__name__}\" does not have attribute \"{attrname}\".")
        else:
            raise AttributeError(f"Configurable \"{str(self)}\" is not frozen, which may be the reason of not having attribute \"{attrname}\".")

    def clone(self, deepcopy=True):
        return self._cls(**clone(self._kwds, deepcopy=deepcopy))

    def freeze(self):
        # make twice freezing legal.
        if self._frozen: return self._obj
        # mark as frozen so that orig__init__ will be able to manipulate the original instance via __getattribute__.
        self._frozen = True
        # initialize the class instance
        self._cls.__freeze__(self._obj, **self._kwds)
        return self._obj


class _Builder(Configurable):
    """This class stores the arguments and meta informations that is needed by ``configurable`` decorator.

    This class is not intended to be visited directly by end user. See ``configurable`` for usage guide instead.
    """

    def __init__(self, cls, obj, *args, **kwds) -> None:
        self._cls = cls
        self._obj = obj
        super().__init__(*args, **kwds)
        if not in_main_process():
            # auto freeze()
            for argname, arg in signature(self._cls.__freeze__).parameters.items():
                if arg.kind == Parameter.VAR_POSITIONAL:
                    if argname not in self._kwds:
                        break
                    else: continue
                else: continue
            else:
                self.freeze()

    
    def __reduce__(self) -> tuple[Any, ...]:
        return (partial(self._cls, **self._kwds), ())