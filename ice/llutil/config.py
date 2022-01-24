import torch
import copy
from functools import wraps


def _inplace_surrogate(cls, funcname):
    backupname = "__backup" + funcname
    setattr(cls, backupname, getattr(cls, funcname, None))
    def newfunc(self, *a, **k):
        try:
            cfg = object.__getattribute__(self, "_config")
            if cfg._frozen:
                fn = getattr(cls, backupname)  # will get None instead of raise AttributeError if not hasattr(original_class, funcname), so
                if fn is None: raise AttributeError(funcname) # we raise it manually.
                value = fn(self, *a, **k)
            else:
                value = getattr(cfg, funcname)(*a, **k)
        except AttributeError:
            # this will only occur during unpickling, the unpickler tries to retrieve some class meta variables.
            if funcname == "__getattribute__":
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

    # backup the original initialization function.
    cls.orig__init__ = getattr(cls, "__init__", None)
    
    # substitute the original initialization function.
    @wraps(cls.orig__init__)
    def make_config(self, *args, **kwds):
        argnames = cls.orig__init__.__code__.co_varnames[1:]
        object.__setattr__(self, "_config", _Config(cls, self, argnames, *args, **kwds))
    cls.__init__ = make_config

    # surrogate other functions
    _inplace_surrogate(cls, "__getattribute__")
    _inplace_surrogate(cls, "__getattr__")
    _inplace_surrogate(cls, "__repr__")
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
    elif is_configurable(obj):
        obj = obj._config.clone(deepcopy=deepcopy)
    elif isinstance(obj, torch.Tensor):
        obj = obj.clone()
    elif deepcopy:
        obj = copy.deepcopy(obj)
    else:
        obj = copy.copy(obj)
    return obj


def freeze(obj, deepcopy=True):
    """freeze configurables recursively.

    **Freezing** is the process of building the configuration into real objects.
    Original `__init__()` functions of configurable classes declared by ``configurable``
     or ``make_configurable`` now will be called recursively to initialize the real instance,
     also known as the frozen version of a configurable.

    Args:
        obj (configurable or list/dict of configurables): the configurable object to be freeze.
        deepcopy (bool, optional): copy resources by value. Defaults to True.

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
        obj = [freeze(x, deepcopy=deepcopy) for x in obj]
    elif isinstance(obj, tuple):
        obj = tuple(freeze(x, deepcopy=deepcopy) for x in obj)
    elif isinstance(obj, dict):
        obj = obj.__class__({k: freeze(v, deepcopy=deepcopy) for k, v in obj.items()})
    elif is_configurable(obj):
        obj = obj._config.freeze()
    elif isinstance(obj, torch.Tensor):
        obj = obj.clone()
    elif deepcopy:
        obj = copy.deepcopy(obj)
    else:
        obj = copy.copy(obj)
    return obj


class _Config:
    """This class stores the arguments and meta informations that is needed by ``configurable`` decorator.

    This class is not intended to be visited directly by end user. See ``configurable`` for usage guide instead.
    """
    
    def __init__(self, cls, obj, argnames, *args, **kwds) -> None:
        self._cls = cls
        self._obj = obj
        self._argnames = argnames
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

    def __getattr__(self, attrname):
        reprstr = repr(self)
        if len(reprstr) > 60: reprstr = reprstr[:60] + " ... "
        raise AttributeError(f"Configurable \"{reprstr}\" is not frozen before being used.")

    def update(self, explicit={}, **implicit):
        explicit.update(implicit)
        for k, v in explicit.items():
            self[k] = v
    
    def clone(self, deepcopy=True):
        return self._cls(**clone(self._kwds, deepcopy=deepcopy))

    def freeze(self, deepcopy=True):
        # mark as frozen so that orig__init__ will be able to manipulate the original instance via __getattribute__.
        self._frozen = True
        # initialize the class instance
        self._cls.orig__init__(self._obj, **freeze(self._kwds, deepcopy=deepcopy))
        return self._obj

    def __str__(self):
        kwds = ', '.join([f"{k}={repr(self._kwds[k])}" for k in self._argnames if k in self._kwds])
        return f"{self._cls.__name__}({kwds})"