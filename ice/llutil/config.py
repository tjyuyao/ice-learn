import copy
from functools import wraps

def is_configurable(cls):
    return hasattr(cls, "is_configurable")


def configurable(cls):
    """This wrapper converts ``cls`` to a ``Config`` class which delays the initialization of the original one.
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
        object.__setattr__(self, "_config", Config(cls, self, *args, **kwds))
    cls.__init__ = make_config

    # surrogate the original __getattribute__ function.
    def get_attribute(self, name:str):
        cfg = object.__getattribute__(self, "_config")
        if cfg._frozen:
            value = object.__getattribute__(self, name)
        else:
            value = getattr(cfg, name)
        return value
    cls.__getattribute__ = get_attribute

    # surrogate the original __repr__ function.
    cls.orig__repr__ = getattr(cls, "__repr__", None)
    def repr_func(self):
        cfg = object.__getattribute__(self, "_config")
        if cfg._frozen:
            value = cls.orig__repr__(self)
        else:
            value = repr(cfg)
        return value
    cls.__repr__ = repr_func

    # surrogate the original __getitem__ function.
    cls.orig__getitem__ = getattr(cls, "__getitem__", None)
    def get_item(self, key):
        cfg = object.__getattribute__(self, "_config")
        if cfg._frozen:
            value = cls.orig__getitem__(self, key)
        else:
            value = cfg[key]
        return value
    cls.__getitem__ = get_item

    # surrogate the original __setitem__ function.
    cls.orig__setitem__ = getattr(cls, "__setitem__", None)
    def set_item(self, key, value):
        cfg = object.__getattribute__(self, "_config")
        if cfg._frozen:
            cls.orig__setitem__(self, key, value)
        else:
            cfg[key] = value
    cls.__setitem__ = set_item
    
    # return the modified cls
    return cls


def make_configurable(*classes):
    for cls in classes:
        configurable(cls)


def _clone(obj, deepcopy):
    if isinstance(obj, list):
        obj = [_clone(x, deepcopy=deepcopy) for x in obj]
    elif isinstance(obj, tuple):
        obj = tuple(_clone(x, deepcopy=deepcopy) for x in obj)
    elif isinstance(obj, dict):
        obj = obj.__class__({k: _clone(v, deepcopy=deepcopy) for k, v in obj.items()})
    elif is_configurable(obj):
        obj = obj._config.clone(deepcopy=deepcopy)
    elif deepcopy:
        obj = copy.deepcopy(obj)
    else:
        obj = copy.copy(obj)
    return obj


class Config:
    """Any class converted to ``Config`` instance by ``configurable`` or ``make_configurable`` can store and modify its
    positional and keyword arguments. The real instance of original class will be created only when config is ``freeze()``-d.
    The configuration should be ``clone()``-d for new instances.
    """
    
    def __init__(self, cls, obj, *args, **kwds) -> None:
        self._cls = cls
        self._obj = obj
        self._args = list(args)
        self._kwds = kwds
        self._frozen = False
        
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._args[key]
        elif isinstance(key, str):
            return self._kwds[key]
        else:
            assert False, "Use string key for key word options, integer key for positional arguments."
    
    def __setitem__(self, key, value):
        assert not self._frozen, "Frozen configuration can not be altered, please use clone() at proper time."
        if isinstance(key, int):
            self._args[key] = value
        elif isinstance(key, str):
            self._kwds[key] = value
        else:
            assert False, "Use string key for key word options, integer key for positional arguments."
        
    def __contains__(self, key):
        return key in self._kwds

    def update(self, explicit={}, **implicit):
        explicit.update(implicit)
        for k, v in explicit.items():
            self[k] = v
    
    def clone(self, deepcopy=True):
        return self._cls(*_clone(self._args, deepcopy=deepcopy), 
                          **_clone(self._kwds, deepcopy=deepcopy))

    def freeze(self):
        # recursively freeze arguments
        for arg in self._args:
            if is_configurable(arg):
                arg.freeze()
        for kwd in self._kwds.values():
            if is_configurable(kwd):
                kwd.freeze()
        # mark as frozen so that orig__init__ will be able to manipulate the original instance via __getattribute__.
        self._frozen = True
        # initialize the class instance
        self._cls.orig__init__(self._obj, *self._args, **self._kwds)
        return self._obj

    def __repr__(self):
        args = ', '.join([str(arg) for arg in self._args])
        kwds = ', '.join([f"{k}={repr(v)}" for k, v in self._kwds.items()])
        return f"{self._cls.__name__}({', '.join([args, kwds])})"