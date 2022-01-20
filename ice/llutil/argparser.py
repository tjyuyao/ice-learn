from __future__ import unicode_literals
import sys

_type = type

def isa(obj, types):
    """alias for python built-in ``isinstance``."""
    return isinstance(obj, types)

def _format_arg(arg):
    farg = str(arg)
    farg = farg.replace("'", '\'"\'"\'')
    for ch in farg:
        if ch in " \"()@&$*!?~{}|#`\\":
            farg = f"'{farg}'"
            break
    return farg

class FlexibleArgParser:

    """A flexible and lightweight argument parser that saves loads of code.
    
    This module works differently compared to python built-in ``argparse`` module.
    - It accepts two types of command line arguments, i.e. positional and keyword based (options).
    - The keyword based arguments (options) should be specified as ``key=value`` or ``key="value"``.
    - The positional arguments can be specified same as ``argparse`` would expect.

    **Example 1:**

    >>> import ice
    >>>
    >>> # same as `python <script>.py 2 workers=4` in shell.
    >>> ice.args.parse_args(["2", "workers=4"])
    >>>
    >>> # get 0-th positional argument, as int, default to 4.
    >>> batchsize = ice.args.get(0, int, 4)  
    >>>
    >>> # get option named "workers", as int, default to 4.
    >>> num_workers = ice.args.get("workers", int, 4)
    >>>
    >>> # Following lines have same effect but when default value is invalid will produce error converting `None` into `int`. You can set default value beforehand use ``ice.args.setdefault()`` to avoid this.
    >>> batchsize = int(ice.args[0])
    >>> num_workers = int(ice.args["workers"])
    >>>
    >>> # Following line also works, but only for keyword arguments, as integer literal is not a legal attribute name.
    >>> num_workers = int(ice.args.workers)

    **Example 2:**

    >>> ice.args.parse_args(["2", "k1=4"])
    >>> assert len(ice.args) == 2
    >>> assert 2 == ice.args.get(0, int, 4)
    >>> assert 4 == ice.args.get("k1", int, 8)
    >>> assert 4 == int(ice.args["k1"])
    >>> assert 4 == int(ice.args.k1)
    >>> ice.args.setdefault("k2", 8)
    >>> 
    >>> assert 8 == int(ice.args.k2)
    >>> 
    >>> ice.args.setdefault("k1", 8)
    >>> assert 4 == int(ice.args.k1)
    >>> 
    >>> del ice.args["k1"]
    >>> assert "k1" not in ice.args
    >>> ice.args.setdefault("k1", 8)
    >>> assert "k1" in ice.args
    >>> assert 8 == int(ice.args.k1)
    >>> 
    >>> ice.args.update(k2=0)
    >>> ice.args.update({0: 0})
    >>> assert 0 == ice.args.get(0, int, 4)
    >>> assert 0 == ice.args.get("k2", int, 4)
    """

    def __init__(self) -> None:
        self.parse_args(sys.argv[1:])
    
    def parse_args(self, argv):
        """Manually parse args.

        Args:
            argv (List[str]): simillar to `sys.argv[1:]`.
        """
        object.__setattr__(self, "_args", {})
        iarg = 0
        for token in argv:
            idsp = token.find("=")
            if -1 != idsp:
                self[token[:idsp]] = token[idsp+1:]
            else:
                self[iarg] = token
                iarg += 1
    
    def __setitem__(self, key, value):
        self._args[key] = value

    def __getitem__(self, key):
        return self._args[key]

    def get(self, key, type=None, value=None):
        """get argument as ``type`` with default ``value``.

        Args:
            key (int|str): ``int`` for positional argument and ``str`` for options.
            type (Type, optional): If specified, the type of argument stored will be casted to ``type``. Default command line arguments are ``str``s.
            value (Any, optional): If key not found, will return ``value``. Defaults to None.

        Returns:
            type: specific argument value.
        """
        if key in self._args:
            value = self._args[key]
            if type is not None and _type(value) != type:
                value = type(value)
        
        return value

    def __getattr__(self, attr):
        if attr == "__name__":
            return self.__class__.__name__
        return self.__getitem__(attr)

    def __setattr__(self, attr, item):
        return self.__setitem__(attr, item)

    def setdefault(self, key, value):
        """Set argument value under `key` as `value`, only if original entry does not exists.

        Args:
            key (int|str): the keyword.
            value: default_value to be set when orginal entry does not exists.

        Returns:
            original or updated value.
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            self._args[key] = value
            return value

    def __repr__(self):
        iargs = {k: _format_arg(v) for k, v in self._args.items() if isa(k, int)}
        iargs = [iargs[key] for key in sorted(iargs.keys())]
        kargs = [f"{k}={_format_arg(v)}" for k, v in self._args.items() if isa(k, str)]
        return f'args: {" ".join(kargs + iargs)}'

    def __len__(self):
        return len(self._args)

    def __delitem__(self, key):
        del self._args[key]

    def __contains__(self, k):
        return k in self._args

    def update(self, *args, **kwargs):
        """simillar to dict.update().
        """
        return self._args.update(*args, **kwargs)


args = FlexibleArgParser()