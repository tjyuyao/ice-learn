"""
This module provides helper functions for commonly used argument processing for functions, 
and a FlexibleArgParser for command line argument parsing. The default singleton of this
argument parser is accessable via ``ice.args``.
"""

import sys

_type = type

def isa(obj, types):
    """Helper function: alias for python built-in ``isinstance``."""
    return isinstance(obj, types)

def as_list(maybe_element):
    """Helper function: regularize input into list of element.

    No matter what is input, will output a list for your iteration.
    
    **Basic Examples:**

    >>> assert as_list("string") == ["string"]
    >>> assert as_list(["string", "string"]) == ["string", "string"]
    >>> assert as_list(("string", "string")) == ["string", "string"]
    >>> assert as_list([["string", "string"]]) == ["string", "string"]
    
    **An Application Example:**

    >>> def func(*args):
    >>>     return as_list(args)
    >>>
    >>> assert func("a", "b") == ["a", "b"]
    >>> assert func(["a", "b"]) == ["a", "b"]

    """
    if type(maybe_element) == list:
        maybe_stacked = maybe_element
        if 1 == len(maybe_stacked) and type(maybe_stacked[0]) == list:
            mustbe_list = maybe_stacked[0]
        else:
            mustbe_list = maybe_stacked
    elif type(maybe_element) == tuple:
        maybe_stacked = maybe_element
        if 1 == len(maybe_stacked) and type(maybe_stacked[0]) == list:
            mustbe_list = maybe_stacked[0]
        else:
            mustbe_list = list(maybe_stacked)
    else:
        mustbe_list = [maybe_element]
    return mustbe_list


def as_dict(maybe_element, key):
    """Helper function: regularize input into a dict.

    if ``maybe_element`` is not a dict, will return a dict with single
    key as ``{key:maybe_element}``, else will return ``maybe_element``.

    Args:
        maybe_element: a dict or any object.
        key : the sole key.

    Returns:
        dict: ensures to be a dict.

    Example:

    >>> assert as_dict({"k": "v"}, "k") == {"k": "v"}
    >>> assert as_dict("v", "k") == {"k": "v"}
    """
    if isinstance(maybe_element, dict):
        mustbe_dict = maybe_element
    else:
        mustbe_dict = {key:maybe_element}
    return mustbe_dict


def _format_arg(arg):
    farg = str(arg)
    farg = farg.replace("'", '\'"\'"\'')
    for ch in farg:
        if ch in " \"()@&$*!?~{}|#`\\":
            farg = f"'{farg}'"
            break
    return farg


class ArgumentMissingError(Exception): """Raised when a required argument is missing from command line."""

class ArgumentTypeError(Exception): """Raised when converting an argument failed."""


class FlexibleArgParser:

    """A flexible and lightweight argument parser that saves loads of code.
    
    This module works differently compared to python built-in ``argparse`` module.
    - It accepts two types of command line arguments, i.e. positional and keyword based (options).
    - The keyword based arguments (options) should be specified as ``key=value`` or ``key="value"``.
    - The positional arguments is indexed directly using an integer, but this feature is not recommended.

    Example:

    >>> import ice
    >>>
    >>> # same as `python <script>.py 2 k1=4` in shell.
    >>> ice.args.parse_args(["2", "k1=4"])
    >>>
    >>> ice.args.setdefault("k1", 8, int)  # This line is optional for this example; setdefault() generally is optional.
    >>> ice.args.setdefault("k2", 8)
    >>>
    >>> assert len(ice.args) == 3
    >>> assert 2 == int(ice.args[0])  # default type is str.
    >>> assert 4 == ice.args["k1"]  # as setdefault specified a type, here a conversion is not needed.
    >>> assert 4 == ice.args.k1  # attribute also works.
    >>> assert 8 == ice.args.k2  # use default value.
    >>> 
    >>> ice.args["k1"] = 1
    >>> ice.args.k3 = 1
    >>> ice.args.update(k2=0)
    >>> ice.args.update({0: -1})
    >>> assert -1 == ice.args[0]
    >>> assert  1 == ice.args["k3"]
    >>> assert  0 == ice.args.k2

    Note:
        If you manually call `parse_args()`, call it before `setdefault()`.

    """

    REQUIRED = "ICE.FLEXIBLE_ARGS_PARSER.REQUIRED_ARGUMENT"

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

    def __getattr__(self, attr):
        if attr == "__name__":
            return self.__class__.__name__
        return self.__getitem__(attr)

    def __setattr__(self, attr, item):
        return self.__setitem__(attr, item)

    def setdefault(self, key, default, type=None, help=""):
        """Set argument value under `key` as `value`, only if original entry does not exists.

        Args:
            key (int|str): the keyword.
            value: default_value to be set when orginal entry does not exists.

        Returns:
            original or updated value.
        """
        if default is REQUIRED:
            raise ArgumentTypeError(key, help)
        elif key in self:
            try: self[key] = type(self[key])
            except Exception: raise ArgumentTypeError(key, default, help)
        else:
            self[key] = default

    def __repr__(self):
        iargs = {k: _format_arg(v) for k, v in self._args.items() if isa(k, int)}
        iargs = [iargs[key] for key in sorted(iargs.keys())]
        kargs = [f"{k}={_format_arg(v)}" for k, v in self._args.items() if isa(k, str)]
        return f'args: {" ".join(kargs + iargs)}'

    def __len__(self):
        return len(self._args)

    def __delitem__(self, key):
        del self._args[key]

    def __contains__(self, key):
        return key in self._args

    def update(self, __dict={}, **kwds):
        """simillar to dict.update()."""
        __dict.update(kwds)
        for k, v in __dict.items():
            self[k] = v


args = FlexibleArgParser()

REQUIRED = args.REQUIRED