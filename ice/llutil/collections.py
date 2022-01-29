class Dict(dict):
    """access dict values as attributes."""

    def __getattr__(self, key):
        try: return self.__getitem__(key)
        except KeyError: pass
        raise AttributeError(key)

    def __setattr__(self, key: str, value: int) -> None:
        try:
            self.__getattribute__(key)
            return super().__setattr__(key, value)
        except AttributeError:
            return self.__setitem__(key, value)


class Counter(Dict):
    """count values by group.

    **Features:**
    - Get or set values using dictionary or attribute interface.
    - Returns a zero count for missing items instead of raising a KeyError.
    - a `total()` function that sums all values.

    Example:

    >>> import ice
    >>> cnt = ice.Counter()
    >>> assert 0 == cnt['x']
    >>> assert 0 == cnt.x
    >>> cnt.x += 1
    >>> assert 1 == cnt['x']
    >>> assert 1 == cnt.x
    >>> cnt['y'] += 1
    >>> assert 2 == cnt.total()
    """

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            if key.startswith('__'):  # escape hidden values
                raise e
        return 0

    def total(self):
        return sum(self.values())


class ConfigDict(Dict):
    """stores multi-level configurations easily.

    **Features:**
    - Get or set values using dictionary or attribute interface.
    - Create empty dict for intermediate items instead of raising a KeyError.

    Example:

    >>> import ice
    >>> _C = ice.ConfigDict()
    >>> _C.PROPERTY1 = 1
    >>> _C.GROUP1.PROPERTY1 = 2
    """

    def __getitem__(self, key):
        if not self.__contains__(key):
            super().__setitem__(key, ConfigDict())
        return super().__getitem__(key)