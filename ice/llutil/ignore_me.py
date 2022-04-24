class IgnoreMe:
    
    def __init__(self, *args, **kwds): ...
    
    def __call__(self, *args, **kwds): return self
    
    def __getattr__(self, __name: str): return self

    def __bool__(self): return False

    def __nonzero__(self): return self.__bool__()