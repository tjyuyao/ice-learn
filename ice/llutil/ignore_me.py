class IgnoreMe:
    
    def __init__(self, *args, **kwds): ...
    
    def __call__(self, *args, **kwds): return self
    
    def __getattr__(self, __name: str): return self