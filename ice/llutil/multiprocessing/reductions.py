import torch

if torch.__version__ < "1.11.0":
    from .reductions_1_10 import *
else:
    from .reductions_1_11 import *