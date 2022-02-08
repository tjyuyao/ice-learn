        
import torch

import multiprocessing as python_multiprocessing
from ice.llutil import multiprocessing

torch.multiprocessing = multiprocessing
python_multiprocessing.Queue = multiprocessing.Queue
python_multiprocessing.SimpleQueue = multiprocessing.SimpleQueue

# Register fork handler to initialize OpenMP in child processes (see gh-28389)
from ice.llutil.multiprocessing._atfork import register_after_fork
register_after_fork(torch.get_num_threads)
del register_after_fork

import os
os.environ['PYTHONBREAKPOINT'] = "ice.set_trace"

import dill
dill.settings['byref'] = True

from .api import *