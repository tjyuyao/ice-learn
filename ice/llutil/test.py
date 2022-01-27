"""helps developers of ice to test."""

import torch
import pytest

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")

def requires_n_gpus(n): return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"need {n} gpus")
