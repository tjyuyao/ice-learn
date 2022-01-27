"""logging utilities."""

from typing import Optional
from torch.distributed.elastic.utils.logging import _setup_logger, _derive_module_name


def get_logger(name: Optional[str] = None):
    """set up a simple logger that writes into stderr. 
    
    The loglevel is fetched from the LOGLEVEL
    env. variable or WARNING as default. The function will use the
    module name of the caller if no name is provided.

    Args:
        name: Name of the logger. If no name provided, the name will
              be derived from the call stack.
    """
    
    # Derive the name of the caller, if none provided
    # Use depth=2 since this function takes up one level in the call stack
    return _setup_logger(name or _derive_module_name(depth=2))