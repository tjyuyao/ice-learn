"""logging utilities."""
import inspect
import logging
import os
import warnings
from typing import Optional


# from torch.distributed.elastic.utils.logging import _derive_module_name
def _derive_module_name(depth: int = 1) -> Optional[str]:
    """
    Derives the name of the caller module from the stack frames.

    Args:
        depth: The position of the frame in the stack.
    """
    try:
        stack = inspect.stack()
        assert depth < len(stack)
        # FrameInfo is just a named tuple: (frame, filename, lineno, function, code_context, index)
        frame_info = stack[depth]

        module = inspect.getmodule(frame_info[0])
        if module:
            module_name = module.__name__
        else:
            # inspect.getmodule(frame_info[0]) does NOT work (returns None) in
            # binaries built with @mode/opt
            # return the filename (minus the .py extension) as modulename
            filename = frame_info[1]
            module_name = os.path.splitext(os.path.basename(filename))[0]
        return module_name
    except Exception as e:
        warnings.warn(
            f"Error deriving logger module name, using <None>. Exception: {e}",
            RuntimeWarning,
        )
        return None


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
    log = logging.getLogger(name or _derive_module_name(depth=2))
    log.setLevel(logging.INFO)
    
    return log
