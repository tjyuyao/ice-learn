import inspect

def in_main_process():
    """Whether current process is worker process or main process.
    """
    for frameinfo in inspect.stack():
        if "spawn_main" == frameinfo.function:
            return False
    return True


_AUTO_FREEZE_SWITCH = False

def enable_auto_freeze(enable:bool = True):
    global _AUTO_FREEZE_SWITCH
    _AUTO_FREEZE_SWITCH = enable

def auto_freeze_enabled():
    global _AUTO_FREEZE_SWITCH
    if _AUTO_FREEZE_SWITCH:
        return True
    elif not in_main_process():
        enable_auto_freeze()
        return True
    else:
        return False


def init_torch_multiprocessing():
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