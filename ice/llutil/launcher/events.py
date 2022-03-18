import multiprocess as mp

class Events:
    """Communicate among main process (agent) and subprocesses (workers)."""
    
    def __init__(self, start_method):
        self.pause = mp.get_context(start_method).Event()
        self.paused = mp.get_context(start_method).Event()
        self.resume = mp.get_context(start_method).Event()
        self.stop_all_tasks = mp.get_context(start_method).Event()
        self.trigger_save_checkpoint = mp.get_context(start_method).Event()
        self.finished_save_checkpoint = mp.get_context(start_method).Event()
        self.debugger_start = mp.get_context(start_method).Event()
        self.debugger_end = mp.get_context(start_method).Event()
        self.progress_bar_iter = mp.get_context(start_method).Value('i', -1)
        self.progress_bar_total = mp.get_context(start_method).Value('i', -1)


global_shared_events = {}