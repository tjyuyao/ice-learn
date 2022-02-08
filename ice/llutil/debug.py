import inspect
import pdb
import os
import sys

from ice.llutil.launcher.events import global_shared_events

def _set_trace():
    frame=sys._getframe().f_back.f_back
    pdb = SubProcessPdb()
    pdb.set_trace(frame=frame)
    
def set_trace(debug_local_rank=0):
    if "LOCAL_RANK" not in os.environ:  # main process
        _set_trace()
    elif debug_local_rank == int(os.environ["LOCAL_RANK"]):  # target process
        _set_trace()
    else:  # other process
        global_shared_events["debugger_start"].wait()
        global_shared_events["debugger_end"].wait()
        

class SubProcessPdb(pdb.Pdb):
    """Pdb that works from a multiprocessing child"""

    _original_stdin_fd = None if type(sys.stdin).__name__ == "DontReadFromInput" else sys.stdin.fileno()
    _original_stdin = None
    
    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)
    
    # General interaction function
    def _cmdloop(self):
        backup_stdin = sys.stdin
        if "debugger_end" in global_shared_events:
            global_shared_events["debugger_end"].clear()
            global_shared_events["debugger_start"].set()
            
        while True:
            try:
                if not SubProcessPdb._original_stdin:
                    SubProcessPdb._original_stdin = os.fdopen(self._original_stdin_fd)
                sys.stdin = SubProcessPdb._original_stdin
            
                # keyboard interrupts allow for an easy way to cancel
                # the current command, so allow them during interactive input
                self.allow_kbdint = True
                self.cmdloop()
                self.allow_kbdint = False
                break
            except KeyboardInterrupt:
                self.message('--KeyboardInterrupt--')
            finally:
                sys.stdin = backup_stdin
        
        if "debugger_end" in global_shared_events:
            global_shared_events["debugger_end"].set()
            global_shared_events["debugger_start"].clear()