import pdb
import os
import sys
from turtle import pd


def set_trace(local_rank=0):
    if "LOCAL_RANK" not in os.environ or local_rank == int(os.environ["LOCAL_RANK"]):
        frame = sys._getframe().f_back  # pop the current stackframe off
        pdb = SubProcessPdb()
        pdb.set_trace(frame=frame)
        

class SubProcessPdb(pdb.Pdb):
    """Pdb that works from a multiprocessing child"""

    _original_stdin_fd = None if type(sys.stdin).__name__ == "DontReadFromInput" else sys.stdin.fileno()
    _original_stdin = None
    
    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)
        
    def _cmdloop(self) -> None:
        backup_stdin = sys.stdin
        try:
            if not SubProcessPdb._original_stdin:
                SubProcessPdb._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = SubProcessPdb._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = backup_stdin