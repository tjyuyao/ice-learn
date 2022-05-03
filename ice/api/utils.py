import inspect
import os
import sys

from ice.llutil.launcher.launcher import _parse_devices_and_backend


def parse_devices(devices:str):
    return _parse_devices_and_backend(devices)[0]

# def caller_file():
#     return os.path.abspath(inspect.stack()[1][1])

def extend_sys_path(relpath):
    dirname = os.path.dirname(os.path.abspath(inspect.stack()[1][1]))
    newpath = os.path.abspath(os.path.join(dirname, relpath))
    sys.path.insert(1, newpath)
