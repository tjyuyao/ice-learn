import os

from ice.llutil.logger import get_logger


DEBUG_ICE = bool(os.environ.get("DEBUG_ICE", 0))

if DEBUG_ICE:
    get_logger().warn("DEBUG_ICE mode enabled.")
    SHADOW_PATTERNS = (
        'torch/nn/modules/',
        'pickle.py',
        'dill/_dill.py',
        'multiprocess/',
        'torch/distributed/elastic/',
        'ice/llutil/',
        'ice/core/',
        'multiprocessing/',
        'threading.py',
        )
else:
    SHADOW_PATTERNS = ()

import sys

from traceback import extract_tb, format_list, format_exception_only


def _check_file(name):
    if not name: return False
    for pattern in SHADOW_PATTERNS:
        if name.find(pattern) != -1: return False
    return True


def shadow(etype, evalue, tb):
    extracted_tb = extract_tb(tb)
    show = [fs for fs in extracted_tb[:-1] if _check_file(fs.filename)]
    show.append(extracted_tb[-1])
    fmt = format_list(show) + format_exception_only(etype, evalue)
    print(''.join(fmt), end='', file=sys.stderr)


sys.excepthook = shadow


try:
    ipython = get_ipython()

    def shadow_ipython(etype, evalue, stb):
        stb = [s for s in stb if _check_file(s)]
        val = '\n'.join(stb)
        try:
            print(val)
        except UnicodeEncodeError:
            print(val.encode("utf-8", "backslashreplace").decode())
    
    ipython._showtraceback = shadow_ipython
    
except: pass