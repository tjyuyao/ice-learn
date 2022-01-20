"""
usage: waitfinish [-h] PIDS [PIDS ...]

This command just blocks until all processes specified in PIDS exits.

positional arguments:
  PIDS

optional arguments:
  -h, --help  show this help message and exit
"""

import argparse
import os

def _cli():
    parser = argparse.ArgumentParser(description="This command just blocks until all processes specified in PIDS exits.")
    parser.add_argument("PIDS", nargs='+')
    args = parser.parse_args()
    for pid in args.PIDS:
        os.system(f"while ps -p {pid} > /dev/null; do sleep 15; done")
