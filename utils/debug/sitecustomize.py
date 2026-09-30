"""Debug-only: periodically dump every thread's Python stack to stderr.

Put this directory on PYTHONPATH and set STACK_DUMP_EVERY=<seconds>; Python
imports it at startup and prints all stacks every STACK_DUMP_EVERY seconds.
Used to see where a hanging RoboTwin client is stuck without a debugger.
"""

import faulthandler
import os
import sys

_every = os.environ.get("STACK_DUMP_EVERY")
if _every:
    faulthandler.dump_traceback_later(float(_every), repeat=True, file=sys.stderr)
