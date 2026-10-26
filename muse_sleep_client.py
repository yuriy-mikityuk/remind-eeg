"""Deprecated: moved to muse_tmr.ble.muse_sleep_client; this top-level name goes in 0.5.

It still works as an import and as a script, with a DeprecationWarning (#232).
"""

import importlib
import sys
import warnings

_NEW_NAME = "muse_tmr.ble.muse_sleep_client"
_MESSAGE = "muse_sleep_client moved to muse_tmr.ble.muse_sleep_client; the old name is removed in 0.5"

if __name__ == "__main__":
    import runpy

    warnings.warn(_MESSAGE, DeprecationWarning)
    runpy.run_module(_NEW_NAME, run_name="__main__")
else:
    warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)
    sys.modules[__name__] = importlib.import_module(_NEW_NAME)
