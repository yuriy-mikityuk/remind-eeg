"""Deprecated compatibility module for the original amused public API.

It re-exports the names below from muse_tmr.ble, where the BLE modules moved in #232;
import them from there. It goes in 0.5, with the top-level muse_* shims. `amused` is
also the name of the upstream amused-py package on PyPI, so this file is not in the
wheel and an editable install does not map it either: `import amused` resolves only
with the repository root on sys.path, as under pytest or `python -m` in the checkout.
"""

import warnings

warnings.warn(
    "amused is deprecated: import from muse_tmr.ble instead; the module is removed in 0.5",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "1.1.0"
__author__ = "nexon33 & Claude"

from muse_tmr.ble.muse_stream_client import MuseStreamClient
from muse_tmr.ble.muse_exact_client import MuseExactClient
from muse_tmr.ble.muse_sleep_client import MuseSleepClient
from muse_tmr.ble.muse_raw_stream import MuseRawStream, RawPacket
from muse_tmr.ble.muse_realtime_decoder import MuseRealtimeDecoder, DecodedData
from muse_tmr.ble.muse_replay import MuseReplayPlayer, MuseBinaryParser
from muse_tmr.ble.muse_integrated_parser import MuseIntegratedParser
from muse_tmr.ble.muse_sleep_parser import MuseSleepParser
from muse_tmr.ble.muse_data_parser import MuseDataParser
from muse_tmr.ble.muse_ppg_heart_rate import PPGHeartRateExtractor, HeartRateResult
from muse_tmr.ble.muse_fnirs_processor import FNIRSProcessor, FNIRSData
from muse_tmr.ble.muse_discovery import (
    MuseDevice,
    find_muse_devices,
    select_device,
    connect_to_address,
    quick_connect,
)

__all__ = [
    "MuseStreamClient",
    "MuseExactClient",
    "MuseSleepClient",
    "MuseRawStream",
    "RawPacket",
    "MuseRealtimeDecoder",
    "DecodedData",
    "MuseReplayPlayer",
    "MuseBinaryParser",
    "MuseIntegratedParser",
    "MuseSleepParser",
    "MuseDataParser",
    "PPGHeartRateExtractor",
    "HeartRateResult",
    "FNIRSProcessor",
    "FNIRSData",
    "MuseDevice",
    "find_muse_devices",
    "select_device",
    "connect_to_address",
    "quick_connect",
]


def get_version():
    """Return the current Amused compatibility API version."""
    return __version__
