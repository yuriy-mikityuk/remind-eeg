"""Compatibility module for the original amused public API.

The repository uses top-level modules today. This shim makes `import amused`
work from a checkout and from editable installs until the project has a
package-layout migration.
"""

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
