import asyncio
import math
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

from muse_tmr.sources import brainflow_source
from muse_tmr.sources.brainflow_source import (
    BrainFlowDependencyError,
    BrainFlowSource,
    BrainFlowSourceConfig,
    _BrainFlowBackend,
    _last_finite,
    _load_brainflow_backend,
    _row_series,
    _sample_count,
)


class FakeBrainFlowBoard:
    def __init__(self, data_by_preset, *, prepare_delay_seconds=0.0):
        self.data_by_preset = {key: list(value) for key, value in data_by_preset.items()}
        self.prepare_delay_seconds = prepare_delay_seconds
        self.prepared = False
        self.started = False
        self.stopped = False
        self.released = False

    def prepare_session(self):
        if self.prepare_delay_seconds:
            time.sleep(self.prepare_delay_seconds)
        self.prepared = True

    def start_stream(self, buffer_size=450000, streamer_params=""):
        self.started = True
        self.buffer_size = buffer_size
        self.streamer_params = streamer_params

    def get_board_data(self, max_samples, preset):
        batches = self.data_by_preset.get(preset, [])
        if not batches:
            return []
        return batches.pop(0)

    def stop_stream(self):
        self.stopped = True

    def release_session(self):
        self.released = True


class FakeBrainFlowBackend:
    name = "fake-brainflow"
    DEFAULT_PRESET = 0
    AUXILIARY_PRESET = 1
    ANCILLARY_PRESET = 2

    def __init__(self, board):
        self.board = board
        self.params = None

    def board_id_value(self, name):
        self.board_name = name
        return 9001

    def preset_value(self, name):
        return getattr(self, name)

    def input_params(self):
        self.params = SimpleNamespace()
        return self.params

    def board_shim(self, board_id, params):
        self.board_id = board_id
        self.params = params
        return self.board

    def eeg_channels(self, board_id, preset):
        return (1, 2, 3, 4)

    def eeg_names(self, board_id, preset):
        return ("TP9", "AF7", "AF8", "TP10")

    def other_channels(self, board_id, preset):
        return (5,)

    def accel_channels(self, board_id, preset):
        return (1, 2, 3)

    def gyro_channels(self, board_id, preset):
        return (4, 5, 6)

    def optical_channels(self, board_id, preset):
        return (1, 2)

    def battery_channel(self, board_id, preset):
        return 3

    def timestamp_channel(self, board_id, preset):
        return 0


class TestBrainFlowSource(unittest.IsolatedAsyncioTestCase):
    async def test_import_and_instantiation_do_not_require_brainflow_dependency(self):
        source = BrainFlowSource()

        self.assertEqual(source.source_name, "brainflow")
        self.assertEqual(source.strategy, "optional-brainflow")

    async def test_discover_returns_configured_brainflow_pseudo_device(self):
        backend = FakeBrainFlowBackend(FakeBrainFlowBoard({}))
        source = BrainFlowSource(
            BrainFlowSourceConfig(address="AA:BB", preset="p1041"),
            brainflow_backend=backend,
        )

        devices = await source.discover()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].name, "BrainFlow Muse S Athena (MUSE_S_ATHENA_BOARD)")
        self.assertEqual(devices[0].address, "AA:BB")
        self.assertEqual(devices[0].metadata["source"], "brainflow")
        self.assertEqual(devices[0].metadata["preset"], "p1041")

    async def test_connect_and_stream_maps_brainflow_batches_to_muse_frames(self):
        board = FakeBrainFlowBoard(
            {
                0: [
                    [
                        [1000.0, 1000.1],
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                        [9.0, 10.0],
                    ]
                ],
                1: [
                    [
                        [1001.0, 1001.1],
                        [0.1, 0.2],
                        [0.3, 0.4],
                        [0.5, 0.6],
                        [1.1, 1.2],
                        [1.3, 1.4],
                        [1.5, 1.6],
                    ]
                ],
                2: [
                    [
                        [1002.0, 1002.1],
                        [11.0, 12.0],
                        [13.0, 14.0],
                        [88.0, 89.0],
                    ]
                ],
            }
        )
        backend = FakeBrainFlowBackend(board)
        source = BrainFlowSource(
            BrainFlowSourceConfig(
                address="AA:BB",
                serial_number="Muse-Test",
                duration_seconds=0.0,
                max_chunk_samples=2,
                session_cooldown_seconds=0.0,
            ),
            brainflow_backend=backend,
        )

        metadata = await source.connect()
        stream = source.stream().__aiter__()
        eeg_frame = await stream.__anext__()
        imu_frame = await stream.__anext__()
        ppg_frame = await stream.__anext__()
        await source.stop()

        self.assertEqual(metadata.source_name, "brainflow")
        self.assertEqual(metadata.device_id, "AA:BB")
        self.assertEqual(metadata.metadata["preset"], "p1041")
        self.assertEqual(backend.params.mac_address, "AA:BB")
        self.assertEqual(backend.params.serial_number, "Muse-Test")
        self.assertEqual(backend.params.timeout, 20)
        self.assertEqual(backend.params.other_info, "preset=p1041;low_latency=true")

        self.assertEqual(eeg_frame.timestamp, 1000.1)
        self.assertEqual(eeg_frame.source, "brainflow")
        self.assertEqual(eeg_frame.eeg.channels_uv["TP9"], (1.0, 2.0))
        self.assertEqual(eeg_frame.eeg.channels_uv["TP10"], (7.0, 8.0))
        self.assertEqual(eeg_frame.eeg.channels_uv["OTHER_0"], (9.0, 10.0))

        self.assertEqual(imu_frame.imu.accelerometer_g[1]["z"], 0.6)
        self.assertEqual(imu_frame.imu.gyroscope_dps[1]["z"], 1.6)

        self.assertEqual(ppg_frame.ppg.channels["OPTICAL_0"], (11.0, 12.0))
        self.assertEqual(ppg_frame.ppg.channels["OPTICAL_1"], (13.0, 14.0))
        self.assertEqual(ppg_frame.battery.percent, 89.0)

        self.assertTrue(board.prepared)
        self.assertTrue(board.started)
        self.assertTrue(board.stopped)
        self.assertTrue(board.released)

    async def test_connect_timeout_fails_without_hanging_event_loop(self):
        board = FakeBrainFlowBoard({}, prepare_delay_seconds=0.2)
        backend = FakeBrainFlowBackend(board)
        source = BrainFlowSource(
            BrainFlowSourceConfig(
                connect_timeout_seconds=0.01,
                session_cooldown_seconds=0.0,
            ),
            brainflow_backend=backend,
        )

        with self.assertRaisesRegex(RuntimeError, "prepare_session timed out"):
            await source.connect()

        self.assertEqual(source.disconnect_reason, "connect_timeout")

    async def _sleeps_during_stop(self, cooldown_seconds):
        source = BrainFlowSource(
            BrainFlowSourceConfig(session_cooldown_seconds=cooldown_seconds),
            brainflow_backend=FakeBrainFlowBackend(FakeBrainFlowBoard({})),
        )
        await source.connect()

        # Record the delays instead of timing stop() with the wall clock: on
        # Windows time.monotonic() ticks at about 15.6 ms. The stop/release
        # polling still yields to the loop so its worker threads can finish.
        real_sleep = asyncio.sleep
        delays = []

        async def recording_sleep(delay, *args, **kwargs):
            delays.append(delay)
            await real_sleep(0)

        with patch.object(brainflow_source.asyncio, "sleep", recording_sleep):
            await source.stop()
        return delays

    async def test_stop_applies_configured_session_cooldown(self):
        delays = await self._sleeps_during_stop(30.0)

        self.assertEqual(delays.count(30.0), 1)
        self.assertEqual(delays[-1], 30.0)

    async def test_stop_skips_sleep_without_session_cooldown(self):
        delays = await self._sleeps_during_stop(0.0)

        self.assertNotIn(0.0, delays)


class TestBrainFlowSourceConfigValidation(unittest.TestCase):
    def test_config_rejects_invalid_values(self):
        invalid_kwargs = (
            {"board_name": ""},
            {"duration_seconds": -1.0},
            {"poll_interval_seconds": 0.0},
            {"max_chunk_samples": 0},
            {"connect_timeout_seconds": 0.0},
            {"stream_start_timeout_seconds": 0.0},
            {"stop_timeout_seconds": 0.0},
            {"session_cooldown_seconds": -1.0},
        )
        for kwargs in invalid_kwargs:
            with self.assertRaises(ValueError):
                BrainFlowSourceConfig(**kwargs)


class TestBrainFlowSourceErrorPaths(unittest.IsolatedAsyncioTestCase):
    async def test_discover_respects_name_filter(self):
        backend = FakeBrainFlowBackend(FakeBrainFlowBoard({}))
        source = BrainFlowSource(
            BrainFlowSourceConfig(name_filter="OtherHeadset"),
            brainflow_backend=backend,
        )

        devices = await source.discover()

        self.assertEqual(devices, ())

    async def test_connect_prefers_explicit_device_address(self):
        from muse_tmr.sources.base_source import MuseDeviceInfo

        backend = FakeBrainFlowBackend(FakeBrainFlowBoard({}))
        source = BrainFlowSource(
            BrainFlowSourceConfig(address="AA:BB", session_cooldown_seconds=0.0),
            brainflow_backend=backend,
        )
        device = MuseDeviceInfo(name="Muse", address="CC:DD", rssi=0)

        metadata = await source.connect(device)

        self.assertEqual(backend.params.mac_address, "CC:DD")
        self.assertEqual(metadata.device_id, "CC:DD")

    async def test_stream_start_timeout_sets_disconnect_reason(self):
        class SlowStartBoard(FakeBrainFlowBoard):
            def start_stream(self, buffer_size=450000, streamer_params=""):
                time.sleep(0.2)
                super().start_stream(buffer_size, streamer_params)

        board = SlowStartBoard({})
        backend = FakeBrainFlowBackend(board)
        source = BrainFlowSource(
            BrainFlowSourceConfig(
                stream_start_timeout_seconds=0.01,
                session_cooldown_seconds=0.0,
            ),
            brainflow_backend=backend,
        )
        await source.connect()

        with self.assertRaisesRegex(RuntimeError, "start_stream timed out"):
            async for _ in source.stream():
                pass

        self.assertEqual(source.disconnect_reason, "stream_start_timeout")

    async def test_stop_stream_timeout_still_releases_session(self):
        class SlowStopBoard(FakeBrainFlowBoard):
            def stop_stream(self):
                time.sleep(0.2)
                super().stop_stream()

        board = SlowStopBoard({})
        backend = FakeBrainFlowBackend(board)
        source = BrainFlowSource(
            BrainFlowSourceConfig(
                stop_timeout_seconds=0.01,
                session_cooldown_seconds=0.0,
            ),
            brainflow_backend=backend,
        )
        await source.connect()
        source._streaming = True

        await source.stop()

        self.assertEqual(source.disconnect_reason, "stop_timeout")
        self.assertTrue(board.released)
        self.assertIsNone(source._board)

    async def test_diagnostics_reports_counts_and_disconnect_reason(self):
        backend = FakeBrainFlowBackend(FakeBrainFlowBoard({}))
        source = BrainFlowSource(
            BrainFlowSourceConfig(session_cooldown_seconds=0.0),
            brainflow_backend=backend,
        )

        diagnostics = source.diagnostics()

        self.assertEqual(diagnostics["source"], "brainflow")
        self.assertEqual(diagnostics["frame_count"], 0)
        self.assertIsNone(diagnostics["last_poll_age_seconds"])
        self.assertIsNone(diagnostics["disconnect_reason"])


class _FakeEnum:
    def __init__(self, value):
        self.value = value


class _FakeBoardShim:
    @staticmethod
    def get_eeg_channels(board_id, preset):
        return [1, 2]

    @staticmethod
    def get_eeg_names(board_id, preset):
        raise RuntimeError("names unavailable")

    @staticmethod
    def get_other_channels(board_id, preset):
        raise RuntimeError("no other channels")

    @staticmethod
    def get_accel_channels(board_id, preset):
        return [3, 4, 5]

    @staticmethod
    def get_gyro_channels(board_id, preset):
        return [6, 7, 8]

    @staticmethod
    def get_optical_channels(board_id, preset):
        raise RuntimeError("no optical channels")

    @staticmethod
    def get_ppg_channels(board_id, preset):
        return [9, 10]

    @staticmethod
    def get_battery_channel(board_id, preset):
        raise RuntimeError("no battery channel")

    @staticmethod
    def get_timestamp_channel(board_id, preset):
        return 0


class TestBrainFlowBackendWrapper(unittest.TestCase):
    def setUp(self):
        module = SimpleNamespace(
            BoardIds=SimpleNamespace(MUSE_S_ATHENA_BOARD=_FakeEnum(38)),
            BrainFlowPresets=SimpleNamespace(DEFAULT_PRESET=_FakeEnum(0)),
            BrainFlowInputParams=SimpleNamespace,
            BoardShim=_FakeBoardShim,
        )
        self.backend = _BrainFlowBackend(module)

    def test_enum_values_are_unwrapped(self):
        self.assertEqual(self.backend.board_id_value("MUSE_S_ATHENA_BOARD"), 38)
        self.assertEqual(self.backend.preset_value("DEFAULT_PRESET"), 0)

    def test_channel_lookups_degrade_to_empty_or_none_on_backend_errors(self):
        self.assertEqual(self.backend.eeg_channels(38, 0), (1, 2))
        self.assertEqual(self.backend.eeg_names(38, 0), ())
        self.assertEqual(self.backend.other_channels(38, 0), ())
        self.assertIsNone(self.backend.battery_channel(38, 0))
        self.assertEqual(self.backend.timestamp_channel(38, 0), 0)

    def test_optical_channels_fall_back_to_ppg_channels(self):
        self.assertEqual(self.backend.optical_channels(38, 0), (9, 10))

    def test_missing_brainflow_dependency_raises_actionable_error(self):
        with patch(
            "muse_tmr.sources.brainflow_source.importlib.import_module",
            side_effect=ImportError("no brainflow"),
        ):
            with self.assertRaisesRegex(BrainFlowDependencyError, "brainflow"):
                _load_brainflow_backend()


class TestBrainFlowDataHelpers(unittest.TestCase):
    def test_sample_count_handles_lists_scalars_and_empty_data(self):
        self.assertEqual(_sample_count([[1.0, 2.0], [3.0, 4.0]]), 2)
        self.assertEqual(_sample_count([]), 0)
        self.assertEqual(_sample_count([None]), 0)
        self.assertEqual(_sample_count([5.0]), 0)
        self.assertEqual(_sample_count(5.0), 0)

    def test_row_series_handles_missing_rows_and_scalars(self):
        self.assertEqual(_row_series([[1.0, 2.0]], -1), ())
        self.assertEqual(_row_series([[1.0, 2.0]], 5), ())
        self.assertEqual(_row_series([5.0], 0), (5.0,))
        self.assertEqual(_row_series([[1.0, 2.0]], 0), (1.0, 2.0))

    def test_last_finite_skips_nan_and_handles_all_nan(self):
        self.assertEqual(_last_finite([1.0, 2.0, math.nan]), 2.0)
        self.assertEqual(_last_finite([math.nan, 3.0]), 3.0)
        self.assertIsNone(_last_finite([math.nan, math.inf]))
        self.assertIsNone(_last_finite([]))


if __name__ == "__main__":
    unittest.main()
