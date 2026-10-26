import io
import json
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from muse_tmr.audio import (
    AudioBackendFallbackError,
    AudioPlayer,
    AudioCuePlayer,
    AudioPlaybackConfig,
    MockAudioBackend,
    TestCue,
    create_audio_backend,
    resolve_audio_backend,
)
from muse_tmr.audio.audio_player import (
    AudioPlaybackRequest,
    DryRunAudioBackend,
    MacOSAfplayBackend,
    _fade_envelope,
    _write_test_tone,
)
from muse_tmr.cli.main import build_parser, main


class TestAudioCuePlayer(unittest.TestCase):
    def test_volume_cap_is_enforced_before_backend_playback(self):
        backend = MockAudioBackend()
        player = AudioCuePlayer(
            AudioPlaybackConfig(max_volume=0.20, default_volume=0.05),
            backend=backend,
        )

        result = player.play_test_cue(TestCue(duration_seconds=0.01), volume=0.80)

        self.assertTrue(result.played)
        self.assertTrue(result.volume_capped)
        self.assertEqual(result.requested_volume, 0.80)
        self.assertEqual(result.effective_volume, 0.20)
        self.assertEqual(backend.requests[0].effective_volume, 0.20)
        self.assertIn("volume_capped", result.reason_codes)

    def test_fade_and_device_selection_are_passed_to_backend_and_logs(self):
        backend = MockAudioBackend()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audio.jsonl"
            player = AudioCuePlayer(
                AudioPlaybackConfig(
                    max_volume=0.20,
                    fade_in_seconds=0.10,
                    fade_out_seconds=0.20,
                    device_name="Bedroom Headphones",
                    log_path=log_path,
                ),
                backend=backend,
            )

            result = player.play_test_cue(TestCue(duration_seconds=0.01), volume=0.05)
            event = json.loads(log_path.read_text(encoding="utf-8").strip())

        self.assertEqual(result.device_name, "Bedroom Headphones")
        self.assertEqual(backend.requests[0].device_name, "Bedroom Headphones")
        self.assertEqual(backend.requests[0].fade_in_seconds, 0.10)
        self.assertEqual(backend.requests[0].fade_out_seconds, 0.20)
        self.assertIn("device_selected", result.reason_codes)
        self.assertEqual(event["device_name"], "Bedroom Headphones")
        self.assertEqual(event["fade_in_seconds"], 0.10)

    def test_emergency_stop_blocks_future_playback_until_cleared(self):
        backend = MockAudioBackend()
        player = AudioCuePlayer(backend=backend)

        stop_result = player.emergency_stop()
        blocked = player.play_test_cue(TestCue(duration_seconds=0.01))
        player.clear_emergency_stop()
        played = player.play_test_cue(TestCue(duration_seconds=0.01))

        self.assertEqual(stop_result.status, "stopped")
        self.assertEqual(backend.stop_calls, 1)
        self.assertEqual(blocked.status, "blocked")
        self.assertIn("emergency_stop_active", blocked.reason_codes)
        self.assertTrue(played.played)

    def test_dry_run_backend_is_available_without_audio_device(self):
        player = AudioCuePlayer(backend=create_audio_backend("dry-run"))

        result = player.play_test_cue(TestCue(duration_seconds=0.01))

        self.assertTrue(result.played)
        self.assertEqual(result.backend_name, "dry-run")
        self.assertIn("dry_run", result.reason_codes)

    def test_invalid_volume_is_rejected(self):
        player = AudioCuePlayer(backend=MockAudioBackend())

        with self.assertRaises(ValueError):
            player.play_test_cue(volume=1.5)

    def test_audio_player_alias_keeps_legacy_max_volume_constructor(self):
        backend = MockAudioBackend()
        player = AudioPlayer(max_volume=0.10, backend=backend)

        result = player.play_test_cue(TestCue(duration_seconds=0.01), volume=0.20)

        self.assertEqual(result.effective_volume, 0.10)

    def test_playback_result_is_not_rem_gate_or_scheduler_decision(self):
        result = AudioCuePlayer(backend=MockAudioBackend()).play_test_cue(
            TestCue(duration_seconds=0.01)
        )

        self.assertFalse(hasattr(result, "gate_open"))
        self.assertFalse(hasattr(result, "should_play"))


class TestAudioPlaybackValidation(unittest.TestCase):
    def test_config_rejects_out_of_range_volumes_and_negative_fades(self):
        invalid_configs = (
            AudioPlaybackConfig(max_volume=1.5),
            AudioPlaybackConfig(max_volume=-0.1),
            AudioPlaybackConfig(default_volume=1.5),
            AudioPlaybackConfig(fade_in_seconds=-0.1),
            AudioPlaybackConfig(fade_out_seconds=-0.1),
        )
        for config in invalid_configs:
            with self.assertRaises(ValueError):
                config.validate()

    def test_invalid_config_is_rejected_at_player_construction(self):
        with self.assertRaises(ValueError):
            AudioCuePlayer(
                AudioPlaybackConfig(max_volume=1.5),
                backend=MockAudioBackend(),
            )

    def test_test_cue_rejects_invalid_fields(self):
        invalid_cues = (
            TestCue(cue_id=""),
            TestCue(frequency_hz=0.0),
            TestCue(frequency_hz=-440.0),
            TestCue(duration_seconds=0.0),
        )
        for cue in invalid_cues:
            with self.assertRaises(ValueError):
                cue.validate()

    def test_playback_request_to_dict_reports_volume_cap(self):
        request = AudioPlaybackRequest(
            cue_id="test-cue",
            frequency_hz=440.0,
            duration_seconds=0.01,
            requested_volume=0.80,
            effective_volume=0.20,
            max_volume=0.20,
            fade_in_seconds=0.25,
            fade_out_seconds=0.25,
        )

        event = request.to_dict()

        self.assertTrue(request.volume_capped)
        self.assertTrue(event["volume_capped"])
        self.assertEqual(event["requested_volume"], 0.80)
        self.assertEqual(event["effective_volume"], 0.20)


class TestAudioBackendFactory(unittest.TestCase):
    def test_named_backends_are_constructed(self):
        self.assertIsInstance(create_audio_backend("afplay"), MacOSAfplayBackend)
        self.assertIsInstance(create_audio_backend("dry-run"), DryRunAudioBackend)
        self.assertIsInstance(create_audio_backend("mock"), MockAudioBackend)

    def test_unknown_backend_name_is_rejected(self):
        with self.assertRaises(ValueError):
            create_audio_backend("speakers")

    def test_system_backend_falls_back_to_dry_run_without_afplay(self):
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
            backend = create_audio_backend("system")

        self.assertIsInstance(backend, DryRunAudioBackend)

    def test_system_backend_uses_afplay_when_available(self):
        with patch(
            "muse_tmr.audio.audio_player.shutil.which",
            return_value="/usr/bin/afplay",
        ):
            backend = create_audio_backend("system")

        self.assertIsInstance(backend, MacOSAfplayBackend)


class TestMacOSAfplayBackend(unittest.TestCase):
    def test_missing_afplay_skips_playback_instead_of_failing(self):
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
            player = AudioCuePlayer(backend=MacOSAfplayBackend())
            result = player.play_test_cue(TestCue(duration_seconds=0.01))

        self.assertEqual(result.status, "skipped")
        self.assertIn("afplay_unavailable", result.reason_codes)

    @unittest.skipIf(
        sys.platform == "win32",
        "afplay is macOS only, and the backend writes the tone into its open NamedTemporaryFile by name, "
        "which Windows refuses",
    )
    def test_afplay_playback_writes_tone_and_reports_device_limitation(self):
        config = AudioPlaybackConfig(device_name="Bedroom Headphones")
        with patch(
            "muse_tmr.audio.audio_player.shutil.which",
            return_value="/usr/bin/afplay",
        ), patch("muse_tmr.audio.audio_player.subprocess.run") as run_mock:
            player = AudioCuePlayer(config, backend=MacOSAfplayBackend())
            result = player.play_test_cue(TestCue(duration_seconds=0.01))

        self.assertTrue(result.played)
        self.assertIn("system_playback", result.reason_codes)
        self.assertIn("device_selection_unsupported", result.reason_codes)
        self.assertEqual(run_mock.call_args.args[0][0], "afplay")


class TestToneGeneration(unittest.TestCase):
    @staticmethod
    def _request(effective_volume=0.20, fade_seconds=0.05, duration_seconds=0.2):
        return AudioPlaybackRequest(
            cue_id="test-cue",
            frequency_hz=440.0,
            duration_seconds=duration_seconds,
            requested_volume=effective_volume,
            effective_volume=effective_volume,
            max_volume=effective_volume,
            fade_in_seconds=fade_seconds,
            fade_out_seconds=fade_seconds,
        )

    @staticmethod
    def _samples(path):
        with wave.open(path, "rb") as audio:
            raw = audio.readframes(audio.getnframes())
        return [
            int.from_bytes(raw[index : index + 2], byteorder="little", signed=True)
            for index in range(0, len(raw), 2)
        ]

    def test_written_tone_amplitude_never_exceeds_effective_volume(self):
        request = self._request(effective_volume=0.20)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "tone.wav")
            _write_test_tone(path, request)
            samples = self._samples(path)

        self.assertEqual(len(samples), int(44100 * request.duration_seconds))
        amplitude_cap = int(32767 * request.effective_volume)
        self.assertLessEqual(max(abs(sample) for sample in samples), amplitude_cap)
        self.assertGreater(max(abs(sample) for sample in samples), 0)

    def test_fade_in_and_out_keep_edges_quieter_than_middle(self):
        request = self._request(fade_seconds=0.05, duration_seconds=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "tone.wav")
            _write_test_tone(path, request)
            samples = self._samples(path)

        fade_frames = int(44100 * request.fade_in_seconds)
        edge_peak = max(
            max(abs(sample) for sample in samples[: fade_frames // 4]),
            max(abs(sample) for sample in samples[-fade_frames // 4 :]),
        )
        middle = samples[len(samples) // 4 : -len(samples) // 4]
        middle_peak = max(abs(sample) for sample in middle)
        self.assertLess(edge_peak, middle_peak)
        self.assertEqual(samples[0], 0)

    def test_fade_envelope_is_bounded_and_monotonic_at_edges(self):
        frame_count = 100
        fade_frames = 10

        envelope = [
            _fade_envelope(index, frame_count, fade_frames, fade_frames)
            for index in range(frame_count)
        ]

        self.assertEqual(envelope[0], 0.0)
        self.assertEqual(envelope[frame_count // 2], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in envelope))
        self.assertEqual(envelope[:fade_frames], sorted(envelope[:fade_frames]))
        self.assertEqual(
            envelope[-fade_frames:],
            sorted(envelope[-fade_frames:], reverse=True),
        )

    def test_zero_fade_envelope_is_flat(self):
        envelope = [_fade_envelope(index, 10, 0, 0) for index in range(10)]

        self.assertEqual(envelope, [1.0] * 10)


class TestAudioCuePlayerCli(unittest.TestCase):
    def test_play_test_cue_command_parses_safe_audio_options(self):
        args = build_parser().parse_args([
            "play-test-cue",
            "--backend",
            "dry-run",
            "--volume",
            "0.3",
            "--max-volume",
            "0.2",
            "--fade-in-seconds",
            "0.1",
            "--fade-out-seconds",
            "0.2",
            "--device-name",
            "Bedroom Headphones",
        ])

        self.assertEqual(args.command, "play-test-cue")
        self.assertEqual(args.backend, "dry-run")
        self.assertEqual(args.volume, 0.3)
        self.assertEqual(args.max_volume, 0.2)
        self.assertEqual(args.device_name, "Bedroom Headphones")

    def test_play_test_cue_cli_works_with_dry_run_backend(self):
        with redirect_stdout(io.StringIO()) as output:
            exit_code = main([
                "play-test-cue",
                "--backend",
                "dry-run",
                "--duration-seconds",
                "0.01",
                "--volume",
                "0.3",
                "--max-volume",
                "0.2",
            ])

        self.assertEqual(exit_code, 0)
        self.assertIn("status=played", output.getvalue())
        self.assertIn("volume_capped=True", output.getvalue())

    def test_play_test_cue_cli_writes_jsonl_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "playback.jsonl"
            with redirect_stdout(io.StringIO()):
                exit_code = main([
                    "play-test-cue",
                    "--backend",
                    "dry-run",
                    "--duration-seconds",
                    "0.01",
                    "--log-path",
                    str(log_path),
                ])
            event = json.loads(log_path.read_text(encoding="utf-8").strip())

        self.assertEqual(exit_code, 0)
        self.assertEqual(event["status"], "played")
        self.assertEqual(event["backend_name"], "dry-run")


class TestAudioBackendResolution(unittest.TestCase):
    def test_system_and_afplay_resolve_to_afplay_when_it_is_on_path(self):
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value="/usr/bin/afplay"):
            for name in ("system", "afplay"):
                resolution = resolve_audio_backend(name)
                self.assertEqual(resolution.requested_name, name)
                self.assertEqual(resolution.resolved_name, "afplay")
                self.assertIsInstance(resolution.backend, MacOSAfplayBackend)

    def test_missing_afplay_is_refused_and_names_the_fix(self):
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
            for name in ("system", "afplay"):
                with self.assertRaisesRegex(AudioBackendFallbackError, "--allow-dry-run"):
                    resolve_audio_backend(name)

    def test_missing_afplay_falls_back_to_dry_run_only_when_allowed(self):
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
            for name in ("system", "afplay"):
                resolution = resolve_audio_backend(name, allow_dry_run_fallback=True)
                self.assertEqual(resolution.requested_name, name)
                self.assertEqual(resolution.resolved_name, "dry-run")
                self.assertIsInstance(resolution.backend, DryRunAudioBackend)

    def test_silent_backends_need_no_permission(self):
        with patch("muse_tmr.audio.audio_player.shutil.which", return_value=None):
            for name in ("dry-run", "mock"):
                resolution = resolve_audio_backend(name)
                self.assertEqual((resolution.requested_name, resolution.resolved_name), (name, name))


if __name__ == "__main__":
    unittest.main()
