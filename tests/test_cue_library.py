import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from muse_tmr.audio import (
    CueLibrary,
    CueMetadata,
    default_cue_library,
    load_cue_library,
    validate_cue_library_file,
)
from muse_tmr.cli.main import main


class TestCueLibrary(unittest.TestCase):
    def test_default_library_validates_without_audio_files(self):
        library = default_cue_library()
        report = library.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual({cue.protocol for cue in library.cues}, {"puzzle", "tlr", "generic", "sham"})
        self.assertEqual(library.by_id("puzzle_soft_tone").cue_type, "generated_tone")

    def test_the_starter_library_has_an_untrained_sham_tone(self):
        from muse_tmr.audio import DEFAULT_SHAM_CUE_ID
        from muse_tmr.protocol import ensure_sham_untrained, validate_sham_cue

        library = default_cue_library()
        sham = library.by_id(DEFAULT_SHAM_CUE_ID)
        self.assertEqual((sham.cue_id, sham.protocol, sham.cue_type), ("sham_soft_tone", "sham", "generated_tone"))
        self.assertIn("untrained", sham.tags)
        self.assertNotIn(sham.frequency_hz, {cue.frequency_hz for cue in library.cues if cue.cue_id != sham.cue_id})
        self.assertEqual(library.filter(protocol="sham"), (sham,))
        ensure_sham_untrained(DEFAULT_SHAM_CUE_ID, cue_library=library, tlr_cue_ids=("tlr_soft_tone",))
        validate_sham_cue(DEFAULT_SHAM_CUE_ID, catalog=None, cue_library=library)

    def test_duplicate_cue_ids_are_blocking_validation_errors(self):
        cue = CueMetadata(
            cue_id="duplicate",
            cue_type="silence",
            duration_seconds=1.0,
        )
        report = CueLibrary(cues=(cue, cue)).validate()

        self.assertFalse(report.is_valid)
        self.assertIn("duplicate_cue_id", [issue.reason_code for issue in report.issues])

    def test_sound_cue_missing_file_is_detected_pre_session(self):
        library = CueLibrary(cues=(
            CueMetadata(
                cue_id="private_audio",
                cue_type="sound",
                protocol="puzzle",
                path="private/missing.wav",
                duration_seconds=1.0,
                tags=("puzzle", "private"),
            ),
        ))

        with tempfile.TemporaryDirectory() as tmp:
            report = library.validate(base_dir=Path(tmp))

        self.assertFalse(report.is_valid)
        self.assertEqual(report.blocking_issues[0].reason_code, "cue_file_missing")

    def test_existing_sound_file_passes_file_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cue_path = tmp_path / "private" / "cue.wav"
            cue_path.parent.mkdir()
            cue_path.write_bytes(b"fixture")
            library = CueLibrary(cues=(
                CueMetadata(
                    cue_id="private_audio",
                    cue_type="sound",
                    protocol="tlr",
                    path="private/cue.wav",
                    duration_seconds=1.0,
                    tags=("tlr",),
                ),
            ))

            report = library.validate(base_dir=tmp_path)

        self.assertTrue(report.is_valid)

    def test_json_roundtrip_preserves_metadata_and_filters(self):
        library = default_cue_library()
        with tempfile.TemporaryDirectory() as tmp:
            path = library.save(Path(tmp) / "cues.json")
            loaded = load_cue_library(path)

        tlr = loaded.filter(protocol="tlr")
        generated = loaded.filter(tag="generated")

        self.assertEqual(loaded.library_id, "starter")
        self.assertEqual(len(tlr), 1)
        self.assertEqual(tlr[0].cue_id, "tlr_soft_tone")
        self.assertEqual(len(generated), 3)

    def test_invalid_generated_tone_requires_frequency(self):
        library = CueLibrary(cues=(
            CueMetadata(
                cue_id="broken_generated",
                cue_type="generated_tone",
                duration_seconds=1.0,
            ),
        ))

        report = library.validate()

        self.assertFalse(report.is_valid)
        self.assertIn("generated_frequency_missing", [issue.reason_code for issue in report.issues])

    def test_validate_cue_library_file_uses_catalog_parent_for_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library = CueLibrary(cues=(
                CueMetadata(
                    cue_id="relative_audio",
                    cue_type="sound",
                    path="private/cue.wav",
                    duration_seconds=1.0,
                ),
            ))
            catalog_path = library.save(tmp_path / "library.json")

            missing_report = validate_cue_library_file(catalog_path)
            (tmp_path / "private").mkdir()
            (tmp_path / "private" / "cue.wav").write_bytes(b"fixture")
            valid_report = validate_cue_library_file(catalog_path)

        self.assertFalse(missing_report.is_valid)
        self.assertTrue(valid_report.is_valid)


class TestCueLibraryCli(unittest.TestCase):
    def test_create_validate_and_list_cue_library_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_path = Path(tmp) / "starter.json"
            with redirect_stdout(io.StringIO()):
                create_exit = main([
                    "create-cue-library",
                    "--output",
                    str(library_path),
                ])
            with redirect_stdout(io.StringIO()) as validate_output:
                validate_exit = main(["validate-cue-library", str(library_path)])
            with redirect_stdout(io.StringIO()) as list_output:
                list_exit = main([
                    "list-cues",
                    str(library_path),
                    "--protocol",
                    "puzzle",
                ])

        report = json.loads(validate_output.getvalue())
        self.assertEqual(create_exit, 0)
        self.assertEqual(validate_exit, 0)
        self.assertEqual(list_exit, 0)
        self.assertTrue(report["is_valid"])
        self.assertIn("puzzle_soft_tone", list_output.getvalue())
        self.assertNotIn("tlr_soft_tone", list_output.getvalue())

    def test_validate_cue_library_cli_fails_on_missing_sound_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_path = CueLibrary(cues=(
                CueMetadata(
                    cue_id="missing",
                    cue_type="sound",
                    path="private/missing.wav",
                    duration_seconds=1.0,
                ),
            )).save(Path(tmp) / "library.json")
            with redirect_stdout(io.StringIO()) as output:
                exit_code = main(["validate-cue-library", str(library_path)])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["is_valid"])
        self.assertEqual(report["issues"][0]["reason_code"], "cue_file_missing")

    def test_validate_cue_library_cli_can_skip_file_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_path = CueLibrary(cues=(
                CueMetadata(
                    cue_id="missing",
                    cue_type="sound",
                    path="private/missing.wav",
                    duration_seconds=1.0,
                ),
            )).save(Path(tmp) / "library.json")
            with redirect_stdout(io.StringIO()) as output:
                exit_code = main([
                    "validate-cue-library",
                    str(library_path),
                    "--skip-file-check",
                ])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["is_valid"])


if __name__ == "__main__":
    unittest.main()
