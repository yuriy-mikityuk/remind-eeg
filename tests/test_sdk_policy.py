import importlib.util
import unittest
from pathlib import Path


def _load_policy_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_forbidden_files.py"
    spec = importlib.util.spec_from_file_location("check_forbidden_files", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSdkPolicy(unittest.TestCase):
    def test_forbidden_patterns_catch_sdk_paths_and_binaries(self):
        policy = _load_policy_script()

        self.assertTrue(policy.is_forbidden("sdk/include/Muse.h"))
        self.assertTrue(policy.is_forbidden("vendor/muse_sdk/lib/libmuse.dylib"))
        self.assertTrue(policy.is_forbidden("vendor/muse-sdk/Muse.framework/Muse"))
        self.assertTrue(policy.is_forbidden("MuseSDK.a"))
        self.assertTrue(policy.is_forbidden("MuseSDK.aar"))
        self.assertTrue(policy.is_forbidden("MuseSDK.pkg"))
        self.assertTrue(policy.is_forbidden("Muse-SDK.zip"))

    def test_generated_synthetic_nights_are_not_publishable(self):
        policy = _load_policy_script()

        self.assertTrue(policy.is_forbidden("data/synthetic/demo_20261025_120000/report.html"))
        self.assertTrue(policy.is_forbidden("data/synthetic/clean_night/hypnogram.csv"))
        self.assertFalse(policy.is_forbidden("tests/fixtures/synthetic/agreement_baseline.json"))

    def test_sdk_stub_source_file_is_allowed(self):
        policy = _load_policy_script()

        self.assertFalse(policy.is_forbidden("src/muse_tmr/sources/muse_sdk_source_stub.py"))
        self.assertFalse(policy.is_forbidden("docs/sdk_policy.md"))


if __name__ == "__main__":
    unittest.main()
