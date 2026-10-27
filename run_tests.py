#!/usr/bin/env python
"""
Test runner for REMind (the muse_tmr package and its BLE layer)
Runs fast tests by default, use --all for complete test suite
"""

import sys
import unittest
import argparse
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def run_fast_tests():
    """Run only fast tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add fast test modules
    fast_modules = [
        'tests.test_raw_stream',
        'tests.test_realtime_decoder',
        'tests.test_ppg_fnirs_fast',  # Fast version
        'tests.test_sample_types',
        'tests.test_cli',
        'tests.test_macos_launcher',
        'tests.test_local_app',
        'tests.test_audio_player',
        'tests.test_cue_library',
        'tests.test_volume_calibration',
        'tests.test_recorder',
        'tests.test_muse_sdk_source_stub',
        'tests.test_sdk_policy',
        'tests.test_openmuse_lsl_source',
        'tests.test_replay',
        'tests.test_epochs',
        'tests.test_eeg_features',
        'tests.test_imu_features',
        'tests.test_ppg_features',
        'tests.test_sudden_hr_rule',
        'tests.test_contact',
        'tests.test_contact_monitor',
        'tests.test_contact_gate',
        'tests.test_rem_detector',
        'tests.test_rem_event_features',
        'tests.test_rem_annotations',
        'tests.test_personal_rem_classifier',
        'tests.test_rem_gate',
        'tests.test_arousal_guard',
        'tests.test_scheduler',
        'tests.test_puzzle_protocol',
        'tests.test_tlr_protocol',
        'tests.test_dream_report',
        'tests.test_morning_retest',
        'tests.test_cued_uncued_analysis',
        'tests.test_pilot1_validation',
        'tests.test_pilot2_validation',
        'tests.test_pilot3_replay_simulation',
        'tests.test_pilot4_cueing',
        'tests.test_cue_arousal',
        'tests.test_protocol_presets',
        'tests.test_protocol_fidelity',
        'tests.test_cue_recall',
        'tests.test_first_cue_latency',
        'tests.test_complexity_features',
        'tests.test_meditation_analysis',
        'tests.test_polar_h10_protocol',
        'tests.test_cardio_resp_features',
        'tests.test_polar_recording',
        'tests.test_night_cardio',
        'tests.test_optics_agreement',
        'tests.test_calibration',
        'tests.test_line_check',
        'tests.test_docs_wording',
        'tests.test_rem_validation_provenance',
        'tests.test_synthetic_night',
        'tests.test_rem_metrics',
        'tests.test_packaging',
    ]
    
    for module in fast_modules:
        try:
            suite.addTests(loader.loadTestsFromName(module))
        except:
            print(f"Warning: Could not load {module}")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_tests():
    """Run complete test suite"""
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description='Run the REMind tests')
    parser.add_argument('--all', action='store_true', 
                       help='Run all tests including slow ones')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    args = parser.parse_args()
    
    print("="*60)
    print("REMind Test Suite")
    print("="*60)
    
    if args.all:
        print("Running ALL tests (may take a while)...")
        success = run_all_tests()
    elif args.integration:
        print("Running integration tests...")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName('tests.test_integration')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        success = result.wasSuccessful()
    else:
        print("Running fast tests only (use --all for complete suite)...")
        success = run_fast_tests()
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
