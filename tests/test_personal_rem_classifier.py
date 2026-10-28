import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from muse_tmr.annotations import RemAnnotation, export_rem_annotations
from muse_tmr.cli.main import main
from muse_tmr.models import (
    PersonalRemClassifierConfig,
    PersonalRemModel,
    train_personal_rem_classifier,
)
from muse_tmr.models.ml_rem_detector import _training_matrix, _warm_start

TEST_FEATURES = (
    "p_rem",
    "feature_score_eye_movement_proxy",
    "feature_score_stillness",
)


def annotation(label, index, recording_id, *, p_rem, eye_score, stillness_score):
    return RemAnnotation(
        recording_id=recording_id,
        epoch_index=index,
        start_time=1000.0 + 30.0 * index,
        end_time=1030.0 + 30.0 * index,
        duration_seconds=30.0,
        label=label,
        p_rem=p_rem,
        feature_scores={
            "eye_movement_proxy": eye_score,
            "stillness": stillness_score,
        },
        prediction_source="heuristic",
    )


def labeled_rows():
    rows = []
    index = 0
    for recording_id in ("night-a", "night-b"):
        for offset in range(4):
            rows.append(
                annotation(
                    "probable_rem",
                    index,
                    recording_id,
                    p_rem=0.78 + offset * 0.03,
                    eye_score=0.80 + offset * 0.02,
                    stillness_score=0.85 + offset * 0.02,
                )
            )
            index += 1
        for offset in range(4):
            rows.append(
                annotation(
                    "wake" if offset % 2 == 0 else "nrem",
                    index,
                    recording_id,
                    p_rem=0.10 + offset * 0.03,
                    eye_score=0.05 + offset * 0.02,
                    stillness_score=0.15 + offset * 0.03,
                )
            )
            index += 1
    return tuple(rows)


def test_config():
    return PersonalRemClassifierConfig(
        feature_names=TEST_FEATURES,
        learning_rate=0.10,
        epochs=800,
        l2_penalty=0.0,
        min_training_rows=4,
    )


class TestPersonalRemClassifier(unittest.TestCase):
    def test_train_predicts_rem_like_rows_higher_than_wake_rows(self):
        model = train_personal_rem_classifier(labeled_rows(), config=test_config())

        rem_prediction = model.predict_mapping(
            annotation(
                "unknown",
                100,
                "night-c",
                p_rem=0.86,
                eye_score=0.88,
                stillness_score=0.91,
            ).to_training_dict()
        )
        wake_prediction = model.predict_mapping(
            annotation(
                "unknown",
                101,
                "night-c",
                p_rem=0.12,
                eye_score=0.04,
                stillness_score=0.20,
            ).to_training_dict()
        )

        self.assertEqual(rem_prediction.source, "personal")
        self.assertGreater(rem_prediction.probability, 0.70)
        self.assertLess(wake_prediction.probability, 0.30)
        self.assertIn("personal_model_positive", rem_prediction.reason_codes)
        self.assertFalse(hasattr(rem_prediction, "should_play"))

    def test_unknown_labels_are_skipped_and_missing_features_are_imputed(self):
        unknown = annotation(
            "unknown",
            999,
            "night-x",
            p_rem=0.99,
            eye_score=1.0,
            stillness_score=1.0,
        )
        model = train_personal_rem_classifier(labeled_rows() + (unknown,), config=test_config())

        prediction = model.predict_mapping({"p_rem": 0.8, "feature_score_eye_movement_proxy": 0.9})

        self.assertEqual(model.training_summary.skipped_unknown_rows, 1)
        self.assertIn("missing_features_imputed", prediction.reason_codes)

    def test_requires_both_positive_and_negative_labels(self):
        positive_only = tuple(row for row in labeled_rows() if row.label == "probable_rem")

        with self.assertRaises(ValueError):
            train_personal_rem_classifier(positive_only, config=test_config())

    def test_model_artifact_roundtrip_preserves_predictions_and_metrics(self):
        model = train_personal_rem_classifier(labeled_rows(), config=test_config())
        row = labeled_rows()[0].to_training_dict()

        with tempfile.TemporaryDirectory() as tmp:
            model_path = model.save(Path(tmp) / "personal_rem_model.json")
            loaded = PersonalRemModel.load(model_path)
            payload = json.loads(model_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["model_version"], 2)
        self.assertIsNone(payload["training_data"])
        self.assertEqual(loaded.feature_names, TEST_FEATURES)
        self.assertIn("feature_importance", payload)
        self.assertIn("calibration_bins", payload["training_summary"]["metrics"])
        self.assertEqual(
            payload["training_summary"]["group_holdout_metrics"]["status"],
            "computed",
        )
        self.assertAlmostEqual(
            model.predict_mapping(row).probability,
            loaded.predict_mapping(row).probability,
            places=9,
        )

    def test_cli_trains_and_saves_personal_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotations_path = export_rem_annotations(labeled_rows(), tmp_path / "labels.csv")
            model_path = tmp_path / "model.json"

            with redirect_stdout(io.StringIO()):
                exit_code = main([
                    "train-rem-classifier",
                    str(annotations_path),
                    "--output",
                    str(model_path),
                    "--feature",
                    "p_rem",
                    "--feature",
                    "feature_score_eye_movement_proxy",
                    "--feature",
                    "feature_score_stillness",
                    "--epochs",
                    "800",
                ])

            payload = json.loads(model_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["model_type"], "balanced_logistic_regression")
        self.assertEqual(payload["training_summary"]["training_rows"], 16)
        self.assertEqual(payload["training_data"]["annotations"]["name"], "labels.csv")
        self.assertEqual(payload["training_data"]["recording_ids"], ["night-a", "night-b"])

    def test_a_version_1_artifact_still_loads(self):
        model = train_personal_rem_classifier(labeled_rows(), config=test_config())
        payload = model.to_dict()
        payload["model_version"] = 1
        payload.pop("training_data")
        row = labeled_rows()[0].to_training_dict()

        loaded = PersonalRemModel.from_dict(payload)

        self.assertEqual(loaded.model_version, 1)
        self.assertIsNone(loaded.training_data)
        self.assertEqual(loaded.predict_mapping(row).probability, model.predict_mapping(row).probability)
        payload["model_version"] = 3
        with self.assertRaises(ValueError):
            PersonalRemModel.from_dict(payload)


def shifted_rows():
    """The labelled rows with every feature moved and stretched, so their means and
    scales differ from labeled_rows()."""
    return tuple(
        replace(
            row,
            p_rem=0.5 * row.p_rem + 0.2,
            feature_scores={name: 0.6 * value + 0.1 for name, value in row.feature_scores.items()},
        )
        for row in labeled_rows()
    )


class WarmStartTest(unittest.TestCase):
    """train-rem-classifier --init: the warm start of #226."""

    def setUp(self):
        self.init = train_personal_rem_classifier(
            labeled_rows(), config=test_config(), training_data={"dataset": "boas", "splits_sha256": "abc"}
        )

    def test_the_init_needs_the_same_features_in_the_same_order(self):
        for names in (("p_rem", "feature_score_stillness"), tuple(reversed(TEST_FEATURES))):
            with self.subTest(names=names), self.assertRaises(ValueError):
                train_personal_rem_classifier(
                    labeled_rows(), config=replace(test_config(), feature_names=names), init=self.init
                )

    def test_the_start_predicts_what_the_init_predicts(self):
        rows = [row.to_training_dict() for row in shifted_rows()]
        _, means, scales, _ = _training_matrix(rows, TEST_FEATURES)
        coefficients, bias = _warm_start(self.init, TEST_FEATURES, means, scales)

        for row in rows:
            normalized = [(float(row[name]) - means[name]) / scales[name] for name in TEST_FEATURES]
            start_logit = float(sum(w * z for w, z in zip(coefficients, normalized)) + bias)
            init_logit = self.init.predict_mapping(row).feature_values["personal_model_logit"]
            self.assertAlmostEqual(start_logit, init_logit, places=9)

    def test_the_penalty_pulls_toward_the_init_not_toward_zero(self):
        # Labels turned around: the data pulls the other way, a strong penalty holds.
        flipped = tuple(
            replace(row, label="wake" if row.label == "probable_rem" else "probable_rem") for row in labeled_rows()
        )
        # The balanced loss moves a coefficient by at most about 1 / l2 from where the
        # penalty pulls it; learning rate 0.1 times l2 8 keeps the descent stable.
        config = replace(test_config(), l2_penalty=8.0)
        rows = [row.to_training_dict() for row in flipped]
        _, means, scales, _ = _training_matrix(rows, TEST_FEATURES)
        start, _ = _warm_start(self.init, TEST_FEATURES, means, scales)

        warm = train_personal_rem_classifier(flipped, config=config, init=self.init)
        cold = train_personal_rem_classifier(flipped, config=config)

        self.assertGreater(min(abs(value) for value in start), 0.5)
        for name, anchor in zip(TEST_FEATURES, start):
            self.assertLess(abs(warm.coefficients[name] - anchor), 0.15, name)
            self.assertLess(abs(cold.coefficients[name]), 0.15, name)

    def test_training_data_records_the_init_and_survives_a_save(self):
        warm = train_personal_rem_classifier(
            shifted_rows(), config=test_config(), init=self.init, training_data={"annotations": {"name": "x.csv"}}
        )
        with tempfile.TemporaryDirectory() as tmp:
            loaded = PersonalRemModel.load(warm.save(Path(tmp) / "warm.json"))

        self.assertEqual(loaded.training_data["annotations"], {"name": "x.csv"})
        self.assertEqual(loaded.training_data["init"]["model_version"], 2)
        self.assertEqual(loaded.training_data["init"]["training_data"]["dataset"], "boas")
        self.assertEqual(loaded.training_data["init"]["trained_at_utc"], self.init.trained_at_utc)

    def test_the_cli_takes_the_init_and_its_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            annotations_path = export_rem_annotations(shifted_rows(), tmp_path / "labels.csv")
            init_path = self.init.save(tmp_path / "init.json")
            model_path = tmp_path / "model.json"
            with redirect_stdout(io.StringIO()) as output:
                code = main(["train-rem-classifier", str(annotations_path), "--output", str(model_path),
                             "--init", str(init_path)])
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            init_sha = hashlib.sha256(init_path.read_bytes()).hexdigest()
            refused = tmp_path / "refused.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error:
                bad = main(["train-rem-classifier", str(annotations_path), "--output", str(refused),
                            "--init", str(init_path), "--feature", "p_rem"])

        self.assertEqual(code, 0)
        self.assertIn("init=init.json", output.getvalue())
        self.assertEqual(payload["feature_names"], list(TEST_FEATURES))
        self.assertEqual(payload["training_data"]["init_artifact"], {"name": "init.json", "sha256": init_sha})
        self.assertEqual(payload["training_data"]["init"]["training_data"]["splits_sha256"], "abc")
        self.assertEqual(bad, 2)
        self.assertIn("same feature_names", error.getvalue())
        self.assertFalse(refused.exists())


if __name__ == "__main__":
    unittest.main()
