import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlProject.components.data_transformation import DataTransformation, NUMERIC_FEATURES, FeatureEngineer, OutlierCapper
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformationSplitTest(unittest.TestCase):
    def test_train_test_split_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            data_path = root_dir / "wine.csv"
            rows = []
            for quality in (5, 6):
                for i in range(20):
                    rows.append({
                        "fixed acidity": i, "volatile acidity": i/10, "citric acid": i/20,
                        "residual sugar": i, "chlorides": i/100, "free sulfur dioxide": i,
                        "total sulfur dioxide": i*2, "density": 0.99 + i/1000, "pH": 3.0 + i/100,
                        "sulphates": i/20, "alcohol": 9.0 + i/10, "quality": quality,
                    })
            pd.DataFrame(rows).to_csv(data_path, index=False)

            config = DataTransformationConfig(
                root_dir=root_dir,
                data_path=data_path,
                test_size=0.25,
                random_state=42,
                stratify_column="quality",
                use_scaler=False,
            )

            DataTransformation(config).train_test_spliting()
            first_train = pd.read_csv(root_dir / "train.csv")
            first_test = pd.read_csv(root_dir / "test.csv")

            DataTransformation(config).train_test_spliting()
            second_train = pd.read_csv(root_dir / "train.csv")
            second_test = pd.read_csv(root_dir / "test.csv")

            pd.testing.assert_frame_equal(first_train, second_train)
            pd.testing.assert_frame_equal(first_test, second_test)
            self.assertEqual(len(first_train), 30)
            self.assertEqual(len(first_test), 10)
            self.assertEqual(first_test["quality"].value_counts().to_dict(), {5: 5, 6: 5})


class TestPreprocessing(unittest.TestCase):
    def test_feature_engineer_adds_derived_features(self):
        data = np.array([
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8],
        ], dtype=float)
        fe = FeatureEngineer(
            add_acidity_index=True,
            add_alcohol_sugar_ratio=True,
            add_free_sulfur_pct=True,
        )
        transformed = fe.fit_transform(data)
        self.assertEqual(transformed.shape[1], 14)
        fe_names = fe.get_feature_names_out()
        self.assertIn("acidity_index", fe_names)
        self.assertIn("alcohol_sugar_ratio", fe_names)
        self.assertIn("free_sulfur_pct", fe_names)

    def test_derived_features_mathematically_correct(self):
        data = np.array([
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
            [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8],
        ], dtype=float)
        fe = FeatureEngineer(
            add_acidity_index=True,
            add_alcohol_sugar_ratio=True,
            add_free_sulfur_pct=True,
        )
        transformed = fe.fit_transform(data)
        # acidity_index = fixed_acidity / pH
        expected_acidity = data[:, 0] / data[:, 8]
        np.testing.assert_array_almost_equal(transformed[:, 11], expected_acidity)
        # alcohol_sugar_ratio = alcohol / residual_sugar
        expected_ratio = data[:, 10] / data[:, 3]
        np.testing.assert_array_almost_equal(transformed[:, 12], expected_ratio)
        # free_sulfur_pct = free_sulfur_dioxide / total_sulfur_dioxide * 100
        expected_pct = data[:, 5] / data[:, 6] * 100
        np.testing.assert_array_almost_equal(transformed[:, 13], expected_pct)

    def test_feature_engineer_raises_on_missing_column(self):
        fe = FeatureEngineer(add_acidity_index=True)
        with self.assertRaises(ValueError):
            fe.fit(np.array([[1.0, 2.0, 3.0]]))

    def test_outlier_capper_clips_extreme_values(self):
        data = np.array([[1.0, 1000.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=float)
        capper = OutlierCapper(method="iqr", iqr_multiplier=1.5)
        transformed = capper.fit_transform(data)
        self.assertLess(transformed[0, 1], 1000.0)

    def test_preprocessing_pipeline_saved_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            data_path = root_dir / "wine.csv"
            rows = []
            for quality in (5, 6):
                for i in range(20):
                    rows.append({
                        "fixed acidity": i, "volatile acidity": i/10, "citric acid": i/20,
                        "residual sugar": i, "chlorides": i/100, "free sulfur dioxide": i,
                        "total sulfur dioxide": i*2, "density": 0.99 + i/1000, "pH": 3.0 + i/100,
                        "sulphates": i/20, "alcohol": 9.0 + i/10, "quality": quality,
                    })
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = DataTransformationConfig(
                root_dir=root_dir,
                data_path=data_path,
                test_size=0.25,
                random_state=42,
                stratify_column="quality",
                use_scaler=True,
                scaler_type="standard",
            )
            dt = DataTransformation(config)
            dt.train_test_spliting()
            preprocessor_path = root_dir / "preprocessor.joblib"
            self.assertTrue(preprocessor_path.exists())
            preprocessor = joblib.load(preprocessor_path)
            self.assertIsNotNone(preprocessor)

    def test_build_preprocessing_pipeline_honors_feature_engineering_flags(self):
        config = DataTransformationConfig(
            root_dir=Path("artifacts"),
            data_path=Path("artifacts/wine.csv"),
            test_size=0.25,
            random_state=42,
            stratify_column="quality",
            use_scaler=True,
            scaler_type="standard",
            handle_outliers=False,
            impute_missing=False,
            feature_engineering_flags={
                "add_acidity_index": False,
                "add_alcohol_sugar_ratio": True,
                "add_free_sulfur_pct": False,
            },
        )
        dt = DataTransformation(config)
        pipeline = dt._build_preprocessing_pipeline()
        fe = pipeline.named_steps["feature_engineer"]
        self.assertFalse(fe.add_acidity_index)
        self.assertTrue(fe.add_alcohol_sugar_ratio)
        self.assertFalse(fe.add_free_sulfur_pct)

    def test_build_preprocessing_pipeline_skips_outlier_capper_when_disabled(self):
        config = DataTransformationConfig(
            root_dir=Path("artifacts"),
            data_path=Path("artifacts/wine.csv"),
            test_size=0.25,
            random_state=42,
            stratify_column="quality",
            use_scaler=True,
            scaler_type="standard",
            handle_outliers=False,
            impute_missing=True,
        )
        dt = DataTransformation(config)
        pipeline = dt._build_preprocessing_pipeline()
        numeric = pipeline.named_steps["numeric"]
        step_names = [name for name, _ in numeric.steps]
        self.assertNotIn("outlier_capper", step_names)

    def test_build_preprocessing_pipeline_skips_imputer_when_disabled(self):
        config = DataTransformationConfig(
            root_dir=Path("artifacts"),
            data_path=Path("artifacts/wine.csv"),
            test_size=0.25,
            random_state=42,
            stratify_column="quality",
            use_scaler=True,
            scaler_type="standard",
            handle_outliers=False,
            impute_missing=False,
        )
        dt = DataTransformation(config)
        pipeline = dt._build_preprocessing_pipeline()
        numeric = pipeline.named_steps["numeric"]
        step_names = [name for name, _ in numeric.steps]
        self.assertNotIn("imputer", step_names)

    def test_build_preprocessing_pipeline_uses_outlier_iqr_multiplier_from_config(self):
        config = DataTransformationConfig(
            root_dir=Path("artifacts"),
            data_path=Path("artifacts/wine.csv"),
            test_size=0.25,
            random_state=42,
            stratify_column="quality",
            use_scaler=True,
            scaler_type="standard",
            handle_outliers=True,
            outlier_method="iqr",
            outlier_iqr_multiplier=3.0,
            impute_missing=False,
        )
        dt = DataTransformation(config)
        pipeline = dt._build_preprocessing_pipeline()
        numeric = pipeline.named_steps["numeric"]
        capper = numeric.named_steps["outlier_capper"]
        self.assertEqual(capper.iqr_multiplier, 3.0)


    def test_outlier_capper_runs_before_scaler_in_pipeline(self):
        """Regression test: outlier capping must be applied BEFORE standardization."""
        config = DataTransformationConfig(
            root_dir=Path("/tmp"),
            data_path=Path("/tmp/wine.csv"),
            test_size=0.25,
            random_state=42,
            stratify_column="quality",
            use_scaler=True,
            scaler_type="standard",
            handle_outliers=True,
            outlier_method="iqr",
            outlier_iqr_multiplier=1.5,
            impute_missing=False,
        )
        dt = DataTransformation(config)
        pipeline = dt._build_preprocessing_pipeline()
        numeric = pipeline.named_steps["numeric"]
        step_names = [name for name, _ in numeric.steps]
        scaler_idx = step_names.index("scaler")
        capper_idx = step_names.index("outlier_capper")
        self.assertLess(
            capper_idx, scaler_idx,
            "OutlierCapper must appear before StandardScaler in the pipeline"
        )

    def test_outlier_capper_on_right_skewed_data_before_scaler(self):
        """Verify capping raw data then scaling gives expected distribution."""
        np.random.seed(42)
        data = np.random.lognormal(mean=0, sigma=1.5, size=(1000, 1))

        capper = OutlierCapper(method="iqr", iqr_multiplier=1.5)
        scaler = StandardScaler()

        capped = capper.fit_transform(data)
        scaled = scaler.fit_transform(capped)

        capped_range = scaled.max() - scaled.min()
        self.assertGreater(
            capped_range, 2.0,
            "Capping before scaling should preserve a wider range than capping after scaling"
        )


if __name__ == "__main__":
    unittest.main()
