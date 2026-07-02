import os
import joblib
import numpy as np
from pathlib import Path
from mlProject import logger
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from mlProject import logger

NUMERIC_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]


def _bucket_quality(q, rare_classes):
    """Map a rare quality value to a bucketed group with an adjacent class."""
    rare_set = set(rare_classes)
    if q in rare_set:
        for delta in [-1, 1]:
            neighbor = q + delta
            if neighbor not in rare_set:
                return f"{min(q, neighbor)}-{max(q, neighbor)}"
        return "rare"
    return str(q)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers using IQR-based capping or percentile-based capping."""
    def __init__(self, method: str = "iqr", iqr_multiplier: float = 1.5):
        self.method = method
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        total_capped = 0
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.lower_bounds_[i] = lower
            self.upper_bounds_[i] = upper
            total_capped += int(np.sum((col < lower) | (col > upper)))
        logger.info(
            f"OutlierCapper: {total_capped}/{X_arr.size} values ({100 * total_capped / X_arr.size:.2f}%) "
            f"will be capped using {self.method} method (multiplier={self.iqr_multiplier})"
        )
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        for i in range(X_arr.shape[1]):
            lower = self.lower_bounds_.get(i, -np.inf)
            upper = self.upper_bounds_.get(i, np.inf)
            capped = int(np.sum((X_arr[:, i] < lower) | (X_arr[:, i] > upper)))
            if capped > 0:
                logger.debug(f"Feature {i}: capped {capped} outliers")
            X_arr[:, i] = np.clip(X_arr[:, i], lower, upper)
        return X_arr


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived features: acidity_index, alcohol_sugar_ratio, free_sulfur_pct."""
    def __init__(self, add_acidity_index=True, add_alcohol_sugar_ratio=True, add_free_sulfur_pct=True):
        self.add_acidity_index = add_acidity_index
        self.add_alcohol_sugar_ratio = add_alcohol_sugar_ratio
        self.add_free_sulfur_pct = add_free_sulfur_pct
        self.n_features_in_ = len(NUMERIC_FEATURES)

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self._resolve_indices_by_name(list(X.columns))
        else:
            if X.shape[1] < len(NUMERIC_FEATURES):
                raise ValueError(
                    f"Expected at least {len(NUMERIC_FEATURES)} features, got {X.shape[1]}"
                )
            self._resolve_indices_by_position()
        return self

    def _resolve_indices_by_name(self, cols):
        required = []
        if self.add_acidity_index:
            required.extend(["fixed acidity", "pH"])
        if self.add_alcohol_sugar_ratio:
            required.extend(["alcohol", "residual sugar"])
        if self.add_free_sulfur_pct:
            required.extend(["free sulfur dioxide", "total sulfur dioxide"])
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(
                f"Required columns missing for feature engineering: {missing}. "
                f"Available columns: {cols}"
            )
        if self.add_acidity_index:
            self._fixed_idx = cols.index("fixed acidity")
            self._ph_idx = cols.index("pH")
        if self.add_alcohol_sugar_ratio:
            self._alcohol_idx = cols.index("alcohol")
            self._sugar_idx = cols.index("residual sugar")
        if self.add_free_sulfur_pct:
            self._free_sulfur_idx = cols.index("free sulfur dioxide")
            self._total_sulfur_idx = cols.index("total sulfur dioxide")

    def _resolve_indices_by_position(self):
        idx_map = dict(zip(NUMERIC_FEATURES, range(len(NUMERIC_FEATURES))))
        if self.add_acidity_index:
            self._fixed_idx = idx_map["fixed acidity"]
            self._ph_idx = idx_map["pH"]
        if self.add_alcohol_sugar_ratio:
            self._alcohol_idx = idx_map["alcohol"]
            self._sugar_idx = idx_map["residual sugar"]
        if self.add_free_sulfur_pct:
            self._free_sulfur_idx = idx_map["free sulfur dioxide"]
            self._total_sulfur_idx = idx_map["total sulfur dioxide"]

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        additional = []
        if self.add_acidity_index:
            fixed = X_arr[:, self._fixed_idx]
            ph = X_arr[:, self._ph_idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                idx = np.where(ph > 0, fixed / ph, 0)
            additional.append(idx)
        if self.add_alcohol_sugar_ratio:
            alcohol = X_arr[:, self._alcohol_idx]
            sugar = X_arr[:, self._sugar_idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(sugar > 0, alcohol / sugar, 0)
            additional.append(ratio)
        if self.add_free_sulfur_pct:
            free_sulfur = X_arr[:, self._free_sulfur_idx]
            total_sulfur = X_arr[:, self._total_sulfur_idx]
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = np.where(total_sulfur > 0, free_sulfur / total_sulfur * 100, 0)
            additional.append(pct)
        if additional:
            return np.column_stack([X_arr] + additional)
        return X_arr

    def get_feature_names_out(self, input_features=None):
        names = list(NUMERIC_FEATURES)
        if self.add_acidity_index:
            names.append("acidity_index")
        if self.add_alcohol_sugar_ratio:
            names.append("alcohol_sugar_ratio")
        if self.add_free_sulfur_pct:
            names.append("free_sulfur_pct")
        return names


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _build_preprocessing_pipeline(self):
        """Build a full preprocessing pipeline with imputation, scaling, outlier capping, and feature engineering."""
        scaler_map = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        scaler = scaler_map.get(self.config.scaler_type, StandardScaler())

        numeric_steps = []
        if self.config.impute_missing:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        if self.config.handle_outliers:
            numeric_steps.append(("outlier_capper", OutlierCapper(
                method=self.config.outlier_method,
                iqr_multiplier=self.config.outlier_iqr_multiplier,
            )))
        numeric_steps.append(("scaler", scaler))

        numeric_transformer = Pipeline(steps=numeric_steps)

        fe_flags = self.config.feature_engineering_flags or {}
        preprocessor = Pipeline(steps=[
            ("feature_engineer", FeatureEngineer(
                add_acidity_index=fe_flags.get("add_acidity_index", True),
                add_alcohol_sugar_ratio=fe_flags.get("add_alcohol_sugar_ratio", True),
                add_free_sulfur_pct=fe_flags.get("add_free_sulfur_pct", True),
            )),
            ("numeric", numeric_transformer),
        ])
        return preprocessor

    def train_test_spliting(self):
        try:
            data = pd.read_csv(self.config.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config.data_path}")
            raise
        except Exception as e:
            logger.exception(f"Failed to read data file: {self.config.data_path}")
            raise

        stratify = None
        if self.config.stratify_column:
            if self.config.stratify_column not in data.columns:
                raise ValueError(
                    f"Stratify column '{self.config.stratify_column}' "
                    "not found in transformed data"
                )
            stratify = data[self.config.stratify_column]
            # Bucket rare classes to ensure stratification can succeed
            min_samples = int(1 / self.config.test_size) + 1
            value_counts = data[self.config.stratify_column].value_counts()
            rare_classes = value_counts[value_counts < min_samples].index.tolist()
            if rare_classes and self.config.min_samples_per_class > 0:
                logger.warning(
                    f"Classes {rare_classes} have fewer than {min_samples} samples. "
                    "Bucketing adjacent quality scores for stratification."
                )
                data = data.copy()
                data["_stratify_bucket"] = data[self.config.stratify_column].apply(
                    lambda q: _bucket_quality(q, rare_classes)
                )
                stratify = data["_stratify_bucket"]

        try:
            train, test = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            if stratify is None:
                logger.exception("train_test_split failed")
                raise
            logger.critical(
                "Stratified split failed even after bucketing rare classes. "
                "Test set will not represent the data distribution. "
                "Exception: %s", exc
            )
            raise

        # Drop temporary bucketing column if present
        for col in ("_stratify_bucket",):
            if col in train.columns:
                train = train.drop(columns=[col])
            if col in test.columns:
                test = test.drop(columns=[col])

        if self.config.use_scaler:
            preprocessor = self._build_preprocessing_pipeline()
            train_features = train[NUMERIC_FEATURES]
            test_features = test[NUMERIC_FEATURES]
            train_target = train[[self.config.stratify_column]] if self.config.stratify_column in train.columns else None
            test_target = test[[self.config.stratify_column]] if self.config.stratify_column in test.columns else None

            train_scaled = preprocessor.fit_transform(train_features)
            test_scaled = preprocessor.transform(test_features)

            try:
                feat_names = preprocessor.named_steps["feature_engineer"].get_feature_names_out()
            except Exception:
                feat_names = [f"feat_{i}" for i in range(train_scaled.shape[1])]

            train_scaled_df = pd.DataFrame(train_scaled, columns=feat_names)
            test_scaled_df = pd.DataFrame(test_scaled, columns=feat_names)

            if train_target is not None:
                train_scaled_df[self.config.stratify_column] = train_target.values
                test_scaled_df[self.config.stratify_column] = test_target.values

            preprocessor_path = os.path.join(self.config.root_dir, "preprocessor.joblib")
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessing pipeline saved to {preprocessor_path}")
            assert Path(preprocessor_path).exists(), f"Preprocessor not found at {preprocessor_path}"

            feat_dim = len(NUMERIC_FEATURES)
            fe_flags = self.config.feature_engineering_flags or {}
            num_engineered = sum([
                fe_flags.get("add_acidity_index", True),
                fe_flags.get("add_alcohol_sugar_ratio", True),
                fe_flags.get("add_free_sulfur_pct", True),
            ])
            if train_scaled.shape[1] != feat_dim + num_engineered:
                logger.warning(
                    f"Preprocessor output dimension {train_scaled.shape[1]} "
                    f"does not match expected {feat_dim + num_engineered} (features + engineered)"
                )
        else:
            train_scaled_df = None
            test_scaled_df = None

        try:
            train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
            if self.config.use_scaler and train_scaled_df is not None:
                train_scaled_df.to_csv(os.path.join(self.config.root_dir, "train_scaled.csv"), index=False)
                test_scaled_df.to_csv(os.path.join(self.config.root_dir, "test_scaled.csv"), index=False)
        except OSError as e:
            logger.error(f"Failed to write train/test CSV files: {e}")
            raise

        logger.info("Splited data into training and test sets")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        if self.config.use_scaler:
            logger.info(f"Scaled train shape: {train_scaled_df.shape}, Scaled test shape: {test_scaled_df.shape}")

        logger.info(f"Train: {train.shape}, Test: {test.shape}")
