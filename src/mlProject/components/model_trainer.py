import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def train_with_cv(X, y, cv_folds=5):
    model = RandomForestRegressor(n_estimators=100)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-validation RMSE: {rmse_scores.mean()} (+/- {rmse_scores.std() * 2})")
    
    model.fit(X, y)
    return model


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        train_data_path = getattr(self.config, "train_data_path", Path("artifacts/data_transformation/train.csv"))
        target_column = getattr(self.config, "target_column", "quality")
        model_name = getattr(self.config, "model_name", "model.joblib")
        root_dir = getattr(self.config, "root_dir", Path("artifacts/model_trainer"))
        os.makedirs(root_dir, exist_ok=True)

        train_data = pd.read_csv(train_data_path)
        X = train_data.drop(columns=[target_column])
        y = train_data[target_column]

        model = train_with_cv(X, y)

        model_path = root_dir / model_name
        joblib.dump(model, model_path)

        model_info = {
            "model_path": str(model_path),
            "model_name": model_name,
            "features": list(X.columns),
            "target": target_column,
        }
        info_path = root_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        print(f"Model saved to {model_path}")
        return model