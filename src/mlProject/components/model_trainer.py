import os
import pandas as pd
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        lr = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=42
        )
        lr.fit(train_x, train_y)

        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(lr, model_path)

        logger.info(f"Model saved at {model_path}")


def train_with_cv(X, y, cv_folds=5):
    model = RandomForestRegressor(n_estimators=100)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    logger.info(
        f"Cross-validation RMSE: {rmse_scores.mean()} (+/- {rmse_scores.std() * 2})"
    )
    
    model.fit(X, y)
    return model