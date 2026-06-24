from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_with_cv(X, y, cv_folds=5):
    model = RandomForestRegressor(n_estimators=100)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-validation RMSE: {rmse_scores.mean()} (+/- {rmse_scores.std() * 2})")
    
    model.fit(X, y)
    return model