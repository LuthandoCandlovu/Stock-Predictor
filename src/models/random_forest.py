import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, n_jobs=-1):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        self.model = RandomForestRegressor(**self.params)
        self.is_trained = False
        
    def train(self, X_train, y_train):
        print("🌲 Training Random Forest...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ Training complete")
        
    def predict(self, X_test):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        return self.model.predict(X_test)
    
    def tune(self, X_train, y_train, cv=5):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
        
        search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.params = search.best_params_
        self.is_trained = True
        
        print(f"✅ Best params: {search.best_params_}")
        return search.best_params_
    
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"💾 Saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"📂 Loaded from {filepath}")
