import numpy as np

class EnsembleModel:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def add_model(self, name, model, weight=1.0):
        self.models[name] = model
        self.weights[name] = weight
        print(f"➕ Added {name} (weight: {weight})")
    
    def remove_model(self, name):
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            print(f"➖ Removed {name}")
    
    def train_all(self, X_train, y_train, **kwargs):
        for name, model in self.models.items():
            print(f"\n🎯 Training {name}...")
            if hasattr(model, 'train'):
                model.train(X_train, y_train, **kwargs)
        self.is_trained = True
        print("\n✅ All models trained")
    
    def predict(self, X, method='weighted'):
        if not self.is_trained:
            raise Exception("Models not trained yet!")
        
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                predictions[name] = model.predict(X)
        
        if method == 'weighted':
            total_weight = sum(self.weights.values())
            weighted_sum = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                weighted_sum += pred * (self.weights[name] / total_weight)
            return weighted_sum
        
        elif method == 'average':
            all_preds = np.array(list(predictions.values()))
            return np.mean(all_preds, axis=0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
