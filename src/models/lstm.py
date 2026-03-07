import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import os

class LSTMModel:
    def __init__(self, sequence_length=60, n_features=1, lstm_units=[50, 50],
                 dropout_rate=0.2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.is_trained = False
        
    def build_model(self):
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        
        # Additional LSTM layers
        for units in self.lstm_units[1:]:
            model.add(layers.LSTM(units, return_sequences=False))
        
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dense(1))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print("✅ LSTM model built")
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, patience=10, verbose=1):
        if self.model is None:
            self.build_model()
        
        callbacks_list = []
        if X_val is not None:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
            callbacks_list.append(early_stopping)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.is_trained = True
        print("✅ Training complete")
        return self.history
    
    def predict(self, X_test):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        return self.model.predict(X_test)
    
    def predict_future(self, last_sequence, days=30):
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        predictions = []
        current_seq = last_sequence.copy()
        
        for _ in range(days):
            next_pred = self.model.predict(current_seq.reshape(1, self.sequence_length, self.n_features))
            predictions.append(next_pred[0, 0])
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = next_pred
        
        return np.array(predictions)
    
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        print(f"📂 Loaded from {filepath}")
    
    def plot_training_history(self):
        if self.history is None:
            print("No training history found")
            return None
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
