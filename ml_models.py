"""
Advanced machine learning models for stock prediction.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from prophet import Prophet
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("stockvision.ml_models")

class LSTMStockPredictor:
    """LSTM-based stock prediction model."""
    
    def __init__(self, sequence_length: int = 60, epochs: int = 50, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = ['Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI_14', 'Volatility_30']
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0
        
        # Select features and scale
        features = data[self.feature_columns].values
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, 0])  # Predict Close price
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model."""
        try:
            X, y = self.prepare_data(data)
            
            if len(X) < 100:
                raise ValueError("Not enough data for LSTM training (need at least 100 sequences)")
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build and train model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            return {
                "status": "success",
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "epochs_trained": len(history.history['loss']),
                "final_loss": float(history.history['loss'][-1])
            }
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def predict(self, data: pd.DataFrame, days: int = 30) -> np.ndarray:
        """Make predictions using trained LSTM model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare the last sequence for prediction
        features = data[self.feature_columns].values
        scaled_features = self.scaler.transform(features)
        
        last_sequence = scaled_features[-self.sequence_length:]
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, self.sequence_length, len(self.feature_columns))
            
            # Predict next price
            pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            
            # Update sequence (simplified - in reality, you'd update all features)
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled  # Update Close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        dummy_features = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_features[:, 0] = predictions  # Put predictions in Close price column
        predictions_actual = self.scaler.inverse_transform(dummy_features)[:, 0]
        
        return predictions_actual

class EnsemblePredictor:
    """Ensemble model combining Prophet and LSTM."""
    
    def __init__(self, lstm_config: Optional[Dict] = None):
        self.lstm_predictor = LSTMStockPredictor(**(lstm_config or {}))
        self.prophet_model = None
        self.ensemble_weights = {"prophet": 0.6, "lstm": 0.4}  # Default weights
        
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train both Prophet and LSTM models."""
        results = {}
        
        # Train Prophet model
        try:
            prophet_data = data[['Date', 'Close']].copy()
            prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            self.prophet_model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            self.prophet_model.fit(prophet_data)
            
            # Evaluate Prophet
            prophet_pred = self.prophet_model.predict(prophet_data)
            prophet_mae = mean_absolute_error(prophet_data['y'], prophet_pred['yhat'])
            prophet_rmse = np.sqrt(mean_squared_error(prophet_data['y'], prophet_pred['yhat']))
            
            results["prophet"] = {
                "status": "success",
                "mae": float(prophet_mae),
                "rmse": float(prophet_rmse)
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            results["prophet"] = {"status": "failed", "error": str(e)}
        
        # Train LSTM model
        lstm_results = self.lstm_predictor.train(data)
        results["lstm"] = lstm_results
        
        # Calculate ensemble weights based on performance
        if results["prophet"]["status"] == "success" and results["lstm"]["status"] == "success":
            prophet_mae = results["prophet"]["mae"]
            lstm_mae = results["lstm"]["val_mae"]
            
            # Inverse performance weighting (lower error = higher weight)
            total_error = prophet_mae + lstm_mae
            self.ensemble_weights["prophet"] = 1 - (prophet_mae / total_error)
            self.ensemble_weights["lstm"] = 1 - (lstm_mae / total_error)
            
            results["ensemble_weights"] = self.ensemble_weights
        
        return results
    
    def predict(self, data: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
        """Generate ensemble predictions."""
        if self.prophet_model is None or self.lstm_predictor.model is None:
            raise ValueError("Models not trained yet")
        
        predictions = {}
        
        # Prophet predictions
        try:
            prophet_data = data[['Date', 'Close']].copy()
            prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            future = self.prophet_model.make_future_dataframe(periods=days)
            prophet_forecast = self.prophet_model.predict(future)
            
            prophet_pred = prophet_forecast.tail(days)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            predictions["prophet"] = {
                "values": prophet_pred['yhat'].values,
                "lower": prophet_pred['yhat_lower'].values,
                "upper": prophet_pred['yhat_upper'].values,
                "dates": prophet_pred['ds'].dt.strftime('%Y-%m-%d').tolist()
            }
            
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            predictions["prophet"] = {"error": str(e)}
        
        # LSTM predictions
        try:
            lstm_values = self.lstm_predictor.predict(data, days)
            
            # Generate confidence intervals (simplified)
            lstm_std = np.std(lstm_values) * 0.1
            predictions["lstm"] = {
                "values": lstm_values,
                "lower": lstm_values - lstm_std,
                "upper": lstm_values + lstm_std
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            predictions["lstm"] = {"error": str(e)}
        
        # Ensemble predictions
        if "prophet" in predictions and "lstm" in predictions and "error" not in predictions["prophet"] and "error" not in predictions["lstm"]:
            prophet_vals = predictions["prophet"]["values"]
            lstm_vals = predictions["lstm"]["values"]
            
            ensemble_values = (
                self.ensemble_weights["prophet"] * prophet_vals +
                self.ensemble_weights["lstm"] * lstm_vals
            )
            
            # Ensemble confidence intervals
            ensemble_lower = (
                self.ensemble_weights["prophet"] * predictions["prophet"]["lower"] +
                self.ensemble_weights["lstm"] * predictions["lstm"]["lower"]
            )
            
            ensemble_upper = (
                self.ensemble_weights["prophet"] * predictions["prophet"]["upper"] +
                self.ensemble_weights["lstm"] * predictions["lstm"]["upper"]
            )
            
            predictions["ensemble"] = {
                "values": ensemble_values,
                "lower": ensemble_lower,
                "upper": ensemble_upper,
                "weights": self.ensemble_weights,
                "dates": predictions["prophet"]["dates"]
            }
        
        return predictions
    
    def get_model_confidence(self) -> Dict[str, float]:
        """Get confidence scores for each model."""
        confidence = {}
        
        if self.prophet_model:
            confidence["prophet"] = 0.8  # Prophet is generally reliable
        
        if self.lstm_predictor.model:
            confidence["lstm"] = 0.7  # LSTM can be less stable
        
        if self.prophet_model and self.lstm_predictor.model:
            confidence["ensemble"] = 0.85  # Ensemble typically performs best
        
        return confidence
