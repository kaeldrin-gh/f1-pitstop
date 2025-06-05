"""
F1 Tire Degradation Model
Machine learning model for predicting tire degradation
"""

import logging
import pickle
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TireDegradationModel:
    """
    Machine learning model for predicting F1 tire degradation
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        """
        Initialize the tire degradation model
        
        Args:
            n_estimators: Number of trees in the random forest
            max_depth: Maximum depth of trees
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_columns = []
        self.model_metrics = {}
        self.feature_importance = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the tire degradation model
        
        Args:
            X: Features DataFrame
            y: Target variable (lap times)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training tire degradation model on {len(X)} samples")
        
        # Store feature columns for prediction validation
        self.feature_columns = list(X.columns)
        
        # Train the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X)
        train_r2 = r2_score(y, train_predictions)
        train_mae = mean_absolute_error(y, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y, train_predictions))
        
        self.model_metrics = {
            'train_r2': train_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse
        }
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        logger.info(f"Model training completed:")
        logger.info(f"  R²: {self.model_metrics['train_r2']:.4f}")
        logger.info(f"  MAE: {self.model_metrics['train_mae']:.4f}")
        logger.info(f"  RMSE: {self.model_metrics['train_rmse']:.4f}")
        
        return self.model_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted lap times
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Ensure features match training data
        if list(X.columns) != self.feature_columns:
            missing_features = set(self.feature_columns) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_columns)
            
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            if extra_features:
                logger.warning(f"Extra features will be ignored: {extra_features}")
                X = X[self.feature_columns]
        predictions = self.model.predict(X)
        # Only log for large batches to reduce verbosity
        if len(predictions) > 100:
            logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            X: Test features DataFrame
            y: Test target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info(f"Evaluating model on {len(X)} test samples")
        
        predictions = self.predict(X)
        
        # Calculate metrics with standardized keys
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        metrics = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mean_lap_time': float(y.mean()),
            'std_lap_time': float(y.std())
        }
        
        # Also store with test_ prefix for internal tracking
        test_metrics = {
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse
        }
        
        self.model_metrics.update(test_metrics)
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"  R²: {metrics['r2_score']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if not self.feature_importance:
            raise RuntimeError("Feature importance not available")
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top N features
        top_features = dict(sorted_features[:top_n])
        
        # Create DataFrame
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in top_features.items()
        ])
        
        return importance_df
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model (optional)
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save model that hasn't been fitted")
        
        if filepath is None:
            filepath = config.TIRE_MODEL_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'training_metrics': self.model_metrics,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: Optional[str] = None) -> bool:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from (optional)
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        if filepath is None:
            filepath = config.TIRE_MODEL_PATH
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.model_metrics = model_data['training_metrics']
            self.feature_importance = model_data['feature_importance']
            self.is_fitted = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            return False
    
    def predict_single_lap(self, tire_age: int, compound: str, track_temp: float = 30.0, 
                          air_temp: float = 25.0) -> float:
        """
        Predict lap time for a single lap with given conditions
        
        Args:
            tire_age: Age of tires in laps
            compound: Tire compound ('SOFT', 'MEDIUM', 'HARD')
            track_temp: Track temperature in Celsius
            air_temp: Air temperature in Celsius
            
        Returns:
            Predicted lap time in seconds
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'tire_age': [tire_age],
            'tire_compound': [compound],
            'track_temp': [track_temp],
            'air_temp': [air_temp],
            'stint_lap': [1],  # Default values
            'driver_stint_experience': [0],
            'fuel_load_normalized': [0.5]
        })
        
        # Add any missing features with defaults
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        # Ensure column order matches training
        features = features[self.feature_columns]
        
        prediction = self.predict(features)
        return float(prediction[0])
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary containing model information including metrics and features
        """
        if not self.is_fitted:
            return {
                'fitted': False,
                'n_features': 0,
                'metrics': {}
            }
        
        return {
            'fitted': True,
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns.copy(),
            'metrics': self.model_metrics.copy(),
            'feature_importance': self.feature_importance.copy()
        }
    
    def predict_tire_degradation(self, compound: str, tire_age: int, circuit: str = None, 
                               track_temp: float = 30.0, air_temp: float = 25.0) -> float:
        """
        Predict tire degradation (lap time) for specific conditions
        
        Args:
            compound: Tire compound ('SOFT', 'MEDIUM', 'HARD')
            tire_age: Age of tires in laps
            circuit: Circuit name (optional, used for circuit-specific adjustments)
            track_temp: Track temperature in Celsius
            air_temp: Air temperature in Celsius
            
        Returns:
            Predicted lap time in seconds
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Create feature DataFrame with proper compound encoding
        features = pd.DataFrame({
            'tire_age': [tire_age],
            'tire_compound': [compound],
            'track_temp': [track_temp],
            'air_temp': [air_temp],
            'stint_lap': [tire_age],  # Use tire_age as stint lap
            'driver_stint_experience': [min(tire_age, 10)],  # Cap at 10
            'fuel_load_normalized': [max(0.3, 1.0 - (tire_age * 0.02))]  # Decreasing fuel load
        })
        
        # Add any missing features with sensible defaults
        for col in self.feature_columns:
            if col not in features.columns:
                if 'compound_' in col:
                    # Handle one-hot encoded tire compounds
                    compound_name = col.replace('compound_', '').upper()
                    features[col] = [1 if compound.upper() == compound_name else 0]
                elif 'position' in col.lower():
                    features[col] = [10]  # Mid-field position
                elif 'weather' in col.lower():
                    features[col] = [0]  # Clear weather
                else:
                    features[col] = [0]  # Default to 0
        
        # Ensure column order matches training
        features = features[self.feature_columns]
        
        prediction = self.predict(features)
        return float(prediction[0])

    @property
    def performance_summary(self) -> Dict[str, float]:
        """
        Get summary of model performance metrics
        
        Returns:
            Dictionary with key performance metrics
        """
        if not self.model_metrics:
            return {}
        
        summary = {}
        if 'train_r2' in self.model_metrics:
            summary['training_r2'] = self.model_metrics['train_r2']
        if 'test_r2' in self.model_metrics:
            summary['test_r2'] = self.model_metrics['test_r2']
        if 'train_mae' in self.model_metrics:
            summary['training_mae'] = self.model_metrics['train_mae']
        if 'test_mae' in self.model_metrics:
            summary['test_mae'] = self.model_metrics['test_mae']
        
        return summary
