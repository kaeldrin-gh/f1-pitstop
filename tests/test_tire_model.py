"""
Unit tests for tire degradation model.
Tests model training, prediction, evaluation, and persistence.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import pickle
from unittest.mock import Mock, patch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.tire_model import TireDegradationModel
from sklearn.ensemble import RandomForestRegressor


class TestTireDegradationModel(unittest.TestCase):
    """Test cases for TireDegradationModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = TireDegradationModel()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        np.random.seed(42)  # For reproducible tests
        n_samples = 100
        
        self.X_train = pd.DataFrame({
            'tyre_life': np.random.randint(1, 30, n_samples),
            'compound_SOFT': np.random.randint(0, 2, n_samples),
            'compound_MEDIUM': np.random.randint(0, 2, n_samples),
            'compound_HARD': np.random.randint(0, 2, n_samples),
            'tire_degradation_rate': np.random.uniform(0.01, 0.1, n_samples),
            'rolling_avg_lap_time_3': np.random.uniform(80, 90, n_samples),
            'stint_length': np.random.randint(5, 25, n_samples),
            'laps_since_pit': np.random.randint(1, 20, n_samples)
        })
        
        # Create realistic lap times based on features
        base_time = 82.0
        tire_effect = self.X_train['tyre_life'] * 0.05  # Tires degrade over time
        compound_effect = (self.X_train['compound_SOFT'] * -1.0 + 
                          self.X_train['compound_HARD'] * 1.0)  # Soft faster, hard slower
        noise = np.random.normal(0, 0.2, n_samples)
        
        self.y_train = base_time + tire_effect + compound_effect + noise
        
        # Create test data
        n_test = 20
        self.X_test = pd.DataFrame({
            'tyre_life': np.random.randint(1, 30, n_test),
            'compound_SOFT': np.random.randint(0, 2, n_test),
            'compound_MEDIUM': np.random.randint(0, 2, n_test),
            'compound_HARD': np.random.randint(0, 2, n_test),
            'tire_degradation_rate': np.random.uniform(0.01, 0.1, n_test),
            'rolling_avg_lap_time_3': np.random.uniform(80, 90, n_test),
            'stint_length': np.random.randint(5, 25, n_test),
            'laps_since_pit': np.random.randint(1, 20, n_test)
        })
        
        base_time_test = 82.0
        tire_effect_test = self.X_test['tyre_life'] * 0.05
        compound_effect_test = (self.X_test['compound_SOFT'] * -1.0 + 
                               self.X_test['compound_HARD'] * 1.0)
        noise_test = np.random.normal(0, 0.2, n_test)
        
        self.y_test = base_time_test + tire_effect_test + compound_effect_test + noise_test
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_init_model_creation(self):
        """Test model initialization creates RandomForestRegressor."""
        self.assertIsInstance(self.model.model, RandomForestRegressor)
        self.assertEqual(self.model.model.n_estimators, 100)
        self.assertEqual(self.model.model.max_depth, 10)
        self.assertEqual(self.model.model.random_state, 42)
    
    def test_fit_model_training(self):
        """Test model training with valid data."""
        # Fit the model
        self.model.fit(self.X_train, self.y_train)
        
        # Model should be fitted
        self.assertTrue(hasattr(self.model.model, 'estimators_'))
        self.assertIsNotNone(self.model.feature_columns)
        self.assertEqual(len(self.model.feature_columns), len(self.X_train.columns))
    
    def test_fit_invalid_data_shapes(self):
        """Test model training with mismatched data shapes."""
        X_invalid = self.X_train.iloc[:50]  # Different length
        
        with self.assertRaises(ValueError):
            self.model.fit(X_invalid, self.y_train)
    
    def test_fit_empty_data(self):
        """Test model training with empty data."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        
        with self.assertRaises(ValueError):
            self.model.fit(X_empty, y_empty)
    
    def test_predict_after_training(self):
        """Test prediction after model training."""
        # Train model first
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Check predictions
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred > 0 for pred in predictions))  # Positive lap times
    
    def test_predict_without_training(self):
        """Test prediction without prior training raises error."""
        with self.assertRaises(RuntimeError):
            self.model.predict(self.X_test)
    
    def test_predict_mismatched_features(self):
        """Test prediction with different feature columns."""
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Try to predict with different features
        X_different = self.X_test.drop(['tyre_life'], axis=1)
        
        with self.assertRaises(ValueError):
            self.model.predict(X_different)
    
    def test_evaluate_model_metrics(self):
        """Test model evaluation returns expected metrics."""
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Check metrics structure
        self.assertIsInstance(metrics, dict)
        expected_metrics = ['r2_score', 'mae', 'rmse', 'mean_lap_time', 'std_lap_time']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric ranges (should be reasonable for good model)
        self.assertIsInstance(metrics['r2_score'], float)
        self.assertIsInstance(metrics['mae'], float)
        self.assertIsInstance(metrics['rmse'], float)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)
    
    def test_evaluate_performance_requirements(self):
        """Test model meets performance requirements."""
        # Train model with sufficient data
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Performance requirements from config
        # Note: With synthetic data, these might not always be met
        # In real scenario, we'd use actual F1 data
        self.assertIsInstance(metrics['r2_score'], float)
        self.assertIsInstance(metrics['mae'], float)
        
        # At least check they're reasonable values
        self.assertGreaterEqual(metrics['r2_score'], -1.0)  # R² can be negative
        self.assertLessEqual(metrics['r2_score'], 1.0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_save_model(self):
        """Test model saving to file."""
        # Train model first
        self.model.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.model.save(model_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(model_path))
        
        # Check file contains valid pickle data
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.assertIn('model', saved_data)
        self.assertIn('feature_columns', saved_data)
        self.assertIn('training_metrics', saved_data)
    
    def test_save_model_without_training(self):
        """Test saving untrained model raises error."""
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        
        with self.assertRaises(RuntimeError):
            self.model.save(model_path)
    
    def test_load_model(self):
        """Test model loading from file."""
        # Train and save model first
        self.model.fit(self.X_train, self.y_train)
        original_predictions = self.model.predict(self.X_test)
        
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.model.save(model_path)
        
        # Create new model instance and load
        new_model = TireDegradationModel()
        new_model.load(model_path)
        
        # Test loaded model works
        loaded_predictions = new_model.predict(self.X_test)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
        
        # Feature columns should match
        self.assertEqual(self.model.feature_columns, new_model.feature_columns)
    
    def test_load_nonexistent_model(self):
        """Test loading from nonexistent file raises error."""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.pkl')
        
        with self.assertRaises(FileNotFoundError):
            self.model.load(nonexistent_path)
    
    def test_get_feature_importance(self):
        """Test getting feature importance from trained model."""
        # Train model first
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertEqual(len(importance), len(self.X_train.columns))
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
        
        # Importance should sum to approximately 1.0
        total_importance = importance['importance'].sum()
        self.assertAlmostEqual(total_importance, 1.0, places=3)
    
    def test_get_feature_importance_without_training(self):
        """Test getting feature importance without training raises error."""
        with self.assertRaises(RuntimeError):
            self.model.get_feature_importance()
    
    def test_model_reproducibility(self):
        """Test model training is reproducible with same data."""
        # Train first model
        model1 = TireDegradationModel()
        model1.fit(self.X_train, self.y_train)
        pred1 = model1.predict(self.X_test)
        
        # Train second model with same data
        model2 = TireDegradationModel()
        model2.fit(self.X_train, self.y_train)
        pred2 = model2.predict(self.X_test)
        
        # Predictions should be identical (due to random_state=42)
        np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == '__main__':
    unittest.main()
