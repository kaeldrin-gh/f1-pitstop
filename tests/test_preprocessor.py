"""
Unit tests for data preprocessor module.
Tests data cleaning, validation, and feature engineering.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessor import F1DataPreprocessor
from config import TIRE_COMPOUNDS


class TestF1DataPreprocessor(unittest.TestCase):
    """Test cases for F1DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = F1DataPreprocessor()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'lap_time': [80.5, 81.2, 82.1, 150.0, 79.8, 85.2, 81.5],  # One outlier (150.0)
            'compound': ['SOFT', 'SOFT', 'MEDIUM', 'SOFT', 'HARD', 'MEDIUM', 'SOFT'],
            'tyre_life': [1, 2, 1, 3, 1, 5, 4],
            'stint': [1, 1, 2, 1, 3, 2, 1],
            'driver': ['HAM', 'HAM', 'HAM', 'VER', 'HAM', 'VER', 'LEC'],
            'track_status': ['1', '1', '1', '4', '1', '1', '1'],  # One safety car
            'lap_number': [1, 2, 15, 3, 30, 16, 4],
            'year': [2023, 2023, 2023, 2023, 2023, 2023, 2023],
            'circuit': ['bahrain', 'bahrain', 'bahrain', 'bahrain', 'bahrain', 'bahrain', 'bahrain']
        })
    
    def test_validate_data_structure_valid(self):
        """Test validation passes for valid data structure."""
        result = self.preprocessor._validate_data_structure(self.sample_data)
        self.assertTrue(result)
    
    def test_validate_data_structure_missing_columns(self):
        """Test validation fails for missing required columns."""
        invalid_data = self.sample_data.drop(['lap_time'], axis=1)
        result = self.preprocessor._validate_data_structure(invalid_data)
        self.assertFalse(result)
    
    def test_validate_data_structure_empty(self):
        """Test validation fails for empty dataframe."""
        empty_data = pd.DataFrame()
        result = self.preprocessor._validate_data_structure(empty_data)
        self.assertFalse(result)
    
    def test_remove_invalid_laps_lap_time_range(self):
        """Test removal of laps outside valid time range."""
        # Add some invalid lap times
        test_data = self.sample_data.copy()
        test_data.loc[len(test_data)] = {
            'lap_time': 50.0,  # Too fast
            'compound': 'SOFT',
            'tyre_life': 1,
            'stint': 1,
            'driver': 'HAM',
            'track_status': '1',
            'lap_number': 5,
            'year': 2023,
            'circuit': 'bahrain'
        }
        test_data.loc[len(test_data)] = {
            'lap_time': 200.0,  # Too slow
            'compound': 'SOFT',
            'tyre_life': 1,
            'stint': 1,
            'driver': 'HAM',
            'track_status': '1',
            'lap_number': 6,
            'year': 2023,
            'circuit': 'bahrain'
        }
        
        cleaned_data = self.preprocessor._remove_invalid_laps(test_data)
        
        # Should remove laps outside 60-180s range
        self.assertTrue(all(cleaned_data['lap_time'] >= 60))
        self.assertTrue(all(cleaned_data['lap_time'] <= 180))
    
    def test_remove_invalid_laps_safety_car(self):
        """Test removal of safety car laps."""
        cleaned_data = self.preprocessor._remove_invalid_laps(self.sample_data)
        
        # Should remove laps with track_status '4' (safety car)
        self.assertTrue(all(cleaned_data['track_status'] != '4'))
    
    def test_remove_invalid_laps_missing_data(self):
        """Test removal of laps with missing required data."""
        test_data = self.sample_data.copy()
        test_data.loc[0, 'compound'] = None
        test_data.loc[1, 'lap_time'] = np.nan
        
        cleaned_data = self.preprocessor._remove_invalid_laps(test_data)
        
        # Should remove rows with missing compound or lap_time
        self.assertFalse(cleaned_data['compound'].isna().any())
        self.assertFalse(cleaned_data['lap_time'].isna().any())
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method."""
        cleaned_data = self.preprocessor._remove_outliers(self.sample_data)
        
        # The outlier (150.0) should be removed
        self.assertNotIn(150.0, cleaned_data['lap_time'].values)
        self.assertLess(len(cleaned_data), len(self.sample_data))
    
    def test_engineer_features_tire_degradation(self):
        """Test tire degradation feature engineering."""
        processed_data = self.preprocessor._engineer_features(self.sample_data)
        
        # Should have tire degradation feature
        self.assertIn('tire_degradation_rate', processed_data.columns)
        self.assertTrue(all(processed_data['tire_degradation_rate'] >= 0))
    
    def test_engineer_features_rolling_averages(self):
        """Test rolling average feature engineering."""
        processed_data = self.preprocessor._engineer_features(self.sample_data)
        
        # Should have rolling average features
        self.assertIn('rolling_avg_lap_time_3', processed_data.columns)
        self.assertIn('rolling_avg_lap_time_5', processed_data.columns)
    
    def test_engineer_features_pit_context(self):
        """Test pit context feature engineering."""
        processed_data = self.preprocessor._engineer_features(self.sample_data)
        
        # Should have pit context features
        self.assertIn('laps_since_pit', processed_data.columns)
        self.assertIn('stint_length', processed_data.columns)
    
    def test_engineer_features_compound_encoding(self):
        """Test tire compound encoding."""
        processed_data = self.preprocessor._engineer_features(self.sample_data)
          # Should have encoded compound features
        expected_columns = [f'compound_{compound}' for compound in TIRE_COMPOUNDS]
        for col in expected_columns:
            self.assertIn(col, processed_data.columns)
    
    def test_split_train_test_chronological(self):
        """Test chronological train/test split."""
        # Create data with different dates
        test_data = self.sample_data.copy()
        test_data['lap_number'] = range(1, len(test_data) + 1)  # Sequential laps
        
        train_data, test_data_split = self.preprocessor._split_train_test(test_data)
        
        # Test set should be last 20% chronologically
        expected_test_size = int(len(test_data) * 0.2)
        self.assertEqual(len(test_data_split), expected_test_size)
        
        # Test set should have higher lap numbers (later in time)
        if len(test_data_split) > 0 and len(train_data) > 0:
            min_test_lap = test_data_split['lap_number'].min()
            max_train_lap = train_data['lap_number'].max()
            self.assertGreaterEqual(min_test_lap, max_train_lap)
    
    def test_process_data_complete_pipeline(self):
        """Test complete data processing pipeline."""
        train_data, test_data = self.preprocessor.process_data(self.sample_data)
        
        # Should return both train and test sets
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        
        # Should have engineered features
        required_features = [
            'tire_degradation_rate',
            'rolling_avg_lap_time_3',
            'laps_since_pit',
            'stint_length'
        ]
        
        for feature in required_features:
            self.assertIn(feature, train_data.columns)
            if len(test_data) > 0:
                self.assertIn(feature, test_data.columns)
    
    def test_process_data_validation_failure(self):
        """Test processing with invalid data structure."""
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        with self.assertRaises(ValueError):
            self.preprocessor.process_data(invalid_data)
    
    def test_get_feature_columns(self):
        """Test getting list of feature columns."""
        # Process data first to generate features
        train_data, _ = self.preprocessor.process_data(self.sample_data)
        
        features = self.preprocessor.get_feature_columns()
        
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        
        # Should not include target or identifier columns
        excluded_columns = ['lap_time', 'driver', 'year', 'circuit']
        for col in excluded_columns:
            if col in features:
                self.fail(f"Feature list should not include {col}")
    
    def test_validate_processed_data_valid(self):
        """Test validation of processed data."""
        train_data, test_data = self.preprocessor.process_data(self.sample_data)
        
        # Should pass validation
        train_valid = self.preprocessor._validate_processed_data(train_data)
        self.assertTrue(train_valid)
        
        if len(test_data) > 0:
            test_valid = self.preprocessor._validate_processed_data(test_data)
            self.assertTrue(test_valid)
    
    def test_validate_processed_data_invalid_lap_times(self):
        """Test validation fails for invalid lap times."""
        invalid_data = self.sample_data.copy()
        invalid_data['lap_time'] = [50.0] * len(invalid_data)  # All too fast
        
        result = self.preprocessor._validate_processed_data(invalid_data)
        self.assertFalse(result)
    
    def test_validate_processed_data_excessive_stint_length(self):
        """Test validation fails for excessive stint lengths."""
        invalid_data = self.sample_data.copy()
        invalid_data['tyre_life'] = [60] * len(invalid_data)  # All too long
        
        result = self.preprocessor._validate_processed_data(invalid_data)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
