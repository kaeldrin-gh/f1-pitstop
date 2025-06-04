"""
Unit tests for utility functions.
Tests data validation, performance monitoring, and database operations.
"""

import unittest
import tempfile
import os
import sqlite3
import pandas as pd
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.helpers import (
    validate_lap_time,
    validate_tire_stint,
    validate_data_completeness,
    log_performance_metrics,
    create_database_backup,
    get_system_info,
    format_time_delta,
    calculate_percentile_stats
)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'lap_time': [80.5, 81.2, 82.1, 79.8, 85.2],
            'compound': ['SOFT', 'SOFT', 'MEDIUM', 'HARD', 'MEDIUM'],
            'tyre_life': [1, 2, 1, 1, 5],
            'driver': ['HAM', 'HAM', 'VER', 'HAM', 'VER'],
            'stint': [1, 1, 2, 3, 2]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    def test_validate_lap_time_valid_times(self):
        """Test lap time validation for valid times."""
        valid_times = [75.5, 80.0, 120.5, 180.0]
        
        for lap_time in valid_times:
            self.assertTrue(validate_lap_time(lap_time))
    
    def test_validate_lap_time_invalid_times(self):
        """Test lap time validation for invalid times."""
        invalid_times = [50.0, 59.9, 180.1, 250.0, -5.0, 0]
        
        for lap_time in invalid_times:
            self.assertFalse(validate_lap_time(lap_time))
    
    def test_validate_lap_time_edge_cases(self):
        """Test lap time validation for edge cases."""
        # Boundary values
        self.assertTrue(validate_lap_time(60.0))   # Minimum valid
        self.assertTrue(validate_lap_time(180.0))  # Maximum valid
        self.assertFalse(validate_lap_time(59.999))  # Just below minimum
        self.assertFalse(validate_lap_time(180.001))  # Just above maximum
    
    def test_validate_tire_stint_valid_stints(self):
        """Test tire stint validation for valid values."""
        valid_stints = [1, 10, 25, 50]
        
        for stint_length in valid_stints:
            self.assertTrue(validate_tire_stint(stint_length))
    
    def test_validate_tire_stint_invalid_stints(self):
        """Test tire stint validation for invalid values."""
        invalid_stints = [0, -1, 51, 100]
        
        for stint_length in invalid_stints:
            self.assertFalse(validate_tire_stint(stint_length))
    
    def test_validate_tire_stint_edge_cases(self):
        """Test tire stint validation for edge cases."""
        self.assertTrue(validate_tire_stint(1))    # Minimum valid
        self.assertTrue(validate_tire_stint(50))   # Maximum valid
        self.assertFalse(validate_tire_stint(0))   # Below minimum
        self.assertFalse(validate_tire_stint(51))  # Above maximum
    
    def test_validate_data_completeness_complete_data(self):
        """Test data completeness validation for complete data."""
        result = validate_data_completeness(self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('is_valid', result)
        self.assertIn('missing_data_report', result)
        self.assertIn('completeness_score', result)
        
        # Complete data should be valid
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['completeness_score'], 1.0)
    
    def test_validate_data_completeness_missing_data(self):
        """Test data completeness validation with missing data."""
        incomplete_data = self.sample_data.copy()
        incomplete_data.loc[0, 'compound'] = None
        incomplete_data.loc[1, 'lap_time'] = None
        
        result = validate_data_completeness(incomplete_data)
        
        self.assertFalse(result['is_valid'])
        self.assertLess(result['completeness_score'], 1.0)
        self.assertGreater(len(result['missing_data_report']), 0)
    
    def test_validate_data_completeness_empty_data(self):
        """Test data completeness validation with empty data."""
        empty_data = pd.DataFrame()
        
        result = validate_data_completeness(empty_data)
        
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['completeness_score'], 0.0)
    
    def test_log_performance_metrics_basic(self):
        """Test basic performance metrics logging."""
        metrics = {
            'r2_score': 0.85,
            'mae': 0.45,
            'rmse': 0.62,
            'training_time': 125.5
        }
        
        log_file = os.path.join(self.temp_dir, 'performance.log')
        
        # Should not raise an exception
        log_performance_metrics(metrics, log_file)
        
        # Log file should be created
        self.assertTrue(os.path.exists(log_file))
        
        # Check log content
        with open(log_file, 'r') as f:
            content = f.read()
        
        self.assertIn('r2_score', content)
        self.assertIn('0.85', content)
        self.assertIn('mae', content)
        self.assertIn('0.45', content)
    
    def test_log_performance_metrics_append_mode(self):
        """Test performance metrics logging in append mode."""
        log_file = os.path.join(self.temp_dir, 'performance.log')
        
        # Log first set of metrics
        metrics1 = {'accuracy': 0.8}
        log_performance_metrics(metrics1, log_file)
        
        # Log second set of metrics
        metrics2 = {'accuracy': 0.85}
        log_performance_metrics(metrics2, log_file)
        
        # Both should be in the file
        with open(log_file, 'r') as f:
            content = f.read()
        
        self.assertIn('0.8', content)
        self.assertIn('0.85', content)
        
        # Should have multiple log entries
        log_entries = content.count('Performance Metrics')
        self.assertGreaterEqual(log_entries, 2)
    
    def test_create_database_backup_success(self):
        """Test successful database backup creation."""
        # Create a test database
        db_path = os.path.join(self.temp_dir, 'test.db')
        conn = sqlite3.connect(db_path)
        
        # Add some test data
        conn.execute('CREATE TABLE test (id INTEGER, value TEXT)')
        conn.execute('INSERT INTO test VALUES (1, "test_value")')
        conn.commit()
        conn.close()
        
        # Create backup
        backup_path = os.path.join(self.temp_dir, 'backup.db')
        result = create_database_backup(db_path, backup_path)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(backup_path))
        
        # Verify backup contains same data
        backup_conn = sqlite3.connect(backup_path)
        cursor = backup_conn.execute('SELECT * FROM test')
        rows = cursor.fetchall()
        backup_conn.close()
        
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], (1, 'test_value'))
    
    def test_create_database_backup_source_not_exists(self):
        """Test database backup when source doesn't exist."""
        nonexistent_db = os.path.join(self.temp_dir, 'nonexistent.db')
        backup_path = os.path.join(self.temp_dir, 'backup.db')
        
        result = create_database_backup(nonexistent_db, backup_path)
        
        self.assertFalse(result)
        self.assertFalse(os.path.exists(backup_path))
    
    def test_get_system_info_structure(self):
        """Test system info returns expected structure."""
        info = get_system_info()
        
        self.assertIsInstance(info, dict)
        expected_keys = [
            'python_version',
            'platform',
            'cpu_count',
            'memory_total',
            'memory_available',
            'disk_free_space'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
    
    def test_get_system_info_values(self):
        """Test system info returns reasonable values."""
        info = get_system_info()
        
        # Check data types and reasonable ranges
        self.assertIsInstance(info['python_version'], str)
        self.assertIsInstance(info['platform'], str)
        self.assertIsInstance(info['cpu_count'], int)
        self.assertGreater(info['cpu_count'], 0)
        
        self.assertIsInstance(info['memory_total'], (int, float))
        self.assertGreater(info['memory_total'], 0)
        
        self.assertIsInstance(info['memory_available'], (int, float))
        self.assertGreater(info['memory_available'], 0)
        self.assertLessEqual(info['memory_available'], info['memory_total'])
    
    def test_format_time_delta_seconds(self):
        """Test time delta formatting for seconds."""
        # Test various time intervals
        test_cases = [
            (30.5, "30.50s"),
            (1.0, "1.00s"),
            (59.99, "59.99s")
        ]
        
        for seconds, expected in test_cases:
            result = format_time_delta(seconds)
            self.assertEqual(result, expected)
    
    def test_format_time_delta_minutes(self):
        """Test time delta formatting for minutes."""
        test_cases = [
            (60.0, "1m 0.00s"),
            (90.5, "1m 30.50s"),
            (125.25, "2m 5.25s")
        ]
        
        for seconds, expected in test_cases:
            result = format_time_delta(seconds)
            self.assertEqual(result, expected)
    
    def test_format_time_delta_hours(self):
        """Test time delta formatting for hours."""
        test_cases = [
            (3600.0, "1h 0m 0.00s"),
            (3665.5, "1h 1m 5.50s"),
            (7325.25, "2h 2m 5.25s")
        ]
        
        for seconds, expected in test_cases:
            result = format_time_delta(seconds)
            self.assertEqual(result, expected)
    
    def test_format_time_delta_edge_cases(self):
        """Test time delta formatting edge cases."""
        # Zero time
        self.assertEqual(format_time_delta(0), "0.00s")
        
        # Very small time
        self.assertEqual(format_time_delta(0.001), "0.00s")
        
        # Large time
        result = format_time_delta(25 * 3600 + 30 * 60 + 45.5)
        self.assertIn("25h", result)
        self.assertIn("30m", result)
        self.assertIn("45.50s", result)
    
    def test_calculate_percentile_stats_basic(self):
        """Test basic percentile statistics calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        stats = calculate_percentile_stats(data)
        
        self.assertIsInstance(stats, dict)
        expected_keys = ['min', 'q25', 'median', 'q75', 'max', 'iqr', 'mean', 'std']
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check some specific values
        self.assertEqual(stats['min'], 1)
        self.assertEqual(stats['max'], 10)
        self.assertEqual(stats['median'], 5.5)
        self.assertEqual(stats['mean'], 5.5)
    
    def test_calculate_percentile_stats_empty_data(self):
        """Test percentile statistics with empty data."""
        stats = calculate_percentile_stats([])
        
        # All values should be NaN or None for empty data
        for key, value in stats.items():
            self.assertTrue(pd.isna(value) or value is None)
    
    def test_calculate_percentile_stats_single_value(self):
        """Test percentile statistics with single value."""
        data = [42.5]
        
        stats = calculate_percentile_stats(data)
        
        # All percentiles should equal the single value
        self.assertEqual(stats['min'], 42.5)
        self.assertEqual(stats['max'], 42.5)
        self.assertEqual(stats['median'], 42.5)
        self.assertEqual(stats['q25'], 42.5)
        self.assertEqual(stats['q75'], 42.5)
        self.assertEqual(stats['mean'], 42.5)
        self.assertEqual(stats['std'], 0.0)
        self.assertEqual(stats['iqr'], 0.0)
    
    def test_calculate_percentile_stats_with_outliers(self):
        """Test percentile statistics with outliers."""
        # Data with clear outliers
        data = [1, 2, 3, 4, 5, 100, 200]  # 100, 200 are outliers
        
        stats = calculate_percentile_stats(data)
        
        # Median should be less affected by outliers than mean
        self.assertEqual(stats['median'], 4)
        self.assertGreater(stats['mean'], stats['median'])  # Mean pulled up by outliers
        
        # IQR should be reasonable
        self.assertEqual(stats['iqr'], stats['q75'] - stats['q25'])
        self.assertGreater(stats['iqr'], 0)


if __name__ == '__main__':
    unittest.main()
