"""
Unit tests for data extractor module.
Tests FastF1 data extraction, database operations, and validation.
"""

import unittest
import tempfile
import os
import sqlite3
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_extractor import F1DataExtractor
from config import CIRCUITS, ANALYSIS_YEARS


class TestF1DataExtractor(unittest.TestCase):
    """Test cases for F1DataExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_f1_data.db')
        self.extractor = F1DataExtractor(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            # Close database connections
            if hasattr(self.extractor, 'session'):
                self.extractor.session.close()
            if hasattr(self.extractor, 'engine'):
                self.extractor.engine.dispose()
        except:
            pass
            
        # Remove database file
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass  # File still in use, skip cleanup
        try:
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_init_database(self):
        """Test database initialization creates required tables."""
        # Database should be created during __init__
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check sessions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        self.assertIsNotNone(cursor.fetchone())
          # Check laps table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='laps'")
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()
    
    def test_validate_session_data_valid(self):
        """Test validation passes for valid session data."""
        # Create mock session with required attributes
        mock_session = Mock()
        mock_session.laps = pd.DataFrame({
            'LapTime': pd.to_timedelta(['0:01:20.123', '0:01:21.456', '0:01:22.789']),
            'Compound': ['SOFT', 'MEDIUM', 'HARD'],
            'Driver': ['HAM', 'VER', 'LEC'],
            'LapNumber': [1, 2, 3]        })
        
        result = self.extractor._validate_session_data(mock_session)
        self.assertTrue(result)
    
    def test_validate_session_data_invalid_empty(self):
        """Test validation fails for empty session data."""
        mock_session = Mock()
        mock_session.laps = pd.DataFrame()
        
        result = self.extractor._validate_session_data(mock_session)
        self.assertFalse(result)
    
    def test_validate_session_data_invalid_missing_columns(self):
        """Test validation fails for missing required columns."""
        mock_session = Mock()
        mock_session.laps = pd.DataFrame({
            'LapTime': pd.to_timedelta(['0:01:20.123', '0:01:21.456']),
            # Missing Compound, Driver, LapNumber
        })
        
        result = self.extractor._validate_session_data(mock_session)
        self.assertFalse(result)
    
    @patch('fastf1.get_session')
    def test_extract_session_data_success(self, mock_get_session):
        """Test successful session data extraction."""
        # Mock successful FastF1 session
        mock_session = Mock()
        mock_session.load.return_value = None
        mock_session.laps = pd.DataFrame({
            'LapTime': pd.to_timedelta(['0:01:20.123', '0:01:21.456']),
            'Compound': ['SOFT', 'MEDIUM'],
            'Driver': ['HAM', 'VER'],
            'LapNumber': [1, 2],
            'Stint': [1, 1],
            'TyreLife': [1, 2],
            'TrackStatus': ['1', '1'],
            'IsPersonalBest': [True, False]
        })
        mock_session.info = {
            'Meeting': {'Name': 'Test GP'},
            'Type': 'Race',
            'Date': pd.Timestamp('2023-01-01')        }
        mock_get_session.return_value = mock_session
        
        result = self.extractor.extract_session_data(2023, 'bahrain', 'Race')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('lap_time', result.columns)
        self.assertIn('compound', result.columns)
    
    @patch('fastf1.get_session')
    def test_extract_session_data_failure(self, mock_get_session):
        """Test handling of FastF1 extraction failure."""
        mock_get_session.side_effect = Exception("API Error")
        
        result = self.extractor.extract_session_data(2023, 'bahrain', 'Race')
        self.assertIsNone(result)
    
    def test_save_session_data(self):
        """Test saving session data to database."""
        # Create test data
        test_data = pd.DataFrame({
            'year': [2023, 2023],
            'circuit': ['bahrain', 'bahrain'],
            'session_type': ['Race', 'Race'],
            'driver': ['HAM', 'VER'],
            'lap_number': [1, 2],
            'lap_time': [80.123, 81.456],
            'compound': ['SOFT', 'MEDIUM'],
            'stint': [1, 1],
            'tyre_life': [1, 2],
            'track_status': ['1', '1'],
            'is_personal_best': [True, False]
        })
        
        # Save data
        self.extractor.save_session_data(test_data, 2023, 'bahrain', 'Race')
        
        # Verify data was saved
        conn = sqlite3.connect(self.db_path)
        saved_data = pd.read_sql('SELECT * FROM laps', conn)
        conn.close()
        
        self.assertEqual(len(saved_data), 2)
        self.assertEqual(saved_data.iloc[0]['driver'], 'HAM')
    
    def test_load_all_data(self):
        """Test loading all data from database."""
        # First save some test data
        test_data = pd.DataFrame({
            'year': [2023],
            'circuit': ['bahrain'],
            'session_type': ['Race'],
            'driver': ['HAM'],
            'lap_number': [1],
            'lap_time': [80.123],
            'compound': ['SOFT'],
            'stint': [1],
            'tyre_life': [1],
            'track_status': ['1'],
            'is_personal_best': [True]
        })
        
        self.extractor.save_session_data(test_data, 2023, 'bahrain', 'Race')
        
        # Load all data
        result = self.extractor.load_all_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['driver'], 'HAM')
    
    def test_get_available_sessions(self):
        """Test getting available sessions from database."""
        # Save test data for multiple sessions
        for i, circuit in enumerate(['bahrain', 'jeddah']):
            test_data = pd.DataFrame({
                'year': [2023],
                'circuit': [circuit],
                'session_type': ['Race'],
                'driver': ['HAM'],
                'lap_number': [1],
                'lap_time': [80.123],
                'compound': ['SOFT'],
                'stint': [1],
                'tyre_life': [1],
                'track_status': ['1'],
                'is_personal_best': [True]
            })
            self.extractor.save_session_data(test_data, 2023, circuit, 'Race')
        
        sessions = self.extractor.get_available_sessions()
        
        self.assertIsInstance(sessions, pd.DataFrame)
        self.assertEqual(len(sessions), 2)
        self.assertIn('year', sessions.columns)
        self.assertIn('circuit', sessions.columns)
    
    @patch('fastf1.get_session')
    def test_extract_all_data_partial_success(self, mock_get_session):
        """Test extracting all data with some failures."""
        # Mock some successful and some failed extractions
        def mock_session_side_effect(year, circuit, session_type):
            if circuit == 'bahrain':
                mock_session = Mock()
                mock_session.load.return_value = None
                mock_session.laps = pd.DataFrame({
                    'LapTime': pd.to_timedelta(['0:01:20.123']),
                    'Compound': ['SOFT'],
                    'Driver': ['HAM'],
                    'LapNumber': [1],
                    'Stint': [1],
                    'TyreLife': [1],
                    'TrackStatus': ['1'],
                    'IsPersonalBest': [True]
                })
                mock_session.info = {
                    'Meeting': {'Name': 'Bahrain GP'},
                    'Type': 'Race',
                    'Date': pd.Timestamp('2023-01-01')
                }
                return mock_session
            else:
                raise Exception("API Error")
        
        mock_get_session.side_effect = mock_session_side_effect
        
        # Extract data for a subset
        circuits_to_test = ['bahrain', 'jeddah']
        years_to_test = [2023]
        
        success_count = 0
        for year in years_to_test:
            for circuit in circuits_to_test:
                result = self.extractor.extract_session_data(year, circuit, 'Race')
                if result is not None:
                    success_count += 1
        
        # Should have at least one success (bahrain)
        self.assertGreaterEqual(success_count, 1)


if __name__ == '__main__':
    unittest.main()
