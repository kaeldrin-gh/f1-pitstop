"""
Unit tests for strategy optimizer.
Tests pit stop strategy optimization, race simulation, and confidence scoring.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.strategy_optimizer import PitStopOptimizer
from src.models.tire_model import TireDegradationModel
from config import TIRE_COMPOUNDS, TIRE_DEGRADATION_RATES


class TestPitStopOptimizer(unittest.TestCase):
    """Test cases for PitStopOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""        # Create mock tire model
        self.mock_tire_model = Mock(spec=TireDegradationModel)
        self.mock_tire_model.is_fitted = True  # Add required attribute
        
        # Mock predict method to return realistic lap times
        def mock_predict(X):
            # Simple prediction: base time + tire degradation
            base_time = 82.0
            tire_effect = X['tyre_life'] * 0.05 if 'tyre_life' in X.columns else 0
            return np.full(len(X), base_time) + tire_effect
        
        self.mock_tire_model.predict = mock_predict
        
        # Create optimizer
        self.optimizer = PitStopOptimizer(self.mock_tire_model)
        
        # Sample race parameters
        self.race_params = {
            'total_laps': 50,
            'pit_stop_time': 25.0,
            'safety_car_probability': 0.1,
            'track_characteristics': {
                'overtaking_difficulty': 0.7,
                'tire_degradation_factor': 1.0
            }
        }
        
        # Sample historical data
        self.historical_data = pd.DataFrame({
            'lap_number': list(range(1, 51)) * 3,  # 3 drivers, 50 laps each
            'driver': ['HAM'] * 50 + ['VER'] * 50 + ['LEC'] * 50,
            'lap_time': np.random.uniform(80, 85, 150),
            'compound': (['SOFT'] * 20 + ['MEDIUM'] * 30) * 3,
            'tyre_life': (list(range(1, 21)) + list(range(1, 31))) * 3,
            'stint': ([1] * 20 + [2] * 30) * 3,
            'year': [2023] * 150,
            'circuit': ['bahrain'] * 150
        })
    
    def test_init_optimizer(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.tire_model)
        self.assertEqual(self.optimizer.tire_model, self.mock_tire_model)
    
    def test_calculate_pit_windows_basic(self):
        """Test basic pit window calculation."""
        windows = self.optimizer._calculate_pit_windows(
            total_laps=50,
            tire_compound='SOFT'
        )
        
        self.assertIsInstance(windows, list)
        self.assertGreater(len(windows), 0)
        
        # Check window structure
        for window in windows:
            self.assertIn('start_lap', window)
            self.assertIn('end_lap', window)
            self.assertIn('compound', window)
            self.assertLessEqual(window['start_lap'], window['end_lap'])
    
    def test_calculate_pit_windows_different_compounds(self):
        """Test pit windows for different tire compounds."""
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        
        for compound in compounds:
            windows = self.optimizer._calculate_pit_windows(
                total_laps=50,
                tire_compound=compound
            )
            
            self.assertIsInstance(windows, list)
            self.assertGreater(len(windows), 0)
            
            # Harder compounds should generally have later pit windows
            if compound == 'HARD':
                # Hard tires should have windows starting later
                first_window_start = windows[0]['start_lap'] if windows else 0
                self.assertGreaterEqual(first_window_start, 15)
    
    def test_simulate_stint_performance(self):
        """Test stint performance simulation."""
        stint_config = {
            'start_lap': 1,
            'end_lap': 20,
            'compound': 'SOFT',
            'starting_tire_life': 1
        }
        
        performance = self.optimizer._simulate_stint_performance(stint_config)
        
        self.assertIsInstance(performance, dict)
        self.assertIn('average_lap_time', performance)
        self.assertIn('total_time', performance)
        self.assertIn('tire_degradation', performance)
        
        # Check realistic values
        self.assertGreater(performance['average_lap_time'], 80)
        self.assertLess(performance['average_lap_time'], 90)
        self.assertGreater(performance['total_time'], 0)
    
    def test_generate_strategy_options_single_stop(self):
        """Test generation of single-stop strategy options."""
        options = self.optimizer._generate_strategy_options(
            total_laps=50,
            max_stops=1
        )
        
        self.assertIsInstance(options, list)
        self.assertGreater(len(options), 0)
        
        # Check single-stop strategies
        for option in options:
            self.assertIn('pit_stops', option)
            self.assertIn('compounds', option)
            self.assertLessEqual(len(option['pit_stops']), 1)
            self.assertEqual(len(option['compounds']), len(option['pit_stops']) + 1)
    
    def test_generate_strategy_options_two_stop(self):
        """Test generation of two-stop strategy options."""
        options = self.optimizer._generate_strategy_options(
            total_laps=50,
            max_stops=2
        )
        
        self.assertIsInstance(options, list)
        
        # Should include both single and double stop strategies
        single_stop_count = sum(1 for opt in options if len(opt['pit_stops']) == 1)
        double_stop_count = sum(1 for opt in options if len(opt['pit_stops']) == 2)
        
        self.assertGreater(single_stop_count, 0)
        self.assertGreater(double_stop_count, 0)
    
    def test_evaluate_strategy_complete(self):
        """Test complete strategy evaluation."""
        strategy = {
            'pit_stops': [20],
            'compounds': ['SOFT', 'MEDIUM']
        }
        
        evaluation = self.optimizer._evaluate_strategy(strategy, self.race_params)
        
        self.assertIsInstance(evaluation, dict)
        expected_keys = [
            'total_race_time',
            'average_lap_time',
            'tire_degradation_loss',
            'pit_stop_time_loss',
            'safety_car_risk',
            'track_position_risk'
        ]
        
        for key in expected_keys:
            self.assertIn(key, evaluation)
            self.assertIsInstance(evaluation[key], (int, float))
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        strategy_evaluation = {
            'total_race_time': 4200.0,  # 70 minutes
            'safety_car_risk': 0.1,
            'track_position_risk': 0.3,
            'tire_degradation_loss': 5.0
        }
        
        confidence = self.optimizer._calculate_confidence_score(
            strategy_evaluation, 
            self.historical_data
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_optimize_strategy_basic(self):
        """Test basic strategy optimization."""
        result = self.optimizer.optimize_strategy(
            race_params=self.race_params,
            historical_data=self.historical_data,
            driver_preferences={'aggressive': 0.5},
            weather_conditions={'temperature': 25, 'humidity': 60}
        )
        
        self.assertIsInstance(result, dict)
        expected_keys = [
            'recommended_strategy',
            'alternative_strategies', 
            'confidence_score',
            'risk_assessment',
            'performance_prediction'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_optimize_strategy_recommended_structure(self):
        """Test recommended strategy structure."""
        result = self.optimizer.optimize_strategy(
            race_params=self.race_params,
            historical_data=self.historical_data
        )
        
        recommended = result['recommended_strategy']
        
        self.assertIn('pit_stops', recommended)
        self.assertIn('compounds', recommended)
        self.assertIn('total_time', recommended)
        self.assertIn('description', recommended)
        
        # Logical consistency checks
        num_stops = len(recommended['pit_stops'])
        num_compounds = len(recommended['compounds'])
        self.assertEqual(num_compounds, num_stops + 1)
    
    def test_optimize_strategy_alternatives(self):
        """Test alternative strategies generation."""
        result = self.optimizer.optimize_strategy(
            race_params=self.race_params,
            historical_data=self.historical_data
        )
        
        alternatives = result['alternative_strategies']
        
        self.assertIsInstance(alternatives, list)
        self.assertGreaterEqual(len(alternatives), 2)  # At least 2 alternatives
        
        # Each alternative should have required structure
        for alt in alternatives:
            self.assertIn('pit_stops', alt)
            self.assertIn('compounds', alt)
            self.assertIn('total_time', alt)
            self.assertIn('confidence', alt)
    
    def test_simulate_race_basic(self):
        """Test basic race simulation."""
        strategy = {
            'pit_stops': [20],
            'compounds': ['SOFT', 'MEDIUM']
        }
        
        simulation = self.optimizer.simulate_race(
            strategy=strategy,
            race_params=self.race_params,
            historical_data=self.historical_data
        )
        
        self.assertIsInstance(simulation, dict)
        expected_keys = [
            'lap_by_lap_times',
            'stint_analysis',
            'total_race_time',
            'final_position_estimate',
            'key_events'
        ]
        
        for key in expected_keys:
            self.assertIn(key, simulation)
    
    def test_simulate_race_lap_by_lap(self):
        """Test lap-by-lap simulation details."""
        strategy = {
            'pit_stops': [15, 35],
            'compounds': ['SOFT', 'MEDIUM', 'HARD']
        }
        
        simulation = self.optimizer.simulate_race(
            strategy=strategy,
            race_params=self.race_params,
            historical_data=self.historical_data
        )
        
        lap_times = simulation['lap_by_lap_times']
        
        self.assertIsInstance(lap_times, list)
        self.assertEqual(len(lap_times), self.race_params['total_laps'])
        
        # Each lap should have required info
        for lap_info in lap_times[:5]:  # Check first 5 laps
            self.assertIn('lap_number', lap_info)
            self.assertIn('lap_time', lap_info)
            self.assertIn('compound', lap_info)
            self.assertIn('tire_life', lap_info)
    
    def test_simulate_race_stint_analysis(self):
        """Test stint analysis in race simulation."""
        strategy = {
            'pit_stops': [20],
            'compounds': ['SOFT', 'MEDIUM']
        }
        
        simulation = self.optimizer.simulate_race(
            strategy=strategy,
            race_params=self.race_params,
            historical_data=self.historical_data
        )
        
        stint_analysis = simulation['stint_analysis']
        
        self.assertIsInstance(stint_analysis, list)
        self.assertEqual(len(stint_analysis), 2)  # Two stints for one pit stop
        
        for stint in stint_analysis:
            self.assertIn('stint_number', stint)
            self.assertIn('compound', stint)
            self.assertIn('start_lap', stint)
            self.assertIn('end_lap', stint)
            self.assertIn('average_time', stint)
            self.assertIn('degradation', stint)
    
    def test_analyze_historical_performance(self):
        """Test historical performance analysis."""
        analysis = self.optimizer._analyze_historical_performance(
            self.historical_data,
            'bahrain'
        )
        
        self.assertIsInstance(analysis, dict)
        expected_keys = [
            'average_lap_times_by_compound',
            'typical_pit_windows',
            'degradation_patterns',
            'safety_car_frequency'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
    
    def test_weather_impact_calculation(self):
        """Test weather impact on strategy."""
        base_conditions = {'temperature': 25, 'humidity': 60}
        hot_conditions = {'temperature': 35, 'humidity': 80}
        
        base_result = self.optimizer.optimize_strategy(
            race_params=self.race_params,
            historical_data=self.historical_data,
            weather_conditions=base_conditions
        )
        
        hot_result = self.optimizer.optimize_strategy(
            race_params=self.race_params,
            historical_data=self.historical_data,
            weather_conditions=hot_conditions
        )
        
        # Hot weather should generally lead to different strategy
        # (though exact behavior depends on implementation)
        self.assertIsInstance(base_result, dict)
        self.assertIsInstance(hot_result, dict)
        
        # Both should have valid confidence scores
        self.assertGreaterEqual(base_result['confidence_score'], 0)
        self.assertGreaterEqual(hot_result['confidence_score'], 0)


if __name__ == '__main__':
    unittest.main()
