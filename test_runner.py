"""
Test runner script for F1 Pit Stop Strategy Tool.
Runs basic validation tests without requiring pytest.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        # Test configuration
        import config
        print("✓ Config module imported successfully")
        
        # Test data modules
        from src.data.data_extractor import F1DataExtractor
        from src.data.preprocessor import F1DataPreprocessor
        print("✓ Data modules imported successfully")
        
        # Test model modules
        from src.models.tire_model import TireDegradationModel
        from src.models.strategy_optimizer import PitStopOptimizer
        print("✓ Model modules imported successfully")
        
        # Test visualization
        from src.visualization.dashboard import create_dashboard
        print("✓ Visualization module imported successfully")
        
        # Test utilities
        from src.utils.helpers import validate_lap_time, get_system_info
        print("✓ Utility modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test config values
        import config
        assert hasattr(config, 'ANALYSIS_YEARS')
        assert hasattr(config, 'CIRCUITS')
        assert hasattr(config, 'TIRE_COMPOUNDS')
        print("✓ Configuration values are accessible")
        
        # Test utility functions
        from src.utils.helpers import validate_lap_time, format_time_delta
        assert validate_lap_time(80.5) == True
        assert validate_lap_time(50.0) == False
        assert format_time_delta(65.5) == "1m 5.50s"
        print("✓ Utility functions work correctly")
        
        # Test tire model initialization
        from src.models.tire_model import TireDegradationModel
        model = TireDegradationModel()
        assert model.model is not None
        print("✓ Tire model can be initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'config.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/data_extractor.py',
        'src/data/preprocessor.py',
        'src/models/__init__.py',
        'src/models/tire_model.py',
        'src/models/strategy_optimizer.py',
        'src/visualization/__init__.py',
        'src/visualization/dashboard.py',
        'src/utils/__init__.py',
        'src/utils/helpers.py',
        'tests/__init__.py',
        'tests/test_data_extractor.py',
        'tests/test_preprocessor.py',
        'tests/test_tire_model.py',
        'tests/test_strategy_optimizer.py',
        'tests/test_helpers.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files are present")
        return True

def test_requirements():
    """Test that requirements.txt contains expected packages."""
    print("\nTesting requirements...")
    
    try:
        requirements_path = project_root / 'requirements.txt'
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'fastf1==3.1.0',
            'pandas==1.5.3',
            'streamlit==1.28.0',
            'scikit-learn==1.3.0',
            'plotly==5.17.0',
            'seaborn==0.12.2'
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"✗ Missing packages in requirements.txt: {missing_packages}")
            return False
        else:
            print("✓ All required packages are listed in requirements.txt")
            return True
            
    except Exception as e:
        print(f"✗ Requirements test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("F1 Pit Stop Strategy Tool - Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_requirements,
        test_imports,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The project is ready for use.")
        return True
    else:
        print("✗ Some tests failed. Please check the output above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
