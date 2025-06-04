"""
F1 Pit Stop Strategy Tool Configuration
"""

# Analysis Configuration
ANALYSIS_YEARS = [2021, 2022, 2023, 2024]

CIRCUITS = {
    'monaco': 'Monaco Grand Prix',
    'silverstone': 'British Grand Prix', 
    'monza': 'Italian Grand Prix',
    'spa': 'Belgian Grand Prix',
    'suzuka': 'Japanese Grand Prix'
}

# Tire Configuration
TIRE_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

TIRE_DEGRADATION_RATES = {
    'SOFT': 0.35,    # seconds per lap
    'MEDIUM': 0.25,  # seconds per lap
    'HARD': 0.15     # seconds per lap
}

# Pit Stop Configuration
PIT_STOP_PENALTY = 25.0  # seconds

# Data Configuration
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
DATABASE_DIR = 'data/database'
DATABASE_PATH = 'data/database/f1_data.db'

# Model Configuration
MODEL_DIR = 'models'
TIRE_MODEL_PATH = 'models/tire_degradation_model.pkl'

# Dashboard Configuration
STREAMLIT_PORT = 8501

# Data Validation Configuration
MIN_LAP_TIME = 60.0   # seconds
MAX_LAP_TIME = 180.0  # seconds
MAX_STINT_LENGTH = 50 # laps

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
