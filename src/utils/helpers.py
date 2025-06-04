"""
Utility functions for F1 Pit Stop Strategy Tool
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import sqlite3
import json
import os
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=getattr(logging, log_level.upper()), format=log_format)


def validate_lap_time(lap_time: float) -> bool:
    """
    Validate if lap time is within acceptable range
    
    Args:
        lap_time: Lap time in seconds
        
    Returns:
        True if valid, False otherwise
    """
    return config.MIN_LAP_TIME <= lap_time <= config.MAX_LAP_TIME


def validate_tire_compound(compound: str) -> bool:
    """
    Validate tire compound
    
    Args:
        compound: Tire compound string
        
    Returns:
        True if valid, False otherwise
    """
    return compound in config.TIRE_COMPOUNDS


def validate_circuit(circuit: str) -> bool:
    """
    Validate circuit name
    
    Args:
        circuit: Circuit name
        
    Returns:
        True if valid, False otherwise
    """
    return circuit in config.CIRCUITS


def validate_tire_stint(stint_length: int) -> bool:
    """
    Validate if tire stint length is within acceptable range
    
    Args:
        stint_length: Number of laps for the tire stint
        
    Returns:
        True if valid (1-50 laps), False otherwise
    """
    return 1 <= stint_length <= config.MAX_STINT_LENGTH


def calculate_time_gap(lap_time1: float, lap_time2: float, num_laps: int = 1) -> float:
    """
    Calculate time gap between two lap times over multiple laps
    
    Args:
        lap_time1: First lap time
        lap_time2: Second lap time
        num_laps: Number of laps to calculate over
        
    Returns:
        Time gap in seconds
    """
    return (lap_time1 - lap_time2) * num_laps


def convert_lap_time_to_seconds(lap_time_str: str) -> Optional[float]:
    """
    Convert lap time string to seconds
    
    Args:
        lap_time_str: Lap time in format "mm:ss.sss" or seconds
        
    Returns:
        Lap time in seconds or None if invalid
    """
    try:
        if ':' in lap_time_str:
            # Format: "mm:ss.sss"
            parts = lap_time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            # Assume already in seconds
            return float(lap_time_str)
    except (ValueError, IndexError):
        logger.warning(f"Invalid lap time format: {lap_time_str}")
        return None


def format_lap_time(seconds: float) -> str:
    """
    Format lap time from seconds to string
    
    Args:
        seconds: Lap time in seconds
        
    Returns:
        Formatted lap time string
    """
    try:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:06.3f}"
    except:
        return f"{seconds:.3f}s"


def calculate_stint_performance(lap_times: List[float], tire_ages: List[int]) -> Dict[str, float]:
    """
    Calculate stint performance metrics
    
    Args:
        lap_times: List of lap times in stint
        tire_ages: List of tire ages corresponding to lap times
        
    Returns:
        Dictionary with performance metrics
    """
    if not lap_times or len(lap_times) != len(tire_ages):
        return {}
    
    lap_times = np.array(lap_times)
    tire_ages = np.array(tire_ages)
    
    # Calculate degradation rate
    if len(lap_times) > 1:
        degradation_rate = np.polyfit(tire_ages, lap_times, 1)[0]
    else:
        degradation_rate = 0.0
    
    return {
        'average_lap_time': float(np.mean(lap_times)),
        'fastest_lap': float(np.min(lap_times)),
        'slowest_lap': float(np.max(lap_times)),
        'degradation_rate': float(degradation_rate),
        'consistency': float(np.std(lap_times)),
        'stint_length': len(lap_times)
    }


def estimate_fuel_effect(fuel_load: float, base_lap_time: float) -> float:
    """
    Estimate fuel effect on lap time
    
    Args:
        fuel_load: Fuel load in kg
        base_lap_time: Base lap time without fuel effect
        
    Returns:
        Adjusted lap time with fuel effect
    """
    # Simplified fuel effect: ~0.03s per kg of fuel
    fuel_effect_per_kg = 0.03
    fuel_penalty = fuel_load * fuel_effect_per_kg
    return base_lap_time + fuel_penalty


def calculate_track_position_value(position: int, total_cars: int = 20) -> float:
    """
    Calculate the value of track position
    
    Args:
        position: Current track position
        total_cars: Total number of cars
        
    Returns:
        Position value (higher is better)
    """
    # Linear scale where P1 = 1.0, last place = 0.0
    return max(0.0, (total_cars - position + 1) / total_cars)


def estimate_overtaking_difficulty(circuit: str) -> float:
    """
    Estimate overtaking difficulty for a circuit
    
    Args:
        circuit: Circuit name
        
    Returns:
        Difficulty score (1-5, higher is more difficult)
    """
    # Simplified overtaking difficulty ratings
    difficulty_ratings = {
        'monaco': 5.0,      # Very difficult
        'silverstone': 2.0, # Easy
        'monza': 1.0,       # Very easy
        'spa': 2.0,         # Easy
        'suzuka': 3.0       # Moderate
    }
    
    return difficulty_ratings.get(circuit, 3.0)


def calculate_safety_car_probability(lap: int, total_laps: int, circuit: str) -> float:
    """
    Calculate probability of safety car deployment
    
    Args:
        lap: Current lap
        total_laps: Total race laps
        circuit: Circuit name
        
    Returns:
        Safety car probability (0-1)
    """
    # Base probability varies by circuit
    base_probability = {
        'monaco': 0.4,      # High chance
        'silverstone': 0.2, # Medium chance
        'monza': 0.15,      # Low chance
        'spa': 0.25,        # Medium chance
        'suzuka': 0.2       # Medium chance
    }.get(circuit, 0.2)
    
    # Probability increases towards race end
    race_progress = lap / total_laps
    if race_progress > 0.8:
        base_probability *= 1.5
    elif race_progress < 0.2:
        base_probability *= 0.7
    
    return min(1.0, base_probability)


def save_strategy_to_json(strategy_data: Dict[str, Any], filepath: str) -> bool:
    """
    Save strategy data to JSON file
    
    Args:
        strategy_data: Strategy data dictionary
        filepath: File path to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(strategy_data, f, indent=2, default=str)
        logger.info(f"Strategy saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save strategy: {e}")
        return False


def load_strategy_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load strategy data from JSON file
    
    Args:
        filepath: File path to load
        
    Returns:
        Strategy data dictionary or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            strategy_data = json.load(f)
        logger.info(f"Strategy loaded from {filepath}")
        return strategy_data
    except Exception as e:
        logger.error(f"Failed to load strategy: {e}")
        return None


def clean_driver_name(driver_name: str) -> str:
    """
    Clean and standardize driver name
    
    Args:
        driver_name: Raw driver name
        
    Returns:
        Cleaned driver name
    """
    if not driver_name:
        return "Unknown"
    
    # Remove extra whitespace and convert to title case
    cleaned = driver_name.strip().title()
    
    # Handle common abbreviations
    abbreviations = {
        'Ver': 'VER',
        'Ham': 'HAM',
        'Rus': 'RUS',
        'Lec': 'LEC',
        'Sai': 'SAI',
        'Nor': 'NOR',
        'Pia': 'PIA',
        'Alo': 'ALO',
        'Str': 'STR',
        'Per': 'PER'
    }
    
    for abbr, correct in abbreviations.items():
        if cleaned.endswith(abbr):
            cleaned = cleaned.replace(abbr, correct)
            break
    
    return cleaned


def calculate_championship_impact(current_points: int, position_gained: int) -> int:
    """
    Calculate championship points impact of position change
    
    Args:
        current_points: Current championship points
        position_gained: Positions gained/lost (positive = gained)
        
    Returns:
        Points change
    """
    # F1 points system
    points_system = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    
    # Simplified calculation
    points_per_position = 2.0  # Average points per position
    return int(position_gained * points_per_position)


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return metrics
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Data quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'quality_score': 0.0
    }
    
    # Calculate quality score
    quality_score = 1.0
    
    # Penalize missing values
    if metrics['total_rows'] > 0:
        missing_ratio = metrics['missing_values'] / (metrics['total_rows'] * len(df.columns))
        quality_score -= missing_ratio * 0.5
    
    # Penalize duplicates
    if metrics['total_rows'] > 0:
        duplicate_ratio = metrics['duplicate_rows'] / metrics['total_rows']
        quality_score -= duplicate_ratio * 0.3
    
    metrics['quality_score'] = max(0.0, quality_score)
    
    return metrics


def create_data_summary(df: pd.DataFrame) -> str:
    """
    Create a summary string of DataFrame
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary string
    """
    if df.empty:
        return "Empty DataFrame"
    
    summary_parts = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB"
    ]
    
    # Add column info
    if 'lap_time' in df.columns:
        summary_parts.append(f"Lap times: {df['lap_time'].min():.1f}s - {df['lap_time'].max():.1f}s")
    
    if 'year' in df.columns:
        years = sorted(df['year'].unique())
        summary_parts.append(f"Years: {years[0]}-{years[-1]}")
    
    if 'circuit' in df.columns:
        circuits = df['circuit'].nunique()
        summary_parts.append(f"Circuits: {circuits}")
    
    return " | ".join(summary_parts)


def performance_timer(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    
    return wrapper


# Database utilities
def execute_sql_query(query: str, database_path: str = config.DATABASE_PATH) -> pd.DataFrame:
    """
    Execute SQL query and return DataFrame
    
    Args:
        query: SQL query string
        database_path: Path to SQLite database
        
    Returns:
        Query results as DataFrame
    """
    try:
        with sqlite3.connect(database_path) as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return pd.DataFrame()


def get_database_info(database_path: str = config.DATABASE_PATH) -> Dict[str, Any]:
    """
    Get database information
    
    Args:
        database_path: Path to SQLite database
        
    Returns:
        Database information dictionary
    """
    try:
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get table info
            table_info = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]
                table_info[table] = {'rows': row_count}
            
            return {
                'tables': tables,
                'table_info': table_info,
                'database_size': f"{os.path.getsize(database_path) / 1024:.1f} KB"
            }
    
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {}


# Additional utility functions
def validate_data_completeness(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate if DataFrame has complete data
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    if data.empty:
        return {
            'is_valid': False,
            'missing_data_report': {'error': 'DataFrame is empty'},
            'completeness_score': 0.0
        }
    
    # Calculate completeness metrics
    total_cells = data.size
    missing_cells = data.isnull().sum().sum()
    completeness_score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
    
    # Generate missing data report
    missing_data_report = {}
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            missing_data_report[col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(data)) * 100
            }
    
    # Determine if data is valid (100% complete or no missing values)
    is_valid = missing_cells == 0
    
    return {
        'is_valid': is_valid,
        'missing_data_report': missing_data_report,
        'completeness_score': completeness_score
    }


def log_performance_metrics(metrics: Dict[str, float], log_file: str) -> None:
    """
    Log performance metrics to a file
    
    Args:
        metrics: Dictionary of performance metrics
        log_file: Path to log file
    """
    import datetime
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Write metrics to file (append mode)
    with open(log_file, 'a') as f:
        f.write(f"Performance Metrics - {timestamp}\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")


def create_database_backup(db_path: str, backup_path: str) -> bool:
    """
    Create a backup of the database
    
    Args:
        db_path: Path to the original database
        backup_path: Path for the backup file
        
    Returns:
        True if backup successful, False otherwise
    """
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backup created: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil
    import shutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total / 1024**3,  # GB as float
        'memory_available': psutil.virtual_memory().available / 1024**3,  # GB as float
        'disk_free_space': shutil.disk_usage('/').free / 1024**3 if platform.system() != 'Windows' else shutil.disk_usage('C:\\').free / 1024**3  # GB as float
    }


def format_time_delta(seconds: float) -> str:
    """
    Format time difference in a human-readable format
    
    Args:
        seconds: Time difference in seconds
        
    Returns:
        Formatted time string
    """
    if seconds == 0:
        return "0.00s"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"


def calculate_percentile_stats(data) -> Dict[str, float]:
    """
    Calculate percentile statistics for a data series
    
    Args:
        data: Pandas series or list of numeric data
        
    Returns:
        Dictionary with percentile statistics
    """
    # Convert to pandas Series if it's a list
    if isinstance(data, list):
        if len(data) == 0:
            return {
                'min': None, 'q25': None, 'median': None, 'q75': None,
                'max': None, 'iqr': None, 'mean': None, 'std': None
            }
        data = pd.Series(data)
    
    # Handle standard deviation for single value (should be 0)
    std_val = data.std()
    if pd.isna(std_val) and len(data) == 1:
        std_val = 0.0
    
    return {
        'min': data.min(),
        'q25': data.quantile(0.25),
        'median': data.median(),
        'q75': data.quantile(0.75),
        'max': data.max(),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'mean': data.mean(),
        'std': std_val
    }


# Constants and configurations
F1_TEAMS_2024 = [
    'Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren',
    'Aston Martin', 'Alpine', 'Williams', 'RB',
    'Kick Sauber', 'Haas'
]

DRIVER_ABBREVIATIONS = {
    'MAX': 'Max Verstappen', 'PER': 'Sergio Perez',
    'HAM': 'Lewis Hamilton', 'RUS': 'George Russell',
    'LEC': 'Charles Leclerc', 'SAI': 'Carlos Sainz',
    'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri',
    'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll'
}


def get_driver_full_name(abbreviation: str) -> str:
    """
    Get full driver name from abbreviation
    
    Args:
        abbreviation: Driver abbreviation
        
    Returns:
        Full driver name
    """
    return DRIVER_ABBREVIATIONS.get(abbreviation.upper(), abbreviation)
