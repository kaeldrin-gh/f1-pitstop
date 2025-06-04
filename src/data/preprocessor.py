"""
F1 Data Preprocessing Module
Handles data cleaning, feature engineering, and validation
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class F1DataPreprocessor:
    """
    F1 Data Preprocessing class for cleaning and feature engineering
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'lap_time'
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean race data
        
        Args:
            df: Raw race data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        logger.info(f"Starting data validation with {initial_rows} rows")

        # Handle column name compatibility
        if 'compound' in df.columns and 'tire_compound' not in df.columns:
            df['tire_compound'] = df['compound']
        if 'tyre_life' in df.columns and 'tire_age' not in df.columns:
            df['tire_age'] = df['tyre_life']

        # Remove rows with missing critical data
        df = df.dropna(subset=['lap_time', 'tire_compound', 'tire_age', 'driver'])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing critical data")

        # Filter lap times within valid range
        df = df[
            (df['lap_time'] >= config.MIN_LAP_TIME) & 
            (df['lap_time'] <= config.MAX_LAP_TIME)
        ]
        logger.info(f"Filtered to {len(df)} rows with valid lap times")

        # Remove tire stints longer than maximum
        df = df[df['tire_age'] <= config.MAX_STINT_LENGTH]
        logger.info(f"Filtered to {len(df)} rows with valid tire age")

        # Remove outliers using IQR method
        df = self._remove_outliers(df)

        # Validate tire compounds
        valid_compounds = set(config.TIRE_COMPOUNDS)
        df = df[df['tire_compound'].isin(valid_compounds)]
        logger.info(f"Filtered to {len(df)} rows with valid tire compounds")
        
        logger.info(f"Data validation complete: {len(df)} rows remaining")
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, column: str = 'lap_time') -> pd.DataFrame:
        """
        Remove outliers using IQR method
        
        Args:
            df: DataFrame to clean
            column: Column to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_rows = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        outliers_removed = initial_rows - len(df)
        logger.info(f"Removed {outliers_removed} outliers from {column}")
        
        return df
    
    def engineer_features(self, race_data: pd.DataFrame, pit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for machine learning
        
        Args:
            race_data: Race lap data
            pit_data: Pit stop data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Sort data chronologically
        race_data = race_data.sort_values(['year', 'circuit', 'driver', 'lap_number'])
        
        # Calculate tire degradation features
        race_data = self._calculate_tire_degradation(race_data)
        
        # Add track position features
        race_data = self._add_position_features(race_data)
        
        # Add rolling averages
        race_data = self._add_rolling_features(race_data)
        
        # Add pit stop context
        race_data = self._add_pit_context(race_data, pit_data)
        
        # Add circuit-specific features
        race_data = self._add_circuit_features(race_data)
        
        # Encode categorical variables
        race_data = self._encode_categorical_features(race_data)
          # Select feature columns
        self.feature_columns = [
            'tire_age', 'tire_compound_encoded', 'circuit_encoded', 'driver_encoded',
            'lap_number', 'position', 'gap_to_leader', 'tire_degradation_rate',
            'stint_length', 'laps_since_pit', 'rolling_avg_lap_time_3', 'rolling_avg_lap_time_5',
            'position_change', 'relative_performance'
        ]
        
        # Remove rows with missing engineered features
        race_data = race_data.dropna(subset=self.feature_columns + [self.target_column])
        
        logger.info(f"Feature engineering complete: {len(race_data)} rows, {len(self.feature_columns)} features")
        return race_data
    
    def _calculate_tire_degradation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate tire degradation features"""
        # Add tire degradation rate from config
        df['tire_degradation_rate'] = df['tire_compound'].map(config.TIRE_DEGRADATION_RATES)
        
        # Calculate predicted degradation
        df['predicted_degradation'] = df['tire_age'] * df['tire_degradation_rate']
        
        # Calculate stint length
        df['stint_length'] = df.groupby(['year', 'circuit', 'driver', 'tire_compound'])['tire_age'].transform('max')
        
        return df
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track position related features"""
        # Add position column if missing (for test data)
        if 'position' not in df.columns:
            df['position'] = 10  # Default mid-field position
        
        # Fill missing positions
        df['position'] = df['position'].fillna(20)
        
        # Calculate position change within stint
        df['position_change'] = df.groupby(['year', 'circuit', 'driver'])['position'].diff()
        df['position_change'] = df['position_change'].fillna(0)
        
        # Add gap_to_leader column if missing (for test data)
        if 'gap_to_leader' not in df.columns:
            df['gap_to_leader'] = df['position'] * 0.5  # Rough estimate
          # Fill missing gap to leader
        df['gap_to_leader'] = df['gap_to_leader'].fillna(df['gap_to_leader'].median())
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features"""
        # Reset index to ensure compatibility
        df = df.reset_index(drop=True)
        
        # Rolling average lap time (last 3 laps)
        rolling_values_3 = df.groupby(['year', 'circuit', 'driver'])['lap_time'].rolling(
            window=3, min_periods=1).mean().values
        df['rolling_avg_lap_time_3'] = rolling_values_3
        
        # Rolling average lap time (last 5 laps)
        rolling_values_5 = df.groupby(['year', 'circuit', 'driver'])['lap_time'].rolling(
            window=5, min_periods=1).mean().values
        df['rolling_avg_lap_time_5'] = rolling_values_5
        
        # Relative performance compared to session average
        session_avg = df.groupby(['year', 'circuit', 'lap_number'])['lap_time'].transform('mean')
        df['relative_performance'] = df['lap_time'] / session_avg
        
        return df
    
    def _add_pit_context(self, race_data: pd.DataFrame, pit_data: pd.DataFrame) -> pd.DataFrame:
        """Add pit stop context features"""
        # Initialize laps since pit
        race_data['laps_since_pit'] = 0
        
        # Calculate laps since last pit stop for each driver
        for _, pit_stop in pit_data.iterrows():
            mask = (
                (race_data['year'] == pit_stop['year']) &
                (race_data['circuit'] == pit_stop['circuit']) &
                (race_data['driver'] == pit_stop['driver']) &
                (race_data['lap_number'] >= pit_stop['pit_lap'])
            )
            race_data.loc[mask, 'laps_since_pit'] = race_data.loc[mask, 'lap_number'] - pit_stop['pit_lap']
        
        return race_data
    
    def _add_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add circuit-specific features"""
        # Circuit characteristics (simplified)
        circuit_characteristics = {
            'monaco': {'overtaking_difficulty': 5, 'tire_wear': 2},
            'silverstone': {'overtaking_difficulty': 2, 'tire_wear': 4},
            'monza': {'overtaking_difficulty': 1, 'tire_wear': 3},
            'spa': {'overtaking_difficulty': 2, 'tire_wear': 3},
            'suzuka': {'overtaking_difficulty': 3, 'tire_wear': 4}
        }
        
        for circuit, chars in circuit_characteristics.items():
            mask = df['circuit'] == circuit
            for char, value in chars.items():
                df.loc[mask, char] = value
        
        # Fill missing values
        df['overtaking_difficulty'] = df['overtaking_difficulty'].fillna(3)
        df['tire_wear'] = df['tire_wear'].fillna(3)
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_columns = ['tire_compound', 'circuit', 'driver']
        
        # Label encoding for circuit and driver
        for column in ['circuit', 'driver']:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                # Handle new categories
                try:
                    df[f'{column}_encoded'] = self.label_encoders[column].transform(df[column])
                except ValueError:
                    # Add new categories
                    unique_values = df[column].unique()
                    for value in unique_values:
                        if value not in self.label_encoders[column].classes_:
                            self.label_encoders[column].classes_ = np.append(
                                self.label_encoders[column].classes_, value)
                    df[f'{column}_encoded'] = self.label_encoders[column].transform(df[column])
        
        # One-hot encoding for tire compound (as expected by tests)
        from config import TIRE_COMPOUNDS
        for compound in TIRE_COMPOUNDS:
            df[f'compound_{compound}'] = (df['tire_compound'] == compound).astype(int)
        
        # Also keep label encoded version for backwards compatibility
        if 'tire_compound' not in self.label_encoders:
            self.label_encoders['tire_compound'] = LabelEncoder()
            df['tire_compound_encoded'] = self.label_encoders['tire_compound'].fit_transform(df['tire_compound'])
        else:
            try:
                df['tire_compound_encoded'] = self.label_encoders['tire_compound'].transform(df['tire_compound'])
            except ValueError:
                unique_values = df['tire_compound'].unique()
                for value in unique_values:
                    if value not in self.label_encoders['tire_compound'].classes_:
                        self.label_encoders['tire_compound'].classes_ = np.append(
                            self.label_encoders['tire_compound'].classes_, value)
                df['tire_compound_encoded'] = self.label_encoders['tire_compound'].transform(df['tire_compound'])
        
        return df
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train and test sets
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (train_data, test_data)
        """        # Sort by year and circuit to ensure chronological split
        df = df.sort_values(['year', 'circuit', 'lap_number'])
        
        # Split chronologically - last 20% for testing
        test_size = int(len(df) * 0.2)
        if test_size == 0 and len(df) > 0:
            test_size = 1  # Ensure at least one test sample
        
        train_data = df.iloc[:-test_size] if test_size > 0 else df
        test_data = df.iloc[-test_size:] if test_size > 0 else pd.DataFrame()
        
        logger.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
        
        return train_data, test_data
    
    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target variable
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Ensure all feature columns exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}")
            available_features = [col for col in self.feature_columns if col in df.columns]
            self.feature_columns = available_features
        
        features = df[self.feature_columns]
        target = df[self.target_column]
        
        return features, target
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate processed data and return quality metrics
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {
            'total_rows': len(df),
            'total_features': len(self.feature_columns),
            'missing_values': df[self.feature_columns + [self.target_column]].isnull().sum().sum(),
            'unique_drivers': df['driver'].nunique(),
            'unique_circuits': df['circuit'].nunique(),
            'year_range': f"{df['year'].min()}-{df['year'].max()}",
            'lap_time_range': f"{df['lap_time'].min():.2f}-{df['lap_time'].max():.2f}",
            'tire_compounds': df['tire_compound'].unique().tolist()
        }
        
        # Log validation results
        logger.info("Data validation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Check for data quality issues
        if metrics['missing_values'] > 0:
            logger.warning(f"Found {metrics['missing_values']} missing values")
        
        if metrics['total_rows'] < 1000:
            logger.warning(f"Low data volume: {metrics['total_rows']} rows")
        
        return metrics
    
    # Methods expected by tests (aliases to existing functionality)
    def _validate_data_structure(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame structure (test interface)"""
        if df.empty:
            return False
        
        required_columns = ['lap_time', 'compound', 'tyre_life', 'stint', 'driver', 
                           'track_status', 'lap_number', 'year', 'circuit']
        missing_columns = set(required_columns) - set(df.columns)
        return len(missing_columns) == 0
    
    def _remove_invalid_laps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid laps (test interface)"""
        # Filter lap times within valid range
        valid_df = df[
            (df['lap_time'] >= config.MIN_LAP_TIME) & 
            (df['lap_time'] <= config.MAX_LAP_TIME)
        ].copy()
        
        # Remove rows with missing critical data
        valid_df = valid_df.dropna(subset=['lap_time', 'compound', 'tyre_life'])
        
        # Remove safety car laps (track_status != '1')
        if 'track_status' in valid_df.columns:
            valid_df = valid_df[valid_df['track_status'] == '1']
        
        return valid_df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features (test interface)"""
        # Handle column name compatibility first
        if 'compound' in df.columns and 'tire_compound' not in df.columns:
            df['tire_compound'] = df['compound']
        if 'tyre_life' in df.columns and 'tire_age' not in df.columns:
            df['tire_age'] = df['tyre_life']
        
        # Use existing engineer_features but with empty pit_data for simplicity
        empty_pit_data = pd.DataFrame(columns=['driver', 'lap', 'pit_duration'])
        return self.engineer_features(df, empty_pit_data)
    
    def _split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically (test interface)"""
        return self.split_data(df)
    
    def _validate_processed_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data (test interface)"""
        if df.empty:
            return False
        
        # Check for reasonable lap times
        lap_time_valid = df['lap_time'].between(config.MIN_LAP_TIME, config.MAX_LAP_TIME).all()
        
        # Check for reasonable tire stint lengths (if tyre_life exists)
        stint_valid = True
        if 'tyre_life' in df.columns:
            stint_valid = (df['tyre_life'] <= 50).all()
        
        return lap_time_valid and stint_valid
    
    def process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process data pipeline (test interface)"""
        # Validate structure
        if not self._validate_data_structure(df):
            raise ValueError("Invalid data structure")
        
        # Clean data
        clean_data = self._remove_invalid_laps(df)
        
        # Engineer features
        processed_data = self._engineer_features(clean_data)
        
        # Split data
        train_data, test_data = self._split_train_test(processed_data)
        
        return train_data, test_data
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return self.feature_columns.copy() if self.feature_columns else []

def main():
    """Main function for testing preprocessing"""
    from src.data.data_extractor import F1DataExtractor
    
    # Load data
    extractor = F1DataExtractor()
    race_data, pit_data = extractor.load_combined_data()
    
    if race_data.empty:
        logger.error("No race data available")
        return
    
    # Preprocess data
    preprocessor = F1DataPreprocessor()
    
    # Validate data
    clean_data = preprocessor.validate_data(race_data)
    
    # Engineer features
    processed_data = preprocessor.engineer_features(clean_data, pit_data)
    
    # Split data
    train_data, test_data = preprocessor.split_data(processed_data)
    
    # Validate processed data
    preprocessor.validate_processed_data(processed_data)
    
    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()
