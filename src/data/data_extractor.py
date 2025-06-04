"""
F1 Data Extraction Module
Handles downloading and extracting F1 race data using FastF1 API
"""

import logging
import os
import pandas as pd
import fastf1
import sqlite3
from typing import List, Dict, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

Base = declarative_base()

class RaceData(Base):
    """SQLAlchemy model for race data"""
    __tablename__ = 'race_data'
    
    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    circuit = Column(String(50), nullable=False)
    driver = Column(String(50), nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time = Column(Float, nullable=False)
    tire_compound = Column(String(10), nullable=False)
    tire_age = Column(Integer, nullable=False)
    track_status = Column(String(20))
    position = Column(Integer)
    gap_to_leader = Column(Float)

class PitStopData(Base):
    """SQLAlchemy model for pit stop data"""
    __tablename__ = 'pit_stops'
    
    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    circuit = Column(String(50), nullable=False)
    driver = Column(String(50), nullable=False)
    pit_lap = Column(Integer, nullable=False)
    pit_time = Column(Float, nullable=False)
    tire_compound_out = Column(String(10))
    tire_compound_in = Column(String(10))

class F1DataExtractor:
    """
    F1 Data Extraction class for downloading and processing race data
    """
    
    def __init__(self, database_path: str = config.DATABASE_PATH):
        """
        Initialize the data extractor
        
        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = database_path
        self.engine = create_engine(f'sqlite:///{database_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize tables expected by tests
        self._init_database()
          # Enable FastF1 cache
        fastf1.Cache.enable_cache('data/cache')
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with self.engine.connect() as conn:
                # Create sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY,
                        year INTEGER,
                        circuit TEXT,
                        session_type TEXT,
                        date TEXT
                    )
                """))
                
                # Create laps table  
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS laps (
                        id INTEGER PRIMARY KEY,
                        year INTEGER,
                        circuit TEXT,
                        session_type TEXT,
                        driver TEXT,
                        lap_number INTEGER,
                        lap_time REAL,
                        compound TEXT,
                        stint INTEGER,
                        tyre_life INTEGER,
                        track_status TEXT,
                        is_personal_best BOOLEAN
                    )
                """))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        
    def download_race_data(self, year: int, circuit: str) -> Optional[pd.DataFrame]:
        """
        Download race data for a specific year and circuit
        
        Args:
            year: Race year
            circuit: Circuit identifier
            
        Returns:
            DataFrame with race data or None if failed
        """
        try:
            logger.info(f"Downloading race data for {year} {circuit}")
            
            # Load race session
            session = fastf1.get_session(year, circuit, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            # Get lap data
            laps = session.laps
            
            if laps.empty:
                logger.warning(f"No lap data found for {year} {circuit}")
                return None
            
            # Filter valid laps
            valid_laps = laps[
                (laps['LapTime'].dt.total_seconds() >= config.MIN_LAP_TIME) &
                (laps['LapTime'].dt.total_seconds() <= config.MAX_LAP_TIME) &
                (laps['TrackStatus'] == '1')  # Green flag
            ].copy()
            
            if valid_laps.empty:
                logger.warning(f"No valid laps found for {year} {circuit}")
                return None
            
            # Process lap data
            race_data = []
            for _, lap in valid_laps.iterrows():
                try:
                    race_data.append({
                        'year': year,
                        'circuit': circuit,
                        'driver': lap['Driver'],
                        'lap_number': lap['LapNumber'],
                        'lap_time': lap['LapTime'].total_seconds(),
                        'tire_compound': lap['Compound'],
                        'tire_age': lap['TyreLife'],
                        'track_status': lap['TrackStatus'],
                        'position': lap['Position'],
                        'gap_to_leader': lap['GapToLeader']
                    })
                except Exception as e:
                    logger.warning(f"Error processing lap: {e}")
                    continue
            
            df = pd.DataFrame(race_data)
            logger.info(f"Successfully downloaded {len(df)} laps for {year} {circuit}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download race data for {year} {circuit}: {e}")
            return None
    
    def extract_pit_stops(self, year: int, circuit: str) -> Optional[pd.DataFrame]:
        """
        Extract pit stop data for a specific year and circuit
        
        Args:
            year: Race year
            circuit: Circuit identifier
            
        Returns:
            DataFrame with pit stop data or None if failed
        """
        try:
            logger.info(f"Extracting pit stops for {year} {circuit}")
            
            # Load race session
            session = fastf1.get_session(year, circuit, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            # Get pit stop data
            laps = session.laps
            pit_stops = []
            
            for driver in laps['Driver'].unique():
                driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
                
                for i in range(1, len(driver_laps)):
                    current_lap = driver_laps.iloc[i]
                    previous_lap = driver_laps.iloc[i-1]
                    
                    # Check for tire compound change (indicating pit stop)
                    if (current_lap['Compound'] != previous_lap['Compound'] or 
                        current_lap['TyreLife'] < previous_lap['TyreLife']):
                        
                        pit_stops.append({
                            'year': year,
                            'circuit': circuit,
                            'driver': driver,
                            'pit_lap': current_lap['LapNumber'],
                            'pit_time': config.PIT_STOP_PENALTY,  # Simplified pit time
                            'tire_compound_out': previous_lap['Compound'],
                            'tire_compound_in': current_lap['Compound']
                        })
            
            if not pit_stops:
                logger.warning(f"No pit stops found for {year} {circuit}")
                return None
                
            df = pd.DataFrame(pit_stops)
            logger.info(f"Successfully extracted {len(df)} pit stops for {year} {circuit}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract pit stops for {year} {circuit}: {e}")
            return None
    
    def save_to_database(self, race_data: pd.DataFrame, pit_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Save race and pit stop data to database
        
        Args:
            race_data: DataFrame with race data
            pit_data: DataFrame with pit stop data (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            # Save race data
            race_data.to_sql('race_data', self.engine, if_exists='append', index=False)
            logger.info(f"Saved {len(race_data)} race records to database")
            
            # Save pit stop data if provided
            if pit_data is not None and not pit_data.empty:
                pit_data.to_sql('pit_stops', self.engine, if_exists='append', index=False)
                logger.info(f"Saved {len(pit_data)} pit stop records to database")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save data to database: {e}")
            return False
    
    def load_combined_data(self, years: Optional[List[int]] = None, 
                          circuits: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load combined race and pit stop data from database
        
        Args:
            years: List of years to filter (optional)
            circuits: List of circuits to filter (optional)
            
        Returns:
            Tuple of (race_data, pit_stop_data) DataFrames
        """
        try:
            # Build query conditions
            race_conditions = []
            pit_conditions = []
            
            if years:
                year_filter = f"year IN ({','.join(map(str, years))})"
                race_conditions.append(year_filter)
                pit_conditions.append(year_filter)
            
            if circuits:
                circuit_placeholders = ','.join([f"'{c}'" for c in circuits])
                circuit_filter = f"circuit IN ({circuit_placeholders})"
                race_conditions.append(circuit_filter)
                pit_conditions.append(circuit_filter)
            
            # Build queries
            race_query = "SELECT * FROM race_data"
            pit_query = "SELECT * FROM pit_stops"
            
            if race_conditions:
                race_query += " WHERE " + " AND ".join(race_conditions)
                pit_query += " WHERE " + " AND ".join(pit_conditions)
              # Load data
            race_data = pd.read_sql(race_query, self.engine)
            pit_data = pd.read_sql(pit_query, self.engine)
            
            logger.info(f"Loaded {len(race_data)} race records and {len(pit_data)} pit stop records")
            return race_data, pit_data
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _validate_session_data(self, session) -> bool:
        """
        Validate session data structure
        
        Args:
            session: FastF1 session object
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if session.laps.empty:
                return False
            
            required_columns = ['LapTime', 'Compound', 'Driver', 'LapNumber']
            for col in required_columns:
                if col not in session.laps.columns:
                    return False
            
            return True
        except Exception:
            return False
    
    def extract_session_data(self, year: int, circuit: str, session_type: str) -> Optional[pd.DataFrame]:
        """
        Extract session data for a specific year, circuit and session type
        
        Args:
            year: Race year
            circuit: Circuit identifier  
            session_type: Session type (Race, Qualifying, etc.)
            
        Returns:
            DataFrame with session data or None if failed
        """
        try:
            logger.info(f"Extracting {session_type} data for {year} {circuit}")
            
            # Load session
            session = fastf1.get_session(year, circuit, session_type)
            session.load(telemetry=False, weather=False, messages=False)
            
            # Validate session data
            if not self._validate_session_data(session):
                logger.warning(f"Invalid session data for {year} {circuit} {session_type}")
                return None
            
            # Process lap data
            laps = session.laps
            valid_laps = laps[
                (laps['LapTime'].dt.total_seconds() >= config.MIN_LAP_TIME) &
                (laps['LapTime'].dt.total_seconds() <= config.MAX_LAP_TIME) &
                (laps['TrackStatus'] == '1')  # Green flag only
            ].copy()
            
            if valid_laps.empty:
                logger.warning(f"No valid laps found for {year} {circuit} {session_type}")
                return None
            
            # Create DataFrame with required columns
            processed_data = []
            for _, lap in valid_laps.iterrows():
                processed_data.append({
                    'year': year,
                    'circuit': circuit,
                    'session_type': session_type,
                    'driver': lap['Driver'],
                    'lap_number': lap['LapNumber'],
                    'lap_time': lap['LapTime'].total_seconds(),
                    'compound': lap['Compound'],
                    'stint': lap.get('Stint', 1),
                    'tyre_life': lap['TyreLife'],
                    'track_status': lap['TrackStatus'],
                    'is_personal_best': lap.get('IsPersonalBest', False)
                })
            
            df = pd.DataFrame(processed_data)
            logger.info(f"Successfully extracted {len(df)} laps for {year} {circuit} {session_type}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract session data for {year} {circuit} {session_type}: {e}")
            return None
    
    def save_session_data(self, data: pd.DataFrame, year: int, circuit: str, session_type: str) -> bool:
        """
        Save session data to database
        
        Args:
            data: DataFrame with session data
            year: Race year
            circuit: Circuit identifier
            session_type: Session type
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            # Save to laps table
            data.to_sql('laps', self.engine, if_exists='append', index=False)
            logger.info(f"Saved {len(data)} lap records for {year} {circuit} {session_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
            return False
    
    def get_available_sessions(self) -> pd.DataFrame:
        """
        Get available sessions from database
        
        Returns:
            DataFrame with available sessions
        """
        try:
            query = """
            SELECT DISTINCT year, circuit, session_type 
            FROM laps 
            ORDER BY year DESC, circuit
            """
            sessions = pd.read_sql(query, self.engine)
            return sessions
        except Exception as e:
            logger.error(f"Failed to get available sessions: {e}")
            return pd.DataFrame()
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Load all lap data from database
        
        Returns:
            DataFrame with all lap data
        """
        try:
            query = "SELECT * FROM laps"
            data = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(data)} lap records from database")
            return data
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return pd.DataFrame()

    def extract_all_data(self) -> bool:
        """
        Extract data for all configured years and circuits
        
        Returns:
            True if successful, False otherwise
        """
        success_count = 0
        total_count = len(config.ANALYSIS_YEARS) * len(config.CIRCUITS)
        
        for year in config.ANALYSIS_YEARS:
            for circuit in config.CIRCUITS.keys():
                try:
                    logger.info(f"Processing {year} {circuit}")
                    
                    # Download race data
                    race_data = self.download_race_data(year, circuit)
                    if race_data is None:
                        continue
                    
                    # Extract pit stops
                    pit_data = self.extract_pit_stops(year, circuit)
                    
                    # Save to database
                    if self.save_to_database(race_data, pit_data):
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process {year} {circuit}: {e}")
                    continue
        
        logger.info(f"Successfully processed {success_count}/{total_count} year/circuit combinations")
        return success_count > 0


def main():
    """Main function for testing data extraction"""
    extractor = F1DataExtractor()
    
    # Extract all data
    if extractor.extract_all_data():
        logger.info("Data extraction completed successfully")
        
        # Load and display summary
        race_data, pit_data = extractor.load_combined_data()
        logger.info(f"Total records: {len(race_data)} race laps, {len(pit_data)} pit stops")
    else:
        logger.error("Data extraction failed")


if __name__ == "__main__":
    main()
