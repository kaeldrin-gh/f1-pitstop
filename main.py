"""
F1 Pit Stop Strategy Tool - Main Application
Entry point for the F1 strategy analysis application
"""

import logging
import argparse
import sys
import os
from typing import Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_extractor import F1DataExtractor
from src.data.preprocessor import F1DataPreprocessor
from src.models.tire_model import TireDegradationModel
from src.models.strategy_optimizer import PitStopOptimizer
from src.utils.helpers import setup_logging, performance_timer
import config

# Configure logging
setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class F1StrategyApp:
    """
    Main F1 Pit Stop Strategy Application
    """
    
    def __init__(self):
        """Initialize the application"""
        self.extractor = None
        self.preprocessor = None
        self.tire_model = None
        self.optimizer = None
        
        logger.info("F1 Pit Stop Strategy Tool initialized")
    
    @performance_timer
    def setup_data_pipeline(self, force_extract: bool = False) -> bool:
        """
        Setup and run the data pipeline
        
        Args:
            force_extract: Force re-extraction of data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Setting up data pipeline...")
        
        try:
            # Initialize data extractor
            self.extractor = F1DataExtractor()
            
            # Check if data exists
            race_data, pit_data = self.extractor.load_combined_data()
            
            if race_data.empty or force_extract:
                logger.info("Extracting F1 data from FastF1 API...")
                if not self.extractor.extract_all_data():
                    logger.error("Failed to extract data")
                    return False
                
                # Reload data after extraction
                race_data, pit_data = self.extractor.load_combined_data()
            
            if race_data.empty:
                logger.error("No race data available after extraction")
                return False
            
            logger.info(f"Loaded {len(race_data)} race records and {len(pit_data)} pit stop records")
            
            # Initialize preprocessor
            self.preprocessor = F1DataPreprocessor()
            
            # Validate and clean data
            logger.info("Validating and cleaning data...")
            clean_data = self.preprocessor.validate_data(race_data)
            
            if clean_data.empty:
                logger.error("No valid data after cleaning")
                return False
            
            # Engineer features
            logger.info("Engineering features...")
            processed_data = self.preprocessor.engineer_features(clean_data, pit_data)
            
            if processed_data.empty:
                logger.error("No data after feature engineering")
                return False
            
            # Validate processed data
            metrics = self.preprocessor.validate_processed_data(processed_data)
            
            if metrics['total_rows'] < 100:
                logger.warning(f"Low data volume: {metrics['total_rows']} rows")
            
            logger.info("Data pipeline setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup data pipeline: {e}")
            return False
    
    @performance_timer
    def train_models(self, retrain: bool = False) -> bool:
        """
        Train machine learning models
        
        Args:
            retrain: Force retraining of models
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Setting up machine learning models...")
        
        try:
            # Initialize tire model
            self.tire_model = TireDegradationModel()
            
            # Try to load existing model
            if not retrain and self.tire_model.load():
                logger.info("Loaded existing tire degradation model")
                
                # Validate model performance
                metrics = self.tire_model.model_metrics
                if metrics.get('test_r2', 0) < 0.75:
                    logger.warning("Model performance below threshold, retraining...")
                    retrain = True
            else:
                retrain = True
            
            if retrain:
                logger.info("Training tire degradation model...")
                
                # Load processed data
                race_data, pit_data = self.extractor.load_combined_data()
                clean_data = self.preprocessor.validate_data(race_data)
                processed_data = self.preprocessor.engineer_features(clean_data, pit_data)
                
                # Split data
                train_data, test_data = self.preprocessor.split_data(processed_data)
                X_train, y_train = self.preprocessor.get_features_and_target(train_data)
                X_test, y_test = self.preprocessor.get_features_and_target(test_data)
                
                if X_train.empty or y_train.empty:
                    logger.error("No training data available")
                    return False
                
                # Train model
                train_metrics = self.tire_model.fit(X_train, y_train)
                  # Evaluate model
                if not X_test.empty and not y_test.empty:
                    test_metrics = self.tire_model.evaluate(X_test, y_test)
                    
                    # Check performance requirements
                    if test_metrics['r2_score'] < 0.75:
                        logger.warning(f"Model R² ({test_metrics['r2_score']:.4f}) below target (0.75)")
                    if test_metrics['mae'] > 0.5:
                        logger.warning(f"Model MAE ({test_metrics['mae']:.4f}) above target (0.5)")
                else:
                    logger.warning("No test data available for evaluation")
                
                # Save model
                try:
                    saved_path = self.tire_model.save()
                    logger.info(f"Model saved successfully to {saved_path}")
                except Exception as e:
                    logger.error(f"Failed to save trained model: {e}")
                    return False
            
            # Initialize strategy optimizer
            self.optimizer = PitStopOptimizer(self.tire_model)
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return False
    
    def run_dashboard(self, port: int = config.STREAMLIT_PORT) -> None:
        """
        Launch the Streamlit dashboard
        
        Args:
            port: Port to run dashboard on
        """
        logger.info(f"Launching Streamlit dashboard on port {port}")
        
        try:
            import subprocess
            import sys
            
            # Path to dashboard script
            dashboard_path = os.path.join("src", "visualization", "dashboard.py")
            
            # Launch Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                dashboard_path, "--server.port", str(port)
            ]
            
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to launch dashboard: {e}")
        except ImportError:
            logger.error("Streamlit not installed. Please install with: pip install streamlit")
        except Exception as e:
            logger.error(f"Unexpected error launching dashboard: {e}")
    
    def run_analysis(self) -> None:
        """
        Run basic analysis and display results
        """
        logger.info("Running F1 strategy analysis...")
        
        try:
            if not self.tire_model or not self.optimizer:
                logger.error("Models not initialized. Run setup first.")
                return
            
            # Display model information
            model_info = self.tire_model.get_model_info()
            logger.info(f"Tire Model Info:")
            logger.info(f"  Features: {model_info['n_features']}")
            logger.info(f"  Performance: R²={model_info['metrics'].get('test_r2', 'N/A'):.4f}")
            
            # Example tire degradation predictions
            logger.info("Example tire degradation predictions:")
            for compound in config.TIRE_COMPOUNDS:
                for age in [5, 15, 25]:
                    degradation = self.tire_model.predict_tire_degradation(compound, age)
                    logger.info(f"  {compound} at {age} laps: {degradation:.3f}s")
              # Feature importance
            importance_df = self.tire_model.get_feature_importance(5)
            logger.info("Top 5 feature importance:")
            for _, row in importance_df.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to run analysis: {e}")
    
    def get_system_info(self) -> dict:
        """
        Get system and application information
        
        Returns:
            System information dictionary
        """
        import psutil
        import platform
        
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                'memory_available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
                'disk_free': f"{psutil.disk_usage('/').free / (1024**3):.1f} GB",
                'models_loaded': {
                    'tire_model': self.tire_model is not None and self.tire_model.is_fitted,
                    'optimizer': self.optimizer is not None
                }
            }
        except ImportError:
            return {'error': 'psutil not available for system info'}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="F1 Pit Stop Strategy Tool")
    parser.add_argument("--extract", action="store_true", help="Force data extraction")
    parser.add_argument("--train", action="store_true", help="Force model retraining")
    parser.add_argument("--analysis", action="store_true", help="Run analysis only")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    parser.add_argument("--port", type=int, default=config.STREAMLIT_PORT, help="Dashboard port")
    parser.add_argument("--info", action="store_true", help="Show system information")
    
    args = parser.parse_args()
    
    # Initialize application
    app = F1StrategyApp()
    
    # Show system information if requested
    if args.info:
        info = app.get_system_info()
        logger.info("System Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        return
    
    # Setup data pipeline
    logger.info("Starting F1 Pit Stop Strategy Tool...")
    if not app.setup_data_pipeline(force_extract=args.extract):
        logger.error("Failed to setup data pipeline")
        sys.exit(1)
    
    # Train models
    if not app.train_models(retrain=args.train):
        logger.error("Failed to train models")
        sys.exit(1)
    
    # Run analysis only
    if args.analysis:
        app.run_analysis()
        return
    
    # Launch dashboard
    if args.dashboard or len(sys.argv) == 1:  # Default action
        logger.info("Launching dashboard...")
        app.run_dashboard(args.port)
    else:
        app.run_analysis()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
