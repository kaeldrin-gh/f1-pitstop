# F1 Pit Stop Strategy Tool 🏎️

A comprehensive F1 pit stop strategy optimization application featuring an end-to-end data pipeline, machine learning models, and interactive Streamlit dashboard.

## 🎯 Project Overview

This tool analyzes Formula 1 race data to optimize pit stop strategies using machine learning. It provides real-time strategy recommendations, tire degradation analysis, and race simulation capabilities.

### Key Features

- **Data Pipeline**: Automated F1 data extraction using FastF1 API
- **Machine Learning**: Tire degradation prediction model (RandomForest)
- **Strategy Optimization**: AI-powered pit stop strategy recommendations
- **Interactive Dashboard**: Streamlit web application with multiple analysis tabs
- **Race Simulation**: What-if scenario analysis and strategy comparison

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- 4GB RAM minimum
- Internet connection for data extraction

### Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd "F1 Pitstop Anaylsis"
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```powershell
   python main.py --dashboard
   ```

5. **Access the dashboard**
   - Open browser to `http://localhost:8501`
   - Use sidebar controls to select race parameters
   - Explore different analysis tabs

## 📊 Dashboard Features

### 🎯 Strategy Optimizer
- Real-time pit stop recommendations
- Optimal pit window calculations
- Risk assessment and confidence scores
- Multiple strategy comparisons

### 🏁 Tire Analysis
- Tire degradation curves by compound
- Performance comparison charts
- Optimal stint length recommendations
- Interactive Plotly visualizations

### 🏎️ Race Simulation
- What-if scenario analysis
- Safety car impact assessment
- Weather change simulations
- Strategy performance comparison

### 📈 Historical Data
- Multi-year performance trends
- Circuit-specific analysis
- Tire compound usage patterns
- Pit stop timing statistics

## 🛠️ Technical Architecture

### Data Pipeline
```
FastF1 API → Data Extraction → Data Cleaning → Feature Engineering → Model Training
```

### Components
- **Data Layer**: SQLite database with normalized schema
- **ML Layer**: RandomForest tire degradation model
- **Strategy Layer**: Optimization algorithms and race simulation
- **UI Layer**: Streamlit dashboard with interactive visualizations

### Performance Metrics
- **Tire Model**: R² > 0.75, MAE < 0.5s
- **Strategy Optimizer**: >80% accuracy vs actual F1 decisions
- **Training Time**: <5 minutes on standard hardware
- **Response Time**: <2 seconds for strategy generation

## 📁 Project Structure

```
F1 Pitstop Analysis/
├── main.py                 # Main application entry point
├── config.py              # Configuration and constants
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── data/                 # Data storage
│   ├── raw/             # Raw extracted data
│   ├── processed/       # Cleaned and processed data
│   └── database/        # SQLite database files
│
├── src/                 # Source code
│   ├── data/           # Data pipeline modules
│   │   ├── data_extractor.py    # FastF1 data extraction
│   │   └── preprocessor.py      # Data cleaning and feature engineering
│   │
│   ├── models/         # Machine learning models
│   │   ├── tire_model.py        # Tire degradation model
│   │   └── strategy_optimizer.py # Strategy optimization
│   │
│   ├── visualization/  # Dashboard and charts
│   │   └── dashboard.py          # Streamlit dashboard
│   │
│   └── utils/          # Utility functions
│       └── helpers.py            # Common utilities
│
├── models/             # Saved ML models
├── notebooks/          # Jupyter notebooks for analysis
└── tests/             # Unit tests
```

## 🎮 Usage Examples

### Command Line Interface

```powershell
# Extract fresh data
python main.py --extract

# Retrain models
python main.py --train

# Run analysis only
python main.py --analysis

# Launch dashboard on custom port
python main.py --dashboard --port 8502

# Show system information
python main.py --info
```

### Programmatic Usage

```python
from src.models.tire_model import TireDegradationModel
from src.models.strategy_optimizer import PitStopOptimizer, RaceState

# Load trained model
model = TireDegradationModel()
model.load()

# Create optimizer
optimizer = PitStopOptimizer(model)

# Define race state
race_state = RaceState(
    current_lap=25,
    total_laps=58,
    current_position=8,
    current_tire_compound='MEDIUM',
    current_tire_age=12,
    gap_to_leader=15.0,
    gap_to_next=2.0,
    fuel_load=30.0
)

# Generate strategies
strategies = optimizer.generate_strategy_recommendations(race_state)
```

## 📊 Data Sources

- **Primary**: Formula 1 race data via FastF1 API
- **Coverage**: 2021-2024 seasons
- **Circuits**: Monaco, Silverstone, Monza, Spa-Francorchamps, Suzuka
- **Data Points**: Lap times, tire compounds, pit stops, track conditions

### Data Validation
- Lap times: 60-180 seconds only
- Tire stints: Maximum 50 laps
- Automatic outlier removal using IQR method
- Real-time data quality monitoring

## 🧠 Machine Learning Models

### Tire Degradation Model
- **Algorithm**: Random Forest Regressor
- **Features**: Tire age, compound, track conditions, driver, circuit
- **Target**: Lap time prediction
- **Performance**: R² > 0.75, MAE < 0.5s

### Strategy Optimizer
- **Approach**: Simulation-based optimization
- **Factors**: Track position, tire performance, pit stop timing
- **Output**: Ranked strategy recommendations with confidence scores

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Analysis parameters
ANALYSIS_YEARS = [2021, 2022, 2023, 2024]
CIRCUITS = ['monaco', 'silverstone', 'monza', 'spa', 'suzuka']

# Model parameters
TIRE_DEGRADATION_RATES = {
    'SOFT': 0.35,    # seconds per lap
    'MEDIUM': 0.25,
    'HARD': 0.15
}

# Performance thresholds
MIN_LAP_TIME = 60.0
MAX_LAP_TIME = 180.0
PIT_STOP_PENALTY = 25.0
```

## 🧪 Testing

Run the test suite:
```powershell
python -m pytest tests/ -v
```

Test coverage target: 80%+

## 📈 Performance Optimization

### Recommended System Requirements
- **CPU**: 4+ cores for parallel processing
- **RAM**: 8GB+ for large dataset handling
- **Storage**: 2GB+ for data and models
- **Network**: Stable connection for FastF1 API

### Optimization Features
- Data caching for repeated queries
- Model persistence to avoid retraining
- Efficient DataFrame operations with pandas
- Parallel processing for strategy simulation

## 🔍 Troubleshooting

### Common Issues

**1. FastF1 API Errors**
```
Solution: Check internet connection and API rate limits
```

**2. Model Performance Below Threshold**
```
Solution: Increase data volume or retrain with --train flag
```

**3. Dashboard Not Loading**
```
Solution: Ensure port 8501 is available or use --port flag
```

**4. Memory Issues**
```
Solution: Reduce ANALYSIS_YEARS in config.py or increase system RAM
```

### Debug Mode
```powershell
# Enable verbose logging
python main.py --dashboard --verbose
```

## 🚧 Future Enhancements

### Planned Features
- [ ] Real-time race integration
- [ ] Multi-driver strategy coordination
- [ ] Weather prediction integration
- [ ] Advanced risk modeling
- [ ] Mobile-responsive dashboard
- [ ] API endpoints for external integration

### Performance Improvements
- [ ] GPU acceleration for model training
- [ ] Distributed computing support
- [ ] Advanced caching strategies
- [ ] Real-time data streaming

### Analysis Enhancements
- [ ] Driver-specific modeling
- [ ] Car setup impact analysis
- [ ] Championship scenario planning
- [ ] Comparative team analysis

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Follow coding standards (PEP8, type hints, docstrings)
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

### Code Standards
- Type hints for all functions
- Comprehensive docstrings
- Unit tests with 80%+ coverage
- Logging for all major operations
- Error handling and validation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastF1**: Excellent F1 data API and Python library
- **Formula 1**: Official data source
- **Streamlit**: Interactive dashboard framework
- **scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check troubleshooting section
- Review existing documentation

---

**Built with ❤️ for Formula 1 strategy analysis**

*Disclaimer: This tool is for educational and analysis purposes. Not affiliated with Formula 1 or any F1 teams.*
