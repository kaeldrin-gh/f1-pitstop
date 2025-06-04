## General

* **Write modular, readable code:**
  * Use type hints for all functions.
  * Add clear, informative docstrings for every function/class.
  * Follow PEP8 naming and formatting.
* **Add `__init__.py` to all Python package folders.**
* **Handle exceptions and errors clearly with logs or user messages.**
* **Add unit tests for critical logic (aim for 80%+ coverage).**

---

## Data

* **All inter-module data is pandas DataFrame unless otherwise specified.**
* **Document all function/class inputs/outputs with types and brief purpose.**
* **Validate data before saving (disk/database).**

---

## Data Pipeline

* **Extract data using FastF1 per years/circuits in `config.py`.**
* **Log all downloads and extractions.**
* **Handle API errors, rate limits, and missing data robustly.**
* **Preprocess:**

  * Remove outliers, pit laps, missing/invalid entries.
  * Engineer features as described.
  * Encode categorical features as needed.
  * Chronologically split train/test data (last 20% test).

---

## ML Models

* **TireDegradationModel:**

  * Use RandomForestRegressor (100, max\_depth=10).
  * Include fit, predict, save, load.
  * Evaluate/log R², MAE, RMSE.
* **Strategy Optimizer:**

  * Simulate and compare strategies using model/config.
  * Log all outputs and handle all edge cases.

---

## Streamlit Dashboard

* **App must:**

  * Provide sidebar controls for year/circuit/driver/tire.
  * Implement all tabs: Strategy Optimizer, Tire Analysis, Race Simulation, Historical Data.
  * Use Plotly/Seaborn for charts, ensure interactivity.
  * Display errors and loading states.
  * Be responsive on desktop.

---

## Data Validation

* **Use only lap times 60–180s; remove Safety Car/formation laps.**
* **No tire stint >50 laps; all data must have compound and timing.**
* **Log/report failed data checks.**

---

## Documentation

* **README.md must include:**
  Project overview, setup, usage, architecture, metrics, improvements.
* **Code:**

  * Docstrings and type hints everywhere.
  * Examples in docstrings when relevant.
* **Tests:**

  * Provide expected results in docstrings.

---

## Output/Performance

* **Models must meet:**

  * Tire model R²>0.75, MAE<0.5s.
  * Strategy optimizer >80% accuracy vs actual.
* **Streamlit app must run error-free on localhost:8501.**
* **All outputs must be reproducible from raw data with scripts.**

---

## Copilot Behaviors

* **Never skip validation or error checks.**
* **Never add dependencies or change configs unless told.**
* **If a requirement is ambiguous, prompt for clarification.**
* **Strive for maintainable, well-documented, robust solutions.**
* **Respond only with code or file edits—no extra commentary.**

