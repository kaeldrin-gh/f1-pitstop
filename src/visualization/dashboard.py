"""
F1 Pit Stop Strategy Dashboard
Streamlit dashboard for F1 strategy analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_extractor import F1DataExtractor
from src.data.preprocessor import F1DataPreprocessor
from src.models.tire_model import TireDegradationModel
from src.models.strategy_optimizer import PitStopOptimizer, RaceState, PitStop
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="F1 Pit Stop Strategy Tool",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #E10600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #E10600;
    }
    .strategy-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache F1 data"""
    try:
        extractor = F1DataExtractor()
        race_data, pit_data = extractor.load_combined_data()
        
        if race_data.empty:
            st.error("No race data available. Please run data extraction first.")
            return None, None
        
        return race_data, pit_data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None


@st.cache_resource
def load_models():
    """Load and cache ML models"""
    try:
        tire_model = TireDegradationModel()
        if not tire_model.load():
            st.warning("Tire model not found. Training new model...")
            # Train model if not found
            race_data, pit_data = load_data()
            if race_data is not None:
                preprocessor = F1DataPreprocessor()
                clean_data = preprocessor.validate_data(race_data)
                processed_data = preprocessor.engineer_features(clean_data, pit_data)
                train_data, test_data = preprocessor.split_data(processed_data)
                X_train, y_train = preprocessor.get_features_and_target(train_data)
                X_test, y_test = preprocessor.get_features_and_target(test_data)
                
                tire_model.fit(X_train, y_train)
                tire_model.evaluate(X_test, y_test)
                tire_model.save()
                st.success("Tire model trained and saved successfully!")
        
        optimizer = PitStopOptimizer(tire_model)
        return tire_model, optimizer
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None


def create_sidebar():
    """Create sidebar with controls"""
    st.sidebar.title("🏎️ F1 Strategy Controls")
    
    # Load data to get available options
    race_data, _ = load_data()
    if race_data is None:
        return {}
    
    # Year selection
    available_years = sorted(race_data['year'].unique())
    selected_year = st.sidebar.selectbox(
        "Select Year",
        available_years,
        index=len(available_years)-1  # Default to latest year
    )
    
    # Circuit selection
    year_data = race_data[race_data['year'] == selected_year]
    available_circuits = sorted(year_data['circuit'].unique())
    selected_circuit = st.sidebar.selectbox(
        "Select Circuit",
        available_circuits
    )
    
    # Driver selection
    circuit_data = year_data[year_data['circuit'] == selected_circuit]
    available_drivers = sorted(circuit_data['driver'].unique())
    selected_driver = st.sidebar.selectbox(
        "Select Driver",
        available_drivers
    )
    
    # Race simulation parameters
    st.sidebar.subheader("Race Simulation")
    current_lap = st.sidebar.slider("Current Lap", 1, 70, 25)
    total_laps = st.sidebar.slider("Total Laps", 40, 70, 58)
    current_position = st.sidebar.slider("Current Position", 1, 20, 8)
    
    # Tire selection
    tire_compounds = config.TIRE_COMPOUNDS
    current_tire = st.sidebar.selectbox("Current Tire Compound", tire_compounds, index=1)
    tire_age = st.sidebar.slider("Current Tire Age", 0, 30, 12)
    
    return {
        'year': selected_year,
        'circuit': selected_circuit,
        'driver': selected_driver,
        'current_lap': current_lap,
        'total_laps': total_laps,
        'current_position': current_position,
        'current_tire': current_tire,
        'tire_age': tire_age
    }


def strategy_optimizer_tab(params: Dict, tire_model: TireDegradationModel, optimizer: PitStopOptimizer):
    """Strategy Optimizer tab"""
    st.header("🎯 Pit Stop Strategy Optimizer")
    
    # Create race state
    race_state = RaceState(
        current_lap=params['current_lap'],
        total_laps=params['total_laps'],
        current_position=params['current_position'],
        current_tire_compound=params['current_tire'],
        current_tire_age=params['tire_age'],
        gap_to_leader=15.0,  # Default values
        gap_to_next=2.0,
        fuel_load=30.0
    )
    
    # Display current race state
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Lap", f"{race_state.current_lap}/{race_state.total_laps}")
    with col2:
        st.metric("Position", race_state.current_position)
    with col3:
        st.metric("Tire Compound", race_state.current_tire_compound)
    with col4:
        st.metric("Tire Age", f"{race_state.current_tire_age} laps")
    
    # Generate strategies
    if st.button("🔄 Generate Optimal Strategies", type="primary"):
        with st.spinner("Calculating optimal strategies..."):
            try:
                strategies = optimizer.generate_strategy_recommendations(
                    race_state, params['circuit']
                )
                
                if strategies:
                    st.success(f"Generated {len(strategies)} strategy recommendations!")
                    
                    # Display strategies
                    for i, strategy in enumerate(strategies):
                        with st.container():
                            st.markdown(f"""
                            <div class="strategy-card">
                                <h4>Strategy {i+1} - {strategy.risk_level} Risk</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Pit Stops", len(strategy.pit_stops))
                            with col2:
                                st.metric("Total Time", f"{strategy.total_time:.1f}s")
                            with col3:
                                st.metric("Final Position", strategy.final_position)
                            with col4:
                                st.metric("Confidence", f"{strategy.confidence_score:.0%}")
                            
                            # Display pit stop details
                            if strategy.pit_stops:
                                st.write("**Pit Stop Details:**")
                                for j, pit_stop in enumerate(strategy.pit_stops):
                                    st.write(f"Stop {j+1}: Lap {pit_stop.lap} - "
                                           f"{pit_stop.tire_compound_out} → {pit_stop.tire_compound_in}")
                            else:
                                st.write("**No pit stops recommended**")
                            
                            st.divider()
                
                else:
                    st.warning("No strategies generated. Check race parameters.")
                    
            except Exception as e:
                st.error(f"Failed to generate strategies: {e}")
                logger.error(f"Strategy generation error: {e}")


def tire_analysis_tab(params: Dict, tire_model: TireDegradationModel):
    """Tire Analysis tab"""
    st.header("🏁 Tire Performance Analysis")
    
    # Tire degradation curves
    st.subheader("Tire Degradation Curves")
    
    # Create degradation data
    laps = list(range(1, 31))
    degradation_data = []
    
    for compound in config.TIRE_COMPOUNDS:
        for lap in laps:
            try:
                lap_time = tire_model.predict_tire_degradation(compound, lap, params['circuit'])
            except:
                # Fallback calculation
                base_time = 90.0
                degradation_rate = config.TIRE_DEGRADATION_RATES[compound]
                lap_time = base_time + (lap * degradation_rate)
            
            degradation_data.append({
                'Lap': lap,
                'Compound': compound,
                'Lap Time': lap_time,
                'Degradation': lap_time - 90.0  # Assuming 90s base time
            })
    
    degradation_df = pd.DataFrame(degradation_data)
    
    # Plot degradation curves
    fig = px.line(
        degradation_df, 
        x='Lap', 
        y='Lap Time', 
        color='Compound',
        title=f"Tire Degradation at {params['circuit'].title()}",
        labels={'Lap Time': 'Lap Time (seconds)'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tire comparison table
    st.subheader("Tire Compound Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create comparison data
        comparison_data = []
        for compound in config.TIRE_COMPOUNDS:
            degradation_rate = config.TIRE_DEGRADATION_RATES[compound]
            comparison_data.append({
                'Compound': compound,
                'Degradation Rate': f"{degradation_rate:.2f} s/lap",
                'Performance at 10 laps': f"{90 + 10*degradation_rate:.1f}s",
                'Performance at 20 laps': f"{90 + 20*degradation_rate:.1f}s"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        # Optimal stint length analysis
        st.subheader("Optimal Stint Lengths")
        
        stint_data = []
        for compound in config.TIRE_COMPOUNDS:
            # Calculate optimal stint based on degradation
            degradation_rate = config.TIRE_DEGRADATION_RATES[compound]
            if degradation_rate <= 0.2:
                optimal_stint = "25-30 laps"
            elif degradation_rate <= 0.3:
                optimal_stint = "15-25 laps"
            else:
                optimal_stint = "10-15 laps"
            
            stint_data.append({
                'Compound': compound,
                'Optimal Stint': optimal_stint,
                'Max Recommended': f"{config.MAX_STINT_LENGTH} laps"
            })
        
        stint_df = pd.DataFrame(stint_data)
        st.dataframe(stint_df, use_container_width=True)


def race_simulation_tab(params: Dict, optimizer: PitStopOptimizer):
    """Race Simulation tab"""
    st.header("🏎️ Race Strategy Simulation")
    
    # What-if scenario controls
    st.subheader("What-If Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Safety Car Scenario**")
        safety_car_enabled = st.checkbox("Enable Safety Car")
        if safety_car_enabled:
            safety_car_lap = st.slider("Safety Car Lap", 1, params['total_laps'], 35)
        else:
            safety_car_lap = None
    
    with col2:
        st.write("**Weather Scenario**")
        weather_change = st.checkbox("Enable Weather Change")
        if weather_change:
            weather_lap = st.slider("Weather Change Lap", 1, params['total_laps'], 25)
            weather_type = st.selectbox("Weather Type", ["Rain", "Dry"])
        else:
            weather_lap = None
            weather_type = None
    
    # Strategy comparison
    st.subheader("Strategy Comparison")
    
    # Create sample strategies for comparison
    strategy_options = [
        "One-stop (Medium)",
        "One-stop (Hard)",
        "Two-stop (Medium-Soft)",
        "Conservative (Hard-Medium)",
        "Aggressive (Soft finish)"
    ]
    
    selected_strategies = st.multiselect(
        "Select Strategies to Compare",
        strategy_options,
        default=strategy_options[:3]
    )
    
    if st.button("🔄 Run Simulation", type="primary"):
        with st.spinner("Running race simulation..."):
            # Create sample results (in real implementation, use actual simulation)
            simulation_results = []
            
            for strategy_name in selected_strategies:
                # Simulate basic strategy performance
                base_time = 5400  # Base race time in seconds
                
                if "One-stop" in strategy_name:
                    total_time = base_time + np.random.normal(0, 30)
                    final_position = np.random.randint(5, 12)
                elif "Two-stop" in strategy_name:
                    total_time = base_time + np.random.normal(25, 45)
                    final_position = np.random.randint(3, 15)
                else:
                    total_time = base_time + np.random.normal(10, 35)
                    final_position = np.random.randint(4, 16)
                
                # Adjust for safety car
                if safety_car_enabled:
                    total_time -= np.random.uniform(5, 15)  # Benefit from safety car
                
                simulation_results.append({
                    'Strategy': strategy_name,
                    'Total Time': f"{total_time:.1f}s",
                    'Final Position': final_position,
                    'Time Gap': f"+{total_time - min([r.get('Total Time', 5400) if isinstance(r.get('Total Time'), (int, float)) else 5400 for r in simulation_results + [{'Total Time': total_time}]]):.1f}s"
                })
            
            # Display results
            if simulation_results:
                results_df = pd.DataFrame(simulation_results)
                st.dataframe(results_df, use_container_width=True)
                
                # Create visualization
                fig = go.Figure()
                
                for i, result in enumerate(simulation_results):
                    total_time_str = result['Total Time'].replace('s', '')
                    try:
                        total_time = float(total_time_str)
                    except:
                        total_time = 5400 + i * 10  # Fallback
                    
                    fig.add_trace(go.Bar(
                        x=[result['Strategy']],
                        y=[total_time],
                        name=result['Strategy'],
                        text=f"P{result['Final Position']}",
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Strategy Performance Comparison",
                    xaxis_title="Strategy",
                    yaxis_title="Total Race Time (seconds)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)


def historical_data_tab(params: Dict):
    """Historical Data tab"""
    st.header("📊 Historical Performance Analysis")
    
    # Load historical data
    race_data, pit_data = load_data()
    
    if race_data is not None and not race_data.empty:
        # Circuit performance over years
        st.subheader("Circuit Performance Trends")
        
        circuit_data = race_data[race_data['circuit'] == params['circuit']]
        
        if not circuit_data.empty:
            # Average lap times by year
            yearly_performance = circuit_data.groupby('year')['lap_time'].mean().reset_index()
            
            fig = px.line(
                yearly_performance,
                x='year',
                y='lap_time',
                title=f"Average Lap Times at {params['circuit'].title()}",
                labels={'lap_time': 'Average Lap Time (seconds)', 'year': 'Year'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tire compound usage
            st.subheader("Tire Compound Usage Analysis")
            
            compound_usage = circuit_data.groupby(['year', 'tire_compound']).size().reset_index(name='usage')
            
            fig = px.bar(
                compound_usage,
                x='year',
                y='usage',
                color='tire_compound',
                title=f"Tire Compound Usage at {params['circuit'].title()}",
                labels={'usage': 'Number of Laps', 'year': 'Year'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance by tire compound
            st.subheader("Performance by Tire Compound")
            
            compound_performance = circuit_data.groupby('tire_compound').agg({
                'lap_time': ['mean', 'std'],
                'tire_age': 'mean'
            }).round(3)
            
            compound_performance.columns = ['Average Lap Time', 'Std Dev', 'Average Tire Age']
            st.dataframe(compound_performance, use_container_width=True)
        
        else:
            st.warning(f"No historical data available for {params['circuit']}")
    
    # Pit stop analysis
    if pit_data is not None and not pit_data.empty:
        st.subheader("Pit Stop Analysis")
        
        circuit_pits = pit_data[pit_data['circuit'] == params['circuit']]
        
        if not circuit_pits.empty:
            # Pit stop timing distribution
            fig = px.histogram(
                circuit_pits,
                x='pit_lap',
                title=f"Pit Stop Timing Distribution at {params['circuit'].title()}",
                labels={'pit_lap': 'Pit Stop Lap', 'count': 'Frequency'},
                nbins=20
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Most common pit strategies
            pit_summary = circuit_pits.groupby(['tire_compound_out', 'tire_compound_in']).size().reset_index(name='count')
            pit_summary = pit_summary.sort_values('count', ascending=False).head(10)
            
            st.write("**Most Common Tire Changes:**")
            st.dataframe(pit_summary, use_container_width=True)
        
        else:
            st.warning(f"No pit stop data available for {params['circuit']}")


def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🏎️ F1 Pit Stop Strategy Tool</h1>', unsafe_allow_html=True)
    
    # Load models
    tire_model, optimizer = load_models()
    
    if tire_model is None or optimizer is None:
        st.error("Failed to load required models. Please check the setup.")
        return
    
    # Create sidebar
    params = create_sidebar()
    
    if not params:
        st.error("Failed to load control parameters.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Strategy Optimizer",
        "🏁 Tire Analysis", 
        "🏎️ Race Simulation",
        "📊 Historical Data"
    ])
    
    with tab1:
        strategy_optimizer_tab(params, tire_model, optimizer)
    
    with tab2:
        tire_analysis_tab(params, tire_model)
    
    with tab3:
        race_simulation_tab(params, optimizer)
    
    with tab4:
        historical_data_tab(params)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, FastF1, and scikit-learn | "
        "Data source: Formula 1 API via FastF1"
    )


if __name__ == "__main__":
    main()
