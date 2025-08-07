"""
EV Health Monitoring Dashboard

Interactive Streamlit dashboard for monitoring EV health, predictions, and analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.prediction import EVHealthPredictor
from src.data.data_loader import load_all_datasets

# Page config
st.set_page_config(
    page_title="EV Health Monitoring Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .health-excellent { border-left-color: #2ecc71; }
    .health-good { border-left-color: #27ae60; }
    .health-fair { border-left-color: #f39c12; }
    .health-poor { border-left-color: #e67e22; }
    .health-critical { border-left-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the basic RF model directly using joblib."""
    try:
        import joblib
        from pathlib import Path
        
        # Use the basic model that works with 26 features instead of 136
        model_path = Path("models/rul_rf_model.joblib")  # Changed from enhanced_rf_model.joblib
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None
            
        # Load model directly
        model = joblib.load(model_path)
        # st.success(f"✅ Basic RF model loaded successfully from {model_path}")  # Commented out to remove message
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_dashboard_data():
    """Load data for dashboard."""
    try:
        # Try to load existing dashboard data
        if Path("dashboard_data.csv").exists():
            return pd.read_csv("dashboard_data.csv")
        elif Path("final_predictions.csv").exists():
            return pd.read_csv("final_predictions.csv").head(1000)  # Limit for performance
        else:
            # Load features data if available
            if Path("data/features/features_for_modeling.csv").exists():
                return pd.read_csv("data/features/features_for_modeling.csv").head(1000)
            else:
                return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_fleet_summary(df):
    """Get fleet summary statistics."""
    if df is None:
        return {}
        
    summary = {}
    
    # Health score stats
    if 'predicted_health_score' in df.columns:
        summary['avg_health'] = df['predicted_health_score'].mean()
        summary['min_health'] = df['predicted_health_score'].min()
        summary['max_health'] = df['predicted_health_score'].max()
        
        # Health categories
        if 'health_category' in df.columns:
            health_counts = df['health_category'].value_counts()
            summary['health_distribution'] = health_counts.to_dict()
    
    # Battery stats
    if 'SOH' in df.columns:
        summary['avg_soh'] = df['SOH'].mean()
    if 'SOC' in df.columns:
        summary['avg_soc'] = df['SOC'].mean()
        
    # Temperature stats
    if 'Battery_Temp' in df.columns:
        summary['avg_battery_temp'] = df['Battery_Temp'].mean()
    if 'Motor_Temp' in df.columns:
        summary['avg_motor_temp'] = df['Motor_Temp'].mean()
        
    return summary

def create_health_gauge(health_score, title="Health Score"):
    """Create a gauge chart for health score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = health_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_health_distribution_chart(df):
    """Create health distribution chart."""
    if 'health_category' in df.columns:
        health_counts = df['health_category'].value_counts()
        colors = {'excellent': '#2ecc71', 'good': '#27ae60', 'fair': '#f39c12', 
                 'poor': '#e67e22', 'critical': '#e74c3c'}
        
        fig = px.pie(
            values=health_counts.values,
            names=health_counts.index,
            title="Fleet Health Distribution",
            color=health_counts.index,
            color_discrete_map=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    return None

def create_temporal_analysis(df):
    """Create temporal analysis charts."""
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['hour'] = df['Timestamp'].dt.hour
        df['date'] = df['Timestamp'].dt.date
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Average Health Score', 'SOC Distribution by Hour',
                          'Daily Temperature Averages', 'Health Category Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily average health score (much cleaner)
        if 'predicted_health_score' in df.columns:
            daily_health = df.groupby('date')['predicted_health_score'].agg(['mean', 'std']).reset_index()
            daily_health['date'] = pd.to_datetime(daily_health['date'])
            
            fig.add_trace(
                go.Scatter(x=daily_health['date'], y=daily_health['mean'],
                          mode='lines+markers', name='Daily Avg Health',
                          line=dict(color='#1f77b4', width=3),
                          marker=dict(size=6)),
                row=1, col=1
            )
            
            # Add confidence band
            fig.add_trace(
                go.Scatter(x=daily_health['date'], 
                          y=daily_health['mean'] + daily_health['std'],
                          mode='lines', line=dict(width=0), showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=daily_health['date'], 
                          y=daily_health['mean'] - daily_health['std'],
                          mode='lines', line=dict(width=0), 
                          fill='tonexty', fillcolor='rgba(31,119,180,0.2)',
                          name='±1 Std Dev', showlegend=False),
                row=1, col=1
            )
        
        # SOC distribution by hour (cleaner bar chart)
        if 'SOC' in df.columns:
            hourly_soc = df.groupby('hour')['SOC'].agg(['mean', 'count']).reset_index()
            fig.add_trace(
                go.Bar(x=hourly_soc['hour'], y=hourly_soc['mean'], 
                      name='Avg SOC by Hour',
                      marker_color='#2ca02c',
                      text=[f'{x:.1f}%' for x in hourly_soc['mean']],
                      textposition='outside'),
                row=1, col=2
            )
        
        # Daily temperature averages (much cleaner)
        if 'Battery_Temp' in df.columns:
            daily_temp = df.groupby('date').agg({
                'Battery_Temp': 'mean',
                'Motor_Temp': 'mean' if 'Motor_Temp' in df.columns else lambda x: None
            }).reset_index()
            daily_temp['date'] = pd.to_datetime(daily_temp['date'])
            
            fig.add_trace(
                go.Scatter(x=daily_temp['date'], y=daily_temp['Battery_Temp'],
                          mode='lines+markers', name='Battery Temp',
                          line=dict(color='#ff7f0e', width=2),
                          marker=dict(size=4)),
                row=2, col=1
            )
            
            if 'Motor_Temp' in df.columns and daily_temp['Motor_Temp'].notna().any():
                fig.add_trace(
                    go.Scatter(x=daily_temp['date'], y=daily_temp['Motor_Temp'],
                              mode='lines+markers', name='Motor Temp',
                              line=dict(color='#d62728', width=2),
                              marker=dict(size=4)),
                    row=2, col=1
                )
        
        # Health category bar chart (compatible with xy subplot)
        if 'health_category' in df.columns:
            category_counts = df['health_category'].value_counts()
            colors = {'excellent': '#2ecc71', 'good': '#27ae60', 'fair': '#f39c12', 
                     'poor': '#e67e22', 'critical': '#e74c3c'}
            
            fig.add_trace(
                go.Bar(x=category_counts.index, y=category_counts.values,
                      name="Health Categories",
                      marker_color=[colors.get(cat, '#cccccc') for cat in category_counts.index],
                      text=category_counts.values,
                      textposition='outside'),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="Fleet Analytics Overview")
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Health Score", row=1, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
        fig.update_yaxes(title_text="SOC (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_xaxes(title_text="Health Category", row=2, col=2)
        fig.update_yaxes(title_text="Vehicle Count", row=2, col=2)
        
        return fig
    return None

def create_predictive_analytics(df):
    """Create predictive analytics visualizations."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Health vs Battery Performance', 'Temperature Health Impact',
                       'Health Score Distribution', 'Top 10 Maintenance Priorities'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Health Score vs SOH (cleaner scatter with trend)
    if all(col in df.columns for col in ['predicted_health_score', 'SOH']):
        # Sample data for cleaner visualization
        sample_df = df.sample(n=min(500, len(df)), random_state=42)
        
        fig.add_trace(
            go.Scatter(x=sample_df['SOH'], y=sample_df['predicted_health_score'],
                      mode='markers', name='Health vs SOH',
                      marker=dict(
                          size=8,
                          color=sample_df['predicted_health_score'],
                          colorscale='RdYlGn',
                          showscale=True,
                          colorbar=dict(title="Health Score"),
                          opacity=0.7
                      )),
            row=1, col=1
        )
    
    # Temperature impact on health (binned analysis)
    if all(col in df.columns for col in ['Battery_Temp', 'predicted_health_score']):
        # Create temperature bins
        df['temp_bin'] = pd.cut(df['Battery_Temp'], bins=8, labels=[f'{i*5+15}-{(i+1)*5+15}°C' for i in range(8)])
        temp_health = df.groupby('temp_bin')['predicted_health_score'].agg(['mean', 'count']).reset_index()
        temp_health = temp_health[temp_health['count'] >= 10]  # Only bins with enough data
        
        if not temp_health.empty:
            fig.add_trace(
                go.Bar(x=temp_health['temp_bin'], y=temp_health['mean'],
                      name='Avg Health by Temp',
                      marker_color=['#d62728' if x < 0.3 else '#ff7f0e' if x < 0.6 else '#2ca02c' for x in temp_health['mean']],
                      text=[f'{x:.2f}' for x in temp_health['mean']],
                      textposition='outside'),
                row=1, col=2
            )
    
    # Health score distribution histogram
    if 'predicted_health_score' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['predicted_health_score'], nbinsx=20,
                        name='Health Distribution',
                        marker_color='#1f77b4',
                        opacity=0.7),
            row=2, col=1
        )
    
    # Top 10 maintenance priorities (cleaner bar chart)
    if 'predicted_health_score' in df.columns:
        df_with_priority = df.copy()
        df_with_priority['maintenance_priority'] = 1 - df_with_priority['predicted_health_score']
        priority_df = df_with_priority.nlargest(10, 'maintenance_priority').reset_index()
        
        if not priority_df.empty:
            vehicle_ids = [f"Vehicle {i+1}" for i in range(len(priority_df))]
            priority_scores = priority_df['maintenance_priority'].values
            
            fig.add_trace(
                go.Bar(x=vehicle_ids, y=priority_scores,
                      name='Maintenance Priority',
                      marker_color=['#e74c3c' if x > 0.8 else '#e67e22' if x > 0.6 else '#f39c12' for x in priority_scores],
                      text=[f'{x:.2f}' for x in priority_scores],
                      textposition='outside'),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=True, title_text="Predictive Analytics Dashboard")
    
    # Update axes labels
    fig.update_xaxes(title_text="State of Health (%)", row=1, col=1)
    fig.update_yaxes(title_text="Health Score", row=1, col=1)
    fig.update_xaxes(title_text="Temperature Range", row=1, col=2)
    fig.update_yaxes(title_text="Average Health Score", row=1, col=2)
    fig.update_xaxes(title_text="Health Score", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Vehicle", row=2, col=2)
    fig.update_yaxes(title_text="Priority Score", row=2, col=2)
    
    return fig

# Main Dashboard
def main():
    # Header
    st.markdown('<div class="main-header">🚗 EV Health Monitoring Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Fleet Overview", "🔍 Individual Vehicle Prediction"])
    
    with tab1:
        # Load data
        df = load_dashboard_data()
        
        if df is None:
            st.error("⚠️ No data available. Please run predictions first using: `python main.py predict --input-file data/features/features_for_modeling.csv`")
            return
        
        # Sidebar filters
        st.sidebar.header("🔧 Filters & Controls")
        
        # Sample size control
        max_records = len(df)
        sample_size = st.sidebar.slider("Sample Size", 100, min(max_records, 10000), min(1000, max_records))
        df_sample = df.head(sample_size)
        
        # Health category filter
        if 'health_category' in df.columns:
            categories = ['All'] + list(df['health_category'].unique())
            selected_category = st.sidebar.selectbox("Health Category", categories)
            if selected_category != 'All':
                df_sample = df_sample[df_sample['health_category'] == selected_category]
        
        # Get summary stats
        summary = get_fleet_summary(df_sample)
        
        # Main dashboard content for fleet overview
        fleet_overview_content(df_sample, summary)
    
    with tab2:
        # Individual vehicle prediction interface
        individual_prediction_interface()

def fleet_overview_content(df_sample, summary):
    """Content for the fleet overview tab."""
def fleet_overview_content(df_sample, summary):
    """Content for the fleet overview tab."""
    # Main dashboard
    if not df_sample.empty:
        # KPI Row
        st.subheader("📊 Fleet Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if 'avg_health' in summary:
                st.metric("Avg Health Score", f"{summary['avg_health']:.2f}")
            else:
                st.metric("Total Vehicles", len(df_sample))
        
        with col2:
            if 'avg_soh' in summary:
                st.metric("Avg SOH", f"{summary['avg_soh']:.1f}%")
            else:
                st.metric("Features", len(df_sample.columns))
        
        with col3:
            if 'avg_soc' in summary:
                st.metric("Avg SOC", f"{summary['avg_soc']:.1f}%")
            else:
                st.metric("Data Points", len(df_sample))
        
        with col4:
            if 'avg_battery_temp' in summary:
                st.metric("Avg Battery Temp", f"{summary['avg_battery_temp']:.1f}°C")
            else:
                st.metric("Sample Size", len(df_sample))
        
        with col5:
            if 'health_distribution' in summary:
                critical_count = summary['health_distribution'].get('critical', 0)
                st.metric("Critical Vehicles", critical_count, delta_color="inverse")
            else:
                st.metric("Time Span", "5 years")
        
        # Main charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Health Score Gauge")
            if 'avg_health' in summary:
                gauge_fig = create_health_gauge(summary['avg_health'], "Fleet Average Health")
                st.plotly_chart(gauge_fig, use_container_width=True)
            else:
                st.info("Health score data not available")
        
        with col2:
            st.subheader("📈 Health Distribution")
            dist_fig = create_health_distribution_chart(df_sample)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
            else:
                st.info("Health category data not available")
        
        # Temporal Analysis
        st.subheader("⏰ Fleet Analytics Overview")
        temporal_fig = create_temporal_analysis(df_sample)
        if temporal_fig:
            st.plotly_chart(temporal_fig, use_container_width=True)
        else:
            st.info("Temporal data not available")
        
        # Predictive Analytics
        st.subheader("🔮 Advanced Predictive Insights")
        pred_fig = create_predictive_analytics(df_sample)
        if pred_fig:
            st.plotly_chart(pred_fig, use_container_width=True)
        
        # Data Explorer
        st.subheader("🔍 Data Explorer")
        
        # Feature selection for correlation matrix
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            selected_features = st.multiselect(
                "Select features for correlation analysis:", 
                numeric_cols, 
                default=numeric_cols[:5]
            )
            
            if selected_features:
                corr_matrix = df_sample[selected_features].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Feature Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
        
        # Raw data view
        if st.checkbox("Show raw data"):
            st.dataframe(df_sample)
        
        # Download option
        csv = df_sample.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f'ev_health_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
    
    else:
        st.warning("No data available with current filters.")

def individual_prediction_interface():
    """Interface for individual vehicle health prediction."""
    st.subheader("🔍 Individual Vehicle Health Prediction")
    st.write("Enter vehicle parameters below to get a real-time health score prediction.")
    
    # Initialize session state for random values if not exists
    if 'random_params' not in st.session_state:
        st.session_state.random_params = {}
    
    # Randomizer button (outside form to avoid conflicts)
    if st.button("🎲 Randomize All Parameters", help="Generate random test parameters"):
        import random
        st.session_state.random_params = {
            'soc': random.randint(5, 95),
            'soh': random.randint(60, 100),
            'battery_temp': random.randint(15, 55),
            'battery_voltage': random.randint(320, 420),
            'battery_current': random.randint(-150, 150),
            'charging_cycles': random.randint(50, 4500),
            'motor_temp': random.randint(25, 75),
            'motor_rpm': random.randint(0, 7500),
            'motor_torque': random.randint(50, 450),
            'power_consumption': random.randint(5, 45),
            'driving_speed': random.randint(20, 140),
            'ambient_temp': random.randint(-10, 45),
            'ambient_humidity': random.randint(30, 95),
            'brake_pressure': random.randint(20, 150),
            'tire_pressure': round(random.uniform(1.5, 3.0), 1),
            'distance_traveled': random.randint(5000, 300000),
            'maintenance_days': random.randint(5, 300)
        }
        st.rerun()
    
    # Create input form
    with st.form("vehicle_prediction_form", clear_on_submit=False):
        st.subheader("📝 Vehicle Parameters")
        
        # Add a timestamp to ensure fresh predictions
        import time
        form_key = f"form_{int(time.time())}"
        
        # Create three columns for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Battery & Charging**")
            soc = st.slider("State of Charge (%)", 0, 100, 
                          st.session_state.random_params.get('soc', 50), 
                          help="Current battery charge level")
            soh = st.slider("State of Health (%)", 60, 100, 
                          st.session_state.random_params.get('soh', 85), 
                          help="Battery capacity vs new")
            battery_temp = st.number_input("Battery Temperature (°C)", 10, 60, 
                                         st.session_state.random_params.get('battery_temp', 25), 
                                         help="Current battery temperature")
            battery_voltage = st.number_input("Battery Voltage (V)", 300, 450, 
                                            st.session_state.random_params.get('battery_voltage', 380), 
                                            help="Battery pack voltage")
            battery_current = st.number_input("Battery Current (A)", -200, 200, 
                                            st.session_state.random_params.get('battery_current', 0), 
                                            help="Charging (+) or discharging (-)")
            charging_cycles = st.number_input("Charging Cycles", 0, 5000, 
                                            st.session_state.random_params.get('charging_cycles', 500), 
                                            help="Total charging cycles")
        
        with col2:
            st.markdown("**Motor & Drivetrain**")
            motor_temp = st.number_input("Motor Temperature (°C)", 20, 80, 
                                       st.session_state.random_params.get('motor_temp', 35), 
                                       help="Motor operating temperature")
            motor_rpm = st.number_input("Motor RPM", 0, 8000, 
                                      st.session_state.random_params.get('motor_rpm', 2000), 
                                      help="Motor revolutions per minute")
            motor_torque = st.number_input("Motor Torque (Nm)", 0, 500, 
                                         st.session_state.random_params.get('motor_torque', 150), 
                                         help="Motor torque output")
            power_consumption = st.number_input("Power Consumption (kW)", 0, 100, 
                                              st.session_state.random_params.get('power_consumption', 20), 
                                              help="Current power usage")
            driving_speed = st.number_input("Driving Speed (km/h)", 0, 150, 
                                          st.session_state.random_params.get('driving_speed', 50), 
                                          help="Current vehicle speed")
        
        with col3:
            st.markdown("**Environment & Usage**")
            ambient_temp = st.number_input("Ambient Temperature (°C)", -20, 50, 
                                         st.session_state.random_params.get('ambient_temp', 20), 
                                         help="Outside temperature")
            ambient_humidity = st.number_input("Ambient Humidity (%)", 20, 100, 
                                             st.session_state.random_params.get('ambient_humidity', 50), 
                                             help="Air humidity level")
            brake_pressure = st.number_input("Brake Pressure (bar)", 0, 200, 
                                           st.session_state.random_params.get('brake_pressure', 50), 
                                           help="Brake system pressure")
            tire_pressure = st.number_input("Tire Pressure (bar)", 1.5, 3.0, 
                                          st.session_state.random_params.get('tire_pressure', 2.2), 
                                          help="Tire inflation pressure")
            distance_traveled = st.number_input("Distance Traveled (km)", 0, 500000, 
                                               st.session_state.random_params.get('distance_traveled', 50000), 
                                               help="Total vehicle mileage")
            maintenance_days = st.number_input("Days Since Maintenance", 0, 365, 
                                             st.session_state.random_params.get('maintenance_days', 30), 
                                             help="Days since last service")
        
        # Prediction button
        submit_button = st.form_submit_button("🔮 Predict Health Score", use_container_width=True)
        
        if submit_button:
            # Add a small random component to break caching
            import random
            cache_breaker = random.random() * 0.001
            
            # Create input data frame
            input_data = create_input_dataframe({
                'SOC': soc + cache_breaker,  # Add tiny variation to break caching
                'SOH': soh,
                'Battery_Temp': battery_temp,
                'Battery_Voltage': battery_voltage,
                'Battery_Current': battery_current,
                'Charging_Cycles': charging_cycles,
                'Motor_Temp': motor_temp,
                'Motor_RPM': motor_rpm,
                'Motor_Torque': motor_torque,
                'Power_Consumption': power_consumption,
                'Driving_Speed': driving_speed,
                'Ambient_Temperature': ambient_temp,
                'Ambient_Humidity': ambient_humidity,
                'Brake_Pressure': brake_pressure,
                'Tire_Pressure': tire_pressure,
                'Distance_Traveled': distance_traveled,
                'days_since_maintenance': maintenance_days
            })
            
                        # Make prediction
            try:
                predictor = load_predictor()
                # Make prediction using the model directly
                if predictor is None:
                    st.error("❌ Model not loaded")
                    return
                    
                raw_prediction = predictor.predict(input_data)[0]
                
                # Use the raw prediction directly - don't scale it artificially
                # The model predicts RUL in hours, so use that value
                predicted_rul = max(1, raw_prediction)  # Ensure positive value
                
                # Calculate health score directly from RUL
                # Lower RUL = lower health score
                if predicted_rul >= 110:
                    health_score = 0.9   # Excellent (90%+)
                elif predicted_rul >= 105:
                    health_score = 0.75  # Good (75%)
                elif predicted_rul >= 100:
                    health_score = 0.6   # Fair (60%)
                elif predicted_rul >= 95:
                    health_score = 0.4   # Poor (40%)
                else:
                    health_score = 0.2   # Critical (20% or below)
                
                # Use the predicted RUL for display
                raw_prediction = predicted_rul
                
                # Ensure health score is valid
                health_score = max(0.0, min(1.0, health_score))
                
                # Check for NaN and handle it
                if pd.isna(health_score) or not np.isfinite(health_score):
                    st.error("❌ Model returned invalid prediction. Using default safe value.")
                    health_score = 0.5  # Default to middle value
                
                # Determine health category
                if health_score >= 0.8:
                    category = "Excellent"
                    color = "#2ecc71"
                    recommendation = "Vehicle is in excellent condition. Continue regular maintenance."
                elif health_score >= 0.6:
                    category = "Good"
                    color = "#27ae60"
                    recommendation = "Vehicle is in good condition. Monitor key parameters."
                elif health_score >= 0.4:
                    category = "Fair"
                    color = "#f39c12"
                    recommendation = "Schedule maintenance check soon. Monitor closely."
                elif health_score >= 0.2:
                    category = "Poor"
                    color = "#e67e22"
                    recommendation = "Maintenance required. Schedule service immediately."
                else:
                    category = "Critical"
                    color = "#e74c3c"
                    recommendation = "URGENT: Vehicle needs immediate attention. Do not operate."




                # Display results
                st.success("✅ Prediction Complete!")
                
                # Results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if pd.isna(health_score):
                        st.metric(
                            label="Health Score",
                            value="Error",
                            delta="Unable to calculate"
                        )
                    else:
                        st.metric(
                            label="Health Score",
                            value=f"{health_score:.3f}",
                            delta=f"{(health_score * 100):.1f}%"
                        )
                
                with col2:
                    st.metric(
                        label="Remaining Useful Life",
                        value=f"{raw_prediction:.1f} hrs",
                        delta=f"{raw_prediction/24:.1f} days"
                    )
                
                with col3:
                    st.markdown(f"""
                    <div style="padding: 1rem; border-radius: 10px; border-left: 5px solid {color}; background-color: #f0f2f6;">
                        <h3 style="color: {color}; margin: 0;">{category}</h3>
                        <p style="margin: 0.5rem 0; color: #666666;">Health Category</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    # Create mini gauge
                    gauge_fig = create_health_gauge(health_score, "Vehicle Health")
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                st.info(recommendation)
                
                # RUL Analysis
                st.markdown("### ⏱️ Remaining Useful Life Analysis")
                
                rul_col1, rul_col2, rul_col3 = st.columns(3)
                
                with rul_col1:
                    st.metric("Hours Remaining", f"{raw_prediction:.1f}", help="Predicted operating hours before maintenance")
                    
                with rul_col2:
                    days_remaining = raw_prediction / 8  # Assuming 8 hours of operation per day
                    st.metric("Operating Days", f"{days_remaining:.1f}", help="Days of normal operation remaining")
                    
                with rul_col3:
                    weeks_remaining = days_remaining / 7
                    if weeks_remaining < 1:
                        st.metric("Time Frame", "< 1 week", delta="Immediate attention", delta_color="inverse")
                    elif weeks_remaining < 4:
                        st.metric("Time Frame", f"{weeks_remaining:.1f} weeks", delta="Schedule soon", delta_color="normal")
                    else:
                        st.metric("Time Frame", f"{weeks_remaining:.1f} weeks", delta="Plan ahead", delta_color="normal")
                
                # RUL interpretation
                if raw_prediction < 50:
                    rul_status = "🚨 **CRITICAL**: Immediate maintenance required"
                    rul_color = "error"
                elif raw_prediction < 75:
                    rul_status = "⚠️ **WARNING**: Schedule maintenance within days"
                    rul_color = "warning"
                elif raw_prediction < 100:
                    rul_status = "📅 **NOTICE**: Plan maintenance within weeks" 
                    rul_color = "info"
                else:
                    rul_status = "✅ **GOOD**: Vehicle has adequate remaining life"
                    rul_color = "success"
                
                if rul_color == "error":
                    st.error(rul_status)
                elif rul_color == "warning":
                    st.warning(rul_status)
                elif rul_color == "info":
                    st.info(rul_status)
                else:
                    st.success(rul_status)
                
                # Risk factors analysis
                st.markdown("### ⚠️ Risk Factors Analysis")
                risk_factors = []
                
                if battery_temp > 40:
                    risk_factors.append("🔥 High battery temperature detected")
                if soh < 80:
                    risk_factors.append("🔋 Battery degradation detected")
                if motor_temp > 60:
                    risk_factors.append("🌡️ Motor running hot")
                if maintenance_days > 90:
                    risk_factors.append("🔧 Overdue for maintenance")
                if tire_pressure < 2.0:
                    risk_factors.append("🛞 Low tire pressure")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("✅ No immediate risk factors detected")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please ensure the model files are available and all inputs are valid.")

def create_input_dataframe(params):
    """Create a properly formatted dataframe for prediction."""
    # Create input data with the exact feature names expected by the model
    data = {
        'SoC': params.get('SOC', 50),  # Model expects 'SoC' not 'SOC'
        'SoH': params.get('SOH', 85),  # Model expects 'SoH' not 'SOH'
        'Battery_Voltage': params.get('Battery_Voltage', 380),
        'Battery_Current': params.get('Battery_Current', 0),
        'Battery_Temperature': params.get('Battery_Temp', 25),  # Model expects 'Battery_Temperature'
        'Charge_Cycles': params.get('Charging_Cycles', 500),
        'Motor_Temperature': params.get('Motor_Temp', 35),  # Model expects 'Motor_Temperature'
        'Motor_Vibration': 0.5,  # Default value
        'Motor_Torque': params.get('Motor_Torque', 150),
        'Motor_RPM': params.get('Motor_RPM', 2000),
        'Power_Consumption': params.get('Power_Consumption', 20),
        'Brake_Pad_Wear': 50,  # Default value
        'Brake_Pressure': params.get('Brake_Pressure', 50),
        'Reg_Brake_Efficiency': 0.85,  # Default value
        'Tire_Pressure': params.get('Tire_Pressure', 2.2),
        'Tire_Temperature': params.get('Ambient_Temperature', 20) + 5,
        'Suspension_Load': 1000,  # Default value
        'Ambient_Temperature': params.get('Ambient_Temperature', 20),
        'Ambient_Humidity': params.get('Ambient_Humidity', 50),
        'Load_Weight': 1500,  # Default value
        'Driving_Speed': params.get('Driving_Speed', 50),
        'Distance_Traveled': params.get('Distance_Traveled', 50000),
        'Idle_Time': 10,  # Default value
        'Route_Roughness': 0.5,  # Default value
        'TTF': 1000,  # Time to Failure - default value since we can't predict this
        'Component_Health_Score': 0.8,  # Default value
    }
    
    return pd.DataFrame([data])
        
    # Footer
    st.markdown("---")
    st.markdown("*EV Health Monitoring Dashboard - Built with Streamlit & Plotly*")

if __name__ == "__main__":
    main()
