"""
Visualization utilities for the EV Health Monitoring project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class EVDataVisualizer:
    """Visualization utilities for EV health monitoring data."""
    
    def __init__(self, style: str = 'seaborn-v0_8', palette: str = 'husl', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer with styling options.
        
        Args:
            style: Matplotlib style
            palette: Seaborn color palette
            figsize: Default figure size
        """
        plt.style.use(style)
        sns.set_palette(palette)
        self.figsize = figsize
    
    def plot_sensor_distributions(self, df: pd.DataFrame, sensors: List[str], 
                                title: str = "Sensor Distributions", ncols: int = 3) -> None:
        """
        Plot distributions of multiple sensors.
        
        Args:
            df: DataFrame containing sensor data
            sensors: List of sensor column names
            title: Plot title
            ncols: Number of columns in subplot grid
        """
        n_sensors = len(sensors)
        nrows = (n_sensors + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(self.figsize[0] * ncols/3, self.figsize[1] * nrows/2))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, sensor in enumerate(sensors):
            if sensor in df.columns:
                row = i // ncols
                col = i % ncols
                
                axes[row, col].hist(df[sensor].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{sensor} Distribution')
                axes[row, col].set_xlabel(sensor)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_sensors, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_user_comparison(self, df: pd.DataFrame, feature: str, 
                           user_col: str = 'user_type', plot_type: str = 'box') -> go.Figure:
        """
        Compare a feature across different user types using Plotly.
        
        Args:
            df: DataFrame containing the data
            feature: Feature to compare
            user_col: Column containing user type information
            plot_type: Type of plot ('box', 'violin', 'strip')
            
        Returns:
            Plotly figure
        """
        if plot_type == 'box':
            fig = px.box(df, x=user_col, y=feature, 
                        title=f'{feature} Distribution by User Type')
        elif plot_type == 'violin':
            fig = px.violin(df, x=user_col, y=feature,
                           title=f'{feature} Distribution by User Type')
        elif plot_type == 'strip':
            fig = px.strip(df, x=user_col, y=feature,
                          title=f'{feature} Distribution by User Type')
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        fig.update_layout(height=500)
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str], 
                              title: str = "Correlation Matrix") -> None:
        """
        Plot correlation matrix for selected features.
        
        Args:
            df: DataFrame containing the data
            features: List of features to include in correlation matrix
            title: Plot title
        """
        # Select only features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            print(f"Not enough features available for correlation matrix. Found: {available_features}")
            return
        
        plt.figure(figsize=self.figsize)
        corr_matrix = df[available_features].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .5})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_time_series(self, df: pd.DataFrame, time_col: str, value_cols: List[str],
                        title: str = "Time Series", sample_size: int = None) -> go.Figure:
        """
        Plot time series for multiple variables.
        
        Args:
            df: DataFrame containing time series data
            time_col: Column name for time/timestamp
            value_cols: List of columns to plot
            title: Plot title
            sample_size: Number of records to sample (None for all)
            
        Returns:
            Plotly figure
        """
        # Sample data if specified
        plot_df = df.sample(n=sample_size) if sample_size else df
        plot_df = plot_df.sort_values(time_col)
        
        fig = make_subplots(rows=len(value_cols), cols=1, 
                           subplot_titles=value_cols,
                           shared_xaxes=True)
        
        for i, col in enumerate(value_cols):
            if col in plot_df.columns:
                fig.add_trace(
                    go.Scatter(x=plot_df[time_col], y=plot_df[col], 
                             mode='lines', name=col, line=dict(width=1)),
                    row=i+1, col=1
                )
        
        fig.update_layout(height=200*len(value_cols), title_text=title)
        fig.update_xaxes(title_text="Time", row=len(value_cols), col=1)
        
        return fig
    
    def plot_scatter_matrix(self, df: pd.DataFrame, features: List[str], 
                          color_col: str = None, sample_size: int = 1000) -> go.Figure:
        """
        Create a scatter plot matrix for exploring relationships.
        
        Args:
            df: DataFrame containing the data
            features: List of features to include
            color_col: Column to use for coloring points
            sample_size: Number of points to sample for visualization
            
        Returns:
            Plotly figure
        """
        # Sample data to avoid overcrowding
        plot_df = df.sample(n=min(sample_size, len(df)))
        
        # Select only available features
        available_features = [f for f in features if f in plot_df.columns]
        
        if len(available_features) < 2:
            print(f"Not enough features available. Found: {available_features}")
            return None
        
        fig = px.scatter_matrix(plot_df, dimensions=available_features, 
                               color=color_col if color_col in plot_df.columns else None,
                               title="Feature Relationships")
        
        fig.update_traces(diagonal_visible=False)
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                              title: str = "Feature Importance", top_n: int = 20) -> None:
        """
        Plot feature importance from a model.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            title: Plot title
            top_n: Number of top features to display
        """
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=self.figsize)
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, (feature, importance) in enumerate(top_features):
            plt.text(importance + max(importances) * 0.01, i, f'{importance:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                title: str = "Predictions vs Actual") -> go.Figure:
        """
        Plot predicted vs actual values for regression models.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Scatter plot of predictions vs actual
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers',
                               name='Predictions', opacity=0.6))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                               mode='lines', name='Perfect Prediction',
                               line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title=title,
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500
        )
        
        return fig
    
    def create_dashboard_components(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Create multiple dashboard components for EV health monitoring.
        
        Args:
            df: DataFrame containing EV data
            
        Returns:
            Dictionary of Plotly figures for dashboard components
        """
        components = {}
        
        # Health overview (if SOH is available)
        if 'SOH' in df.columns:
            components['health_overview'] = px.histogram(
                df, x='SOH', nbins=50,
                title="Battery State of Health Distribution"
            )
        
        # Charging patterns (if SOC is available)
        if 'SOC' in df.columns:
            components['charging_patterns'] = px.box(
                df, y='SOC',
                title="State of Charge Distribution"
            )
        
        # Temperature monitoring
        temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
        if temp_cols:
            temp_data = df[temp_cols].melt(var_name='Sensor', value_name='Temperature')
            components['temperature_monitoring'] = px.box(
                temp_data, x='Sensor', y='Temperature',
                title="Temperature Sensor Readings"
            )
        
        return components


def create_summary_plots(maintenance_df: pd.DataFrame, patterns_df: pd.DataFrame) -> None:
    """
    Create summary visualization plots for both datasets.
    
    Args:
        maintenance_df: Maintenance dataset
        patterns_df: Patterns dataset
    """
    visualizer = EVDataVisualizer()
    
    # Key sensors for maintenance dataset
    maintenance_sensors = ['SoC', 'SoH', 'Battery_Temperature', 'Motor_Temperature', 'RUL', 'Failure_Probability']
    available_maintenance = [s for s in maintenance_sensors if s in maintenance_df.columns]
    
    if available_maintenance:
        visualizer.plot_sensor_distributions(
            maintenance_df, available_maintenance,
            "Key Sensor Distributions - Maintenance Dataset"
        )
    
    # User comparison for patterns
    if 'SOC' in patterns_df.columns and 'user_type' in patterns_df.columns:
        fig = visualizer.plot_user_comparison(patterns_df, 'SOC', plot_type='box')
        fig.show()
    
    # Correlation analysis
    numeric_cols = maintenance_df.select_dtypes(include=[np.number]).columns[:10]  # Limit for readability
    if len(numeric_cols) > 1:
        visualizer.plot_correlation_matrix(maintenance_df, list(numeric_cols),
                                         "Maintenance Dataset - Key Correlations")
