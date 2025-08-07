"""
Feature engineering utilities for EV health monitoring.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class EVFeatureEngineering:
    """Feature engineering utilities for EV health monitoring data."""
    
    def __init__(self):
        self.scalers = {}
    
    def create_temporal_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Create time-based features from timestamp column.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with additional temporal features
        """
        df_copy = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_copy[timestamp_col]):
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
        
        # Extract temporal features
        df_copy['hour'] = df_copy[timestamp_col].dt.hour
        df_copy['day_of_week'] = df_copy[timestamp_col].dt.dayofweek
        df_copy['day_of_month'] = df_copy[timestamp_col].dt.day
        df_copy['month'] = df_copy[timestamp_col].dt.month
        df_copy['quarter'] = df_copy[timestamp_col].dt.quarter
        df_copy['year'] = df_copy[timestamp_col].dt.year
        
        # Cyclical encoding for temporal features
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        # Time since start
        df_copy['days_since_start'] = (df_copy[timestamp_col] - df_copy[timestamp_col].min()).dt.days
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                              windows: List[int] = [6, 12, 24, 48]) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create rolling features for
            windows: Rolling window sizes (in records/time units)
            
        Returns:
            DataFrame with rolling features
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                for window in windows:
                    # Rolling mean
                    df_copy[f'{col}_rolling_mean_{window}'] = df_copy[col].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling standard deviation
                    df_copy[f'{col}_rolling_std_{window}'] = df_copy[col].rolling(window=window, min_periods=1).std()
                    
                    # Rolling min/max
                    df_copy[f'{col}_rolling_min_{window}'] = df_copy[col].rolling(window=window, min_periods=1).min()
                    df_copy[f'{col}_rolling_max_{window}'] = df_copy[col].rolling(window=window, min_periods=1).max()
                    
                    # Rolling range
                    df_copy[f'{col}_rolling_range_{window}'] = (
                        df_copy[f'{col}_rolling_max_{window}'] - df_copy[f'{col}_rolling_min_{window}']
                    )
        
        return df_copy
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                          lags: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Create lag features for time series prediction.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create lag features for
            lags: Lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns:
                for lag in lags:
                    df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
        
        return df_copy
    
    def create_battery_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create battery health-specific features.
        
        Args:
            df: DataFrame with battery data
            
        Returns:
            DataFrame with battery health features
        """
        df_copy = df.copy()
        
        # Battery degradation rate (if SOH is available)
        if 'SOH' in df_copy.columns:
            df_copy['soh_degradation_rate'] = df_copy['SOH'].diff()
            df_copy['soh_degradation_rate_pct'] = (df_copy['SOH'].pct_change() * 100)
        
        # Charging efficiency (if SOC and charging voltage/current available)
        if all(col in df_copy.columns for col in ['SOC', 'Charging_Voltage']):
            df_copy['charging_efficiency'] = df_copy['SOC'] / (df_copy['Charging_Voltage'] + 1e-6)
        
        # Temperature stress indicators
        if 'Battery_Temperature' in df_copy.columns or 'Battery_Temp' in df_copy.columns:
            temp_col = 'Battery_Temperature' if 'Battery_Temperature' in df_copy.columns else 'Battery_Temp'
            
            # Temperature deviation from optimal range (20-25°C)
            optimal_temp_min, optimal_temp_max = 20, 25
            df_copy['temp_stress'] = np.where(
                df_copy[temp_col] < optimal_temp_min,
                optimal_temp_min - df_copy[temp_col],
                np.where(
                    df_copy[temp_col] > optimal_temp_max,
                    df_copy[temp_col] - optimal_temp_max,
                    0
                )
            )
        
        # Charging cycles impact
        if 'Charging_Cycles' in df_copy.columns:
            df_copy['charging_cycles_per_day'] = df_copy.groupby(df_copy.index // 24)['Charging_Cycles'].transform('max')
            df_copy['cumulative_charging_stress'] = df_copy['Charging_Cycles'].cumsum() / 1000  # Normalize
        
        return df_copy
    
    def create_driving_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create driving behavior-specific features.
        
        Args:
            df: DataFrame with driving data
            
        Returns:
            DataFrame with driving behavior features
        """
        df_copy = df.copy()
        
        # Motor usage patterns
        if 'Motor_RPM' in df_copy.columns:
            df_copy['motor_utilization'] = df_copy['Motor_RPM'] / df_copy['Motor_RPM'].max()
            df_copy['motor_stress_level'] = np.where(df_copy['Motor_RPM'] > df_copy['Motor_RPM'].quantile(0.8), 1, 0)
        
        # Acceleration patterns (if motor torque is available)
        if 'Motor_Torque' in df_copy.columns:
            df_copy['torque_variation'] = df_copy['Motor_Torque'].rolling(window=5, min_periods=1).std()
            df_copy['aggressive_driving'] = np.where(df_copy['Motor_Torque'] > df_copy['Motor_Torque'].quantile(0.9), 1, 0)
        
        # Brake usage patterns
        if 'Brake_Pad_Wear' in df_copy.columns:
            df_copy['brake_wear_rate'] = df_copy['Brake_Pad_Wear'].diff()
            df_copy['heavy_braking'] = np.where(df_copy['brake_wear_rate'] > df_copy['brake_wear_rate'].quantile(0.8), 1, 0)
        
        # Energy efficiency
        if all(col in df_copy.columns for col in ['SOC', 'Motor_RPM']):
            df_copy['energy_efficiency'] = df_copy['SOC'] / (df_copy['Motor_RPM'] + 1)
        
        return df_copy
    
    def create_maintenance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create maintenance-related features including the overall_health_score.
        
        Args:
            df: DataFrame with maintenance data
            
        Returns:
            DataFrame with maintenance features
        """
        df_copy = df.copy()
        
        # Time since last maintenance (if maintenance type is available)
        if 'Maintenance_Type' in df_copy.columns:
            df_copy['days_since_maintenance'] = 0
            maintenance_indices = df_copy[df_copy['Maintenance_Type'] > 0].index
            
            for i in range(len(df_copy)):
                last_maintenance = maintenance_indices[maintenance_indices < i]
                if len(last_maintenance) > 0:
                    df_copy.iloc[i, df_copy.columns.get_loc('days_since_maintenance')] = i - last_maintenance[-1]
                else:
                    df_copy.iloc[i, df_copy.columns.get_loc('days_since_maintenance')] = i
        
        # Create overall health score based on available health indicators
        health_indicators = []
        
        # SOH (State of Health) - higher is better
        if 'SOH' in df_copy.columns:
            soh_normalized = df_copy['SOH'] / 100.0  # Assuming SOH is in percentage
            health_indicators.append(soh_normalized)
        
        # SOC (State of Charge) - normalized to 0-1, higher is better for health
        if 'SOC' in df_copy.columns:
            soc_normalized = df_copy['SOC'] / 100.0  # Assuming SOC is in percentage
            health_indicators.append(soc_normalized)
        
        # Temperature indicators (closer to optimal range is better)
        temp_cols = ['Battery_Temperature', 'Battery_Temp', 'Motor_Temperature', 'Motor_Temp']
        for temp_col in temp_cols:
            if temp_col in df_copy.columns:
                # Optimal temperature range is 20-30°C, create health score
                temp_health = 1 - np.abs(df_copy[temp_col] - 25) / 50  # Normalize around 25°C
                temp_health = np.clip(temp_health, 0, 1)  # Clip to 0-1 range
                health_indicators.append(temp_health)
        
        # Wear indicators (lower wear = better health)
        wear_cols = ['Brake_Pad_Wear']
        for wear_col in wear_cols:
            if wear_col in df_copy.columns:
                # Assuming wear is from 0-100, invert for health score
                wear_normalized = df_copy[wear_col] / df_copy[wear_col].max()
                wear_health = 1 - wear_normalized
                health_indicators.append(wear_health)
        
        # Pressure indicators (closer to optimal is better)
        if 'Tire_Pressure' in df_copy.columns:
            # Assuming optimal tire pressure around 35 PSI
            pressure_health = 1 - np.abs(df_copy['Tire_Pressure'] - 35) / 35
            pressure_health = np.clip(pressure_health, 0, 1)
            health_indicators.append(pressure_health)
        
        # Calculate overall health score as average of available indicators
        if health_indicators:
            df_copy['overall_health_score'] = pd.DataFrame(health_indicators).T.mean(axis=1)
        else:
            # Fallback: create a synthetic health score based on RUL if available
            if 'RUL' in df_copy.columns:
                max_rul = df_copy['RUL'].max()
                df_copy['overall_health_score'] = df_copy['RUL'] / max_rul
            else:
                # Last resort: create a random health score (for testing purposes)
                np.random.seed(42)
                df_copy['overall_health_score'] = np.random.uniform(0.3, 0.9, len(df_copy))
        
        # Ensure health score is between 0 and 1
        df_copy['overall_health_score'] = np.clip(df_copy['overall_health_score'], 0, 1)
        
        return df_copy
    
    def create_user_profile_features(self, df: pd.DataFrame, user_col: str = 'user_type') -> pd.DataFrame:
        """
        Create user profile-specific features.
        
        Args:
            df: DataFrame with user data
            user_col: Column containing user type information
            
        Returns:
            DataFrame with user profile features
        """
        df_copy = df.copy()
        
        if user_col in df_copy.columns:
            # One-hot encode user types
            user_dummies = pd.get_dummies(df_copy[user_col], prefix='user')
            df_copy = pd.concat([df_copy, user_dummies], axis=1)
            
            # User-specific statistics
            for user_type in df_copy[user_col].unique():
                user_mask = df_copy[user_col] == user_type
                
                # Average usage patterns for this user type
                numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in df_copy.columns:
                        user_avg = df_copy[user_mask][col].mean()
                        df_copy[f'{col}_vs_user_avg'] = df_copy[col] / (user_avg + 1e-6)
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, features: List[str], 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale selected features.
        
        Args:
            df: DataFrame with features to scale
            features: List of features to scale
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        df_copy = df.copy()
        available_features = [f for f in features if f in df_copy.columns]
        
        if not available_features:
            return df_copy
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        df_copy[available_features] = scaler.fit_transform(df_copy[available_features])
        self.scalers[method] = scaler
        
        return df_copy
    
    def engineer_all_features(self, df: pd.DataFrame, timestamp_col: str = None,
                            user_col: str = 'user_type') -> pd.DataFrame:
        """
        Apply all feature engineering techniques.
        
        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name
            user_col: User type column name
            
        Returns:
            DataFrame with all engineered features
        """
        df_engineered = df.copy()
        
        # Temporal features
        if timestamp_col and timestamp_col in df_engineered.columns:
            df_engineered = self.create_temporal_features(df_engineered, timestamp_col)
        
        # Identify numeric columns for rolling and lag features
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        key_sensor_cols = [col for col in numeric_cols if any(
            sensor in col.lower() for sensor in ['soc', 'soh', 'temp', 'rpm', 'torque', 'pressure']
        )]
        
        # Rolling features
        if key_sensor_cols:
            df_engineered = self.create_rolling_features(df_engineered, key_sensor_cols[:5])  # Limit to prevent explosion
        
        # Lag features
        if key_sensor_cols:
            df_engineered = self.create_lag_features(df_engineered, key_sensor_cols[:3])  # Limit for performance
        
        # Domain-specific features
        df_engineered = self.create_battery_health_features(df_engineered)
        df_engineered = self.create_driving_behavior_features(df_engineered)
        df_engineered = self.create_maintenance_features(df_engineered)
        
        # User profile features
        if user_col in df_engineered.columns:
            df_engineered = self.create_user_profile_features(df_engineered, user_col)
        
        return df_engineered


def prepare_features_for_modeling(df: pd.DataFrame, target_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for machine learning modeling.
    
    Args:
        df: DataFrame with engineered features
        target_cols: Target columns to exclude from features
        
    Returns:
        Tuple of (features_df, feature_names)
    """
    df_features = df.copy()
    
    # Remove non-numeric columns and target columns
    exclude_cols = ['Timestamp', 'user_type'] + (target_cols or [])
    feature_cols = [col for col in df_features.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col])]
    
    # Handle infinite values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    # Select feature columns
    features_df = df_features[feature_cols]
    
    return features_df, feature_cols
