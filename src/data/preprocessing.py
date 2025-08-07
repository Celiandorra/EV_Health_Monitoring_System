"""
Data preprocessing utilities for EV health monitoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EVDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for EV health monitoring.
    
    This class handles:
    - Data cleaning and quality checks
    - Missing value treatment
    - Outlier detection and treatment
    - Feature harmonization
    - Temporal alignment
    - Data transformation
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {}
        self.preprocessing_history = []
        self.feature_mappings = {}
        
    def clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Clean a dataset with comprehensive preprocessing.
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset for logging
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {dataset_name}")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Record initial state
        initial_shape = df_clean.shape
        self.preprocessing_history.append(f"{dataset_name}_initial: {initial_shape}")
        
        # Step 1: Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 2: Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Step 3: Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Step 4: Data type optimization
        df_clean = self._optimize_data_types(df_clean)
        
        # Record final state
        final_shape = df_clean.shape
        self.preprocessing_history.append(f"{dataset_name}_final: {final_shape}")
        
        logger.info(f"Cleaning complete for {dataset_name}: {initial_shape} → {final_shape}")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with specific strategy for vs_user_avg columns."""
        df_filled = df.copy()
        
        # Special handling for vs_user_avg columns - fill with 0
        vs_user_avg_columns = [
            'Ambient_Humidity_vs_user_avg',
            'Ambient_Temperature_vs_user_avg', 
            'Battery_Current_vs_user_avg',
            'Battery_Voltage_vs_user_avg'
        ]
        
        for col in vs_user_avg_columns:
            if col in df_filled.columns:
                missing_count = df_filled[col].isnull().sum()
                if missing_count > 0:
                    df_filled[col] = df_filled[col].fillna(0)
                    logger.info(f"Filled {missing_count} missing values in {col} with 0")
        
        # For other columns, drop rows with missing values
        initial_length = len(df_filled)
        df_filled = df_filled.dropna()
        dropped_rows = initial_length - len(df_filled)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with missing values")
        
        return df_filled
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_length = len(df)
        df_dedup = df.drop_duplicates()
        duplicates_removed = initial_length - len(df_dedup)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df_dedup
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers using IQR method."""
        df_clean = df.copy()
        
        # Only process numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    logger.info(f"Capped {outliers_count} outliers in {col}")
        
        return df_clean
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        df_optimized = df.copy()
        
        # Convert object columns that are actually numeric
        for col in df_optimized.select_dtypes(include=['object']).columns:
            try:
                df_optimized[col] = pd.to_numeric(df_optimized[col], errors='ignore')
            except:
                pass
        
        # Optimize integer columns
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = df_optimized[col].astype('float32')
        
        return df_optimized
    
    def create_unified_dataset(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             join_strategy: str = 'concat') -> pd.DataFrame:
        """
        Create a unified dataset from multiple sources.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame  
            join_strategy: How to combine ('concat', 'merge')
            
        Returns:
            Unified DataFrame
        """
        logger.info(f"Creating unified dataset using {join_strategy} strategy")
        
        if join_strategy == 'concat':
            # Add source identifiers
            df1_labeled = df1.copy()
            df2_labeled = df2.copy()
            
            df1_labeled['data_source'] = 'maintenance'
            df2_labeled['data_source'] = 'patterns'
            
            # Ensure both have same columns
            all_columns = set(df1_labeled.columns) | set(df2_labeled.columns)
            
            for col in all_columns:
                if col not in df1_labeled.columns:
                    df1_labeled[col] = np.nan
                if col not in df2_labeled.columns:
                    df2_labeled[col] = np.nan
            
            # Align column order
            column_order = sorted(all_columns)
            df1_labeled = df1_labeled[column_order]
            df2_labeled = df2_labeled[column_order]
            
            # Concatenate
            unified_df = pd.concat([df1_labeled, df2_labeled], 
                                 ignore_index=True, sort=False)
            
        elif join_strategy == 'merge':
            # Merge on timestamp (requires common timestamp column)
            timestamp_cols = [col for col in df1.columns if 'timestamp' in col.lower()]
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                unified_df = pd.merge(df1, df2, on=timestamp_col, how='outer', suffixes=('_maint', '_patt'))
            else:
                raise ValueError("No timestamp column found for merging")
        
        else:
            raise ValueError(f"Unsupported join strategy: {join_strategy}")
        
        logger.info(f"Unified dataset created: {unified_df.shape}")
        
        return unified_df
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary of all preprocessing steps performed."""
        return {
            'total_steps': len(self.preprocessing_history),
            'steps_performed': self.preprocessing_history,
            'scalers_fitted': list(self.scalers.keys())
        }


def create_preprocessing_report(df: pd.DataFrame, name: str) -> Dict:
    """
    Create a comprehensive preprocessing report.
    
    Args:
        df: DataFrame to analyze
        name: Dataset name
        
    Returns:
        Dictionary with preprocessing report
    """
    report = {
        'dataset_name': name,
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    # Add data quality score
    quality_score = 100
    if report['missing_values'] > 0:
        quality_score -= min(20, (report['missing_values'] / df.size) * 100)
    if report['duplicate_rows'] > 0:
        quality_score -= min(10, (report['duplicate_rows'] / len(df)) * 100)
    
    report['data_quality_score'] = round(quality_score, 2)
    
    return report
