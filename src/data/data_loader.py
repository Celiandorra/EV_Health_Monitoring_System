"""
Data loading utilities for the EV Health Monitoring project.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading and basic preprocessing of EV datasets."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)
    
    def load_maintenance_dataset(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the predictive maintenance dataset.
        
        Args:
            file_path: Path to the maintenance dataset file
            
        Returns:
            DataFrame with the maintenance data
        """
        if file_path is None:
            file_path = self.base_path / "archive" / "EV_Predictive_Maintenance_Dataset_15min.csv"
        
        try:
            logger.info(f"Loading maintenance dataset from {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic preprocessing
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df = df.sort_values('Timestamp')
            
            logger.info(f"Loaded maintenance dataset: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading maintenance dataset: {e}")
            raise
    
    def load_driving_patterns_dataset(self, pattern_dir: str = None) -> pd.DataFrame:
        """
        Load and combine all driving pattern datasets.
        
        Args:
            pattern_dir: Directory containing the pattern files
            
        Returns:
            Combined DataFrame with all user patterns
        """
        if pattern_dir is None:
            pattern_dir = self.base_path / "archive (1)"
        
        pattern_files = {
            'rare_user': 'rare_user.csv',
            'moderate_user': 'moderate_user.csv',
            'heavy_user': 'heavy_user.csv',
            'daily_user': 'daily_user.csv'
        }
        
        combined_dfs = []
        
        for user_type, filename in pattern_files.items():
            file_path = Path(pattern_dir) / filename
            
            try:
                logger.info(f"Loading {user_type} dataset from {file_path}")
                df = pd.read_csv(file_path)
                df['user_type'] = user_type
                
                # Basic preprocessing
                timestamp_col = df.columns[0]  # Assume first column is timestamp
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values(timestamp_col)
                
                combined_dfs.append(df)
                logger.info(f"Loaded {user_type}: {df.shape}")
                
            except Exception as e:
                logger.warning(f"Could not load {filename}: {e}")
        
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            logger.info(f"Combined patterns dataset: {combined_df.shape}")
            return combined_df
        else:
            raise ValueError("No pattern datasets could be loaded")
    
    def get_dataset_summary(self, df: pd.DataFrame, name: str) -> Dict:
        """
        Generate a summary of the dataset.
        
        Args:
            df: DataFrame to summarize
            name: Name of the dataset
            
        Returns:
            Dictionary with dataset summary
        """
        summary = {
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Add time range if datetime columns exist
        if summary['datetime_columns']:
            time_col = summary['datetime_columns'][0]
            summary['time_range'] = {
                'start': df[time_col].min(),
                'end': df[time_col].max(),
                'duration': df[time_col].max() - df[time_col].min()
            }
        
        return summary
    
    def identify_common_features(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify common and unique features between two datasets.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            
        Returns:
            Dictionary with common, unique_to_df1, and unique_to_df2 features
        """
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        return {
            'common': list(cols1.intersection(cols2)),
            'unique_to_df1': list(cols1 - cols2),
            'unique_to_df2': list(cols2 - cols1)
        }
    
    def validate_data_quality(self, df: pd.DataFrame, name: str) -> Dict:
        """
        Validate data quality and identify potential issues.
        
        Args:
            df: DataFrame to validate
            name: Name of the dataset
            
        Returns:
            Dictionary with quality assessment
        """
        quality_report = {
            'dataset_name': name,
            'total_records': len(df),
            'total_features': len(df.columns),
            'issues': []
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            quality_report['missing_values'] = {
                'total_missing': int(missing_counts.sum()),
                'columns_with_missing': missing_counts[missing_counts > 0].to_dict()
            }
            quality_report['issues'].append('Missing values detected')
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_report['duplicates'] = int(duplicate_count)
            quality_report['issues'].append(f'{duplicate_count} duplicate rows found')
        
        # Check for constant columns
        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_report['constant_columns'] = constant_cols
            quality_report['issues'].append(f'Constant columns: {constant_cols}')
        
        # Check for outliers (using IQR method for numeric columns)
        outlier_info = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100
                }
        
        if outlier_info:
            quality_report['outliers'] = outlier_info
            quality_report['issues'].append('Outliers detected in numeric columns')
        
        if not quality_report['issues']:
            quality_report['issues'].append('No major data quality issues detected')
        
        return quality_report


def load_all_datasets(base_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load both datasets.
    
    Args:
        base_path: Base path for the project
        
    Returns:
        Tuple of (maintenance_df, patterns_df)
    """
    loader = DataLoader(base_path)
    
    maintenance_df = loader.load_maintenance_dataset()
    patterns_df = loader.load_driving_patterns_dataset()
    
    return maintenance_df, patterns_df
