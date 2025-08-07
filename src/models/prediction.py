"""
Prediction utilities for EV health monitoring using trained models.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EVHealthPredictor:
    """
    EV Health Monitoring Prediction Pipeline.
    
    This class handles the complete prediction pipeline including:
    - Data preprocessing
    - Feature engineering
    - Model loading
    - Health score prediction
    """
    
    def __init__(self, model_path: str = None, base_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file
            base_path: Base path for the project
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)
        
        if model_path is None:
            self.model_path = self.base_path / "models" / "rul_rf_model.joblib"
        else:
            self.model_path = Path(model_path)
        
        self.model = None
        self.feature_names = None
        self.is_model_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the trained Random Forest model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Try to get feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Found {len(self.feature_names)} expected features")
            
            self.is_model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            df: Input DataFrame with raw sensor data
            
        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        df_processed = df.copy()
        
        # Handle missing values for the problematic vs_user_avg columns
        impute_cols = [
            'Ambient_Humidity_vs_user_avg',
            'Ambient_Temperature_vs_user_avg',
            'Battery_Current_vs_user_avg',
            'Battery_Voltage_vs_user_avg'
        ]
        
        for col in impute_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0)
        
        # Drop any remaining rows with NaN values
        initial_length = len(df_processed)
        df_processed = df_processed.dropna()
        
        if len(df_processed) < initial_length:
            logger.warning(f"Dropped {initial_length - len(df_processed)} rows with missing values")
        
        return df_processed
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to the preprocessed data.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        from ..features.feature_engineering import EVFeatureEngineering
        
        engineer = EVFeatureEngineering()
        
        # Determine timestamp column
        timestamp_col = None
        timestamp_candidates = ['Timestamp', 'timestamp', 'time']
        for col in timestamp_candidates:
            if col in df.columns:
                timestamp_col = col
                break
        
        # Apply all feature engineering
        df_engineered = engineer.engineer_all_features(
            df, 
            timestamp_col=timestamp_col,
            user_col='user_type' if 'user_type' in df.columns else None
        )
        
        return df_engineered
    
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare engineered features for model prediction.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Remove non-numeric and identifier columns
        exclude_cols = [
            'Timestamp', 'timestamp', 'time',
            'user_type', 'data_source',
            'Maintenance_Type', 'DTC'
        ]
        
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Handle infinite values
        df_features = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with all NaN values
        df_features = df_features.dropna(axis=1, how='all')
        
        # If we have expected feature names from the model, align the features
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df_features.columns)
            extra_features = set(df_features.columns) - set(self.feature_names)
            
            if missing_features:
                logger.warning(f"Missing expected features: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    df_features[feature] = 0
            
            if extra_features:
                logger.warning(f"Extra features will be ignored: {extra_features}")
            
            # Reorder columns to match training data
            df_features = df_features[self.feature_names]
        
        return df_features
    
    def predict_health_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict health scores for the input data.
        
        Args:
            df: DataFrame with prepared features
            
        Returns:
            Array of predicted health scores (0-1 scale)
        """
        if not self.is_model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Get RUL predictions from the model
            rul_predictions = self.model.predict(df)
            
            # Convert RUL to health score (0-1 scale)
            # Use data-driven bounds for better distribution across health categories
            
            # Get actual prediction range and expand slightly for better distribution
            actual_min = rul_predictions.min()
            actual_max = rul_predictions.max()
            
            # Expand the range to ensure we get good distribution across all categories
            range_expansion = (actual_max - actual_min) * 0.2
            min_rul = actual_min - range_expansion
            max_rul = actual_max + range_expansion
            
            # Linear transformation: health = (RUL - min_rul) / (max_rul - min_rul)
            health_scores = (rul_predictions - min_rul) / (max_rul - min_rul)
            
            # Ensure health scores are in valid range [0, 1]
            health_scores = np.clip(health_scores, 0, 1)
            
            logger.info(f"RUL predictions range: {rul_predictions.min():.2f} - {rul_predictions.max():.2f} hours")
            logger.info(f"Health scores range: {health_scores.min():.3f} - {health_scores.max():.3f}")
            
            return health_scores
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_full_pipeline(self, df: pd.DataFrame) -> Dict:
        """
        Run the complete prediction pipeline on input data.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_model_loaded:
            if not self.load_model():
                raise ValueError("Failed to load model")
        
        try:
            logger.info("Starting full prediction pipeline")
            
            # Step 1: Preprocess data
            df_preprocessed = self.preprocess_data(df)
            logger.info(f"Preprocessing complete: {df_preprocessed.shape}")
            
            # Step 2: Engineer features
            df_engineered = self.engineer_features(df_preprocessed)
            logger.info(f"Feature engineering complete: {df_engineered.shape}")
            
            # Step 3: Prepare features for prediction
            df_features = self.prepare_features_for_prediction(df_engineered)
            logger.info(f"Feature preparation complete: {df_features.shape}")
            
            # Step 4: Make predictions
            predictions = self.predict_health_score(df_features)
            logger.info(f"Predictions complete: {len(predictions)} scores generated")
            
            # Create results
            results = {
                'health_scores': predictions,
                'input_shape': df.shape,
                'processed_shape': df_preprocessed.shape,
                'features_shape': df_features.shape,
                'model_type': type(self.model).__name__,
                'feature_count': len(df_features.columns),
                'prediction_summary': {
                    'mean_health_score': float(np.mean(predictions)),
                    'min_health_score': float(np.min(predictions)),
                    'max_health_score': float(np.max(predictions)),
                    'std_health_score': float(np.std(predictions))
                }
            }
            
            # Add health categories
            results['health_categories'] = self._categorize_health_scores(predictions)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise
    
    def _categorize_health_scores(self, scores: np.ndarray) -> Dict:
        """
        Categorize health scores into different health levels.
        
        Args:
            scores: Array of health scores
            
        Returns:
            Dictionary with health categories
        """
        categories = {
            'excellent': np.sum(scores >= 0.8),
            'good': np.sum((scores >= 0.6) & (scores < 0.8)),
            'fair': np.sum((scores >= 0.4) & (scores < 0.6)),
            'poor': np.sum((scores >= 0.2) & (scores < 0.4)),
            'critical': np.sum(scores < 0.2)
        }
        
        total = len(scores)
        percentages = {k: (v / total) * 100 for k, v in categories.items()}
        
        return {
            'counts': categories,
            'percentages': percentages,
            'total_records': total
        }
    
    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the loaded model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance or None if not available
        """
        if not self.is_model_loaded:
            logger.warning("Model not loaded")
            return None
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature importance")
            return None
        
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
                'importance': self.model.feature_importances_
            })
            
            importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None


def predict_single_record(data: Dict, model_path: str = None) -> Dict:
    """
    Convenience function to predict health score for a single record.
    
    Args:
        data: Dictionary with sensor data
        model_path: Path to the trained model
        
    Returns:
        Dictionary with prediction result
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([data])
    
    # Create predictor and run pipeline
    predictor = EVHealthPredictor(model_path=model_path)
    results = predictor.predict_full_pipeline(df)
    
    # Return single prediction
    return {
        'health_score': float(results['health_scores'][0]),
        'health_category': _get_health_category(results['health_scores'][0]),
        'model_info': {
            'model_type': results['model_type'],
            'feature_count': results['feature_count']
        }
    }


def _get_health_category(score: float) -> str:
    """Get health category for a single score."""
    if score >= 0.8:
        return 'excellent'
    elif score >= 0.6:
        return 'good'
    elif score >= 0.4:
        return 'fair'
    elif score >= 0.2:
        return 'poor'
    else:
        return 'critical'


def batch_predict(input_file: str, output_file: str, model_path: str = None) -> Dict:
    """
    Predict health scores for a batch of records from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        model_path: Path to the trained model
        
    Returns:
        Dictionary with batch prediction summary
    """
    try:
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} records from {input_file}")
        
        # Create predictor and run pipeline
        predictor = EVHealthPredictor(model_path=model_path)
        results = predictor.predict_full_pipeline(df)
        
        # Add predictions to original data
        df_output = df.copy()
        df_output['predicted_health_score'] = results['health_scores']
        df_output['health_category'] = [_get_health_category(score) for score in results['health_scores']]
        
        # Save results
        df_output.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
        return {
            'input_records': len(df),
            'output_file': output_file,
            'prediction_summary': results['prediction_summary'],
            'health_categories': results['health_categories']
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise
