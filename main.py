"""
Main entry point for the EV Health Monitoring project.
"""

import click
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import load_all_datasets
from src.data.preprocessing import EVDataPreprocessor
from src.features.feature_engineering import EVFeatureEngineering
from src.models.prediction import EVHealthPredictor
from src.utils.config import config


@click.group()
def cli():
    """Advanced EV Health Monitoring and Predictive Maintenance CLI."""
    pass


@cli.command()
def setup():
    """Setup the project environment."""
    click.echo("🚗 Setting up EV Health Monitoring project...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "setup_project.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo(result.stdout)
        else:
            click.echo(f"Setup failed: {result.stderr}")
            
    except Exception as e:
        click.echo(f"Error during setup: {e}")


@cli.command()
def load_data():
    """Load and validate datasets."""
    click.echo("📁 Loading datasets...")
    
    try:
        maintenance_df, patterns_df = load_all_datasets()
        
        click.echo(f"✅ Maintenance dataset: {maintenance_df.shape}")
        click.echo(f"✅ Patterns dataset: {patterns_df.shape}")
        click.echo(f"✅ Total records: {maintenance_df.shape[0] + patterns_df.shape[0]:,}")
        
        # Basic validation
        if maintenance_df.empty or patterns_df.empty:
            click.echo("⚠️  Warning: One or more datasets are empty")
        else:
            click.echo("🎉 Data loaded successfully!")
            
    except Exception as e:
        click.echo(f"❌ Error loading data: {e}")


@cli.command()
@click.option('--input-file', '-i', required=True, help='Input CSV file with sensor data')
@click.option('--output-file', '-o', help='Output CSV file for predictions (optional)')
@click.option('--model-path', '-m', help='Path to trained model file')
def predict(input_file, output_file, model_path):
    """Make health score predictions on input data."""
    click.echo(f"� Making predictions on {input_file}")
    
    try:
        # Set default output file if not provided
        if not output_file:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_predictions.csv"
        
        # Load data
        df = pd.read_csv(input_file)
        click.echo(f"📊 Loaded {len(df)} records")
        
        # Create predictor
        predictor = EVHealthPredictor(model_path=model_path)
        
        # Run prediction pipeline
        results = predictor.predict_full_pipeline(df)
        
        # Add predictions to original data
        df_output = df.copy()
        df_output['predicted_health_score'] = results['health_scores']
        
        # Add health categories
        health_categories = []
        for score in results['health_scores']:
            if score >= 0.8:
                health_categories.append('excellent')
            elif score >= 0.6:
                health_categories.append('good')
            elif score >= 0.4:
                health_categories.append('fair')
            elif score >= 0.2:
                health_categories.append('poor')
            else:
                health_categories.append('critical')
        
        df_output['health_category'] = health_categories
        
        # Save results
        df_output.to_csv(output_file, index=False)
        
        # Display summary
        click.echo(f"✅ Predictions completed!")
        click.echo(f"📁 Results saved to: {output_file}")
        click.echo(f"📊 Prediction Summary:")
        click.echo(f"   Mean Health Score: {results['prediction_summary']['mean_health_score']:.3f}")
        click.echo(f"   Min Health Score:  {results['prediction_summary']['min_health_score']:.3f}")
        click.echo(f"   Max Health Score:  {results['prediction_summary']['max_health_score']:.3f}")
        
        # Health category breakdown
        click.echo(f"🏷️  Health Categories:")
        for category, count in results['health_categories']['counts'].items():
            percentage = results['health_categories']['percentages'][category]
            click.echo(f"   {category.capitalize()}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        click.echo(f"❌ Error during prediction: {e}")


@cli.command()
@click.option('--model-path', '-m', help='Path to trained model file')
@click.option('--top-n', '-n', default=20, help='Number of top features to show')
def feature_importance(model_path, top_n):
    """Show feature importance from the trained model."""
    click.echo("📊 Analyzing feature importance...")
    
    try:
        predictor = EVHealthPredictor(model_path=model_path)
        if not predictor.load_model():
            click.echo("❌ Failed to load model")
            return
        
        importance_df = predictor.get_feature_importance(top_n=top_n)
        
        if importance_df is not None:
            click.echo(f"🔝 Top {top_n} Most Important Features:")
            click.echo("=" * 60)
            for idx, row in importance_df.iterrows():
                click.echo(f"{row['feature']:<40} {row['importance']:.4f}")
        else:
            click.echo("⚠️  Feature importance not available for this model")
            
    except Exception as e:
        click.echo(f"❌ Error analyzing feature importance: {e}")


@cli.command()
def preprocess():
    """Run data preprocessing pipeline."""
    click.echo("🔧 Running data preprocessing pipeline...")
    
    try:
        # Load raw data
        click.echo("📁 Loading raw datasets...")
        maintenance_df, patterns_df = load_all_datasets()
        
        # Initialize preprocessor
        preprocessor = EVDataPreprocessor()
        
        # Clean datasets
        click.echo("🧹 Cleaning datasets...")
        maintenance_clean = preprocessor.clean_dataset(maintenance_df, "maintenance")
        patterns_clean = preprocessor.clean_dataset(patterns_df, "patterns")
        
        # Save cleaned data
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        maintenance_clean.to_csv(output_dir / "maintenance_cleaned.csv", index=False)
        patterns_clean.to_csv(output_dir / "patterns_cleaned.csv", index=False)
        
        click.echo(f"✅ Cleaned data saved to {output_dir}/")
        click.echo(f"📊 Maintenance: {maintenance_df.shape} → {maintenance_clean.shape}")
        click.echo(f"📊 Patterns: {patterns_df.shape} → {patterns_clean.shape}")
        
    except Exception as e:
        click.echo(f"❌ Error during preprocessing: {e}")


@cli.command()
def engineer_features():
    """Run feature engineering pipeline."""
    click.echo("🔬 Running feature engineering pipeline...")
    
    try:
        # Load processed data
        processed_dir = Path("data/processed")
        if not (processed_dir / "maintenance_cleaned.csv").exists():
            click.echo("❌ Processed data not found. Run 'preprocess' command first.")
            return
        
        click.echo("📁 Loading processed datasets...")
        maintenance_df = pd.read_csv(processed_dir / "maintenance_cleaned.csv")
        patterns_df = pd.read_csv(processed_dir / "patterns_cleaned.csv")
        
        # Initialize feature engineer
        engineer = EVFeatureEngineering()
        
        # Engineer features
        click.echo("🔬 Engineering features...")
        
        # Determine timestamp columns
        timestamp_col_maint = 'Timestamp' if 'Timestamp' in maintenance_df.columns else None
        timestamp_col_patt = 'timestamp' if 'timestamp' in patterns_df.columns else None
        
        maintenance_features = engineer.engineer_all_features(
            maintenance_df, 
            timestamp_col=timestamp_col_maint
        )
        patterns_features = engineer.engineer_all_features(
            patterns_df, 
            timestamp_col=timestamp_col_patt,
            user_col='user_type'
        )
        
        # Save engineered features
        output_dir = Path("data/features")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        maintenance_features.to_csv(output_dir / "maintenance_features.csv", index=False)
        patterns_features.to_csv(output_dir / "patterns_features.csv", index=False)
        
        # Create combined dataset for modeling
        # Select numeric columns only
        maint_numeric = maintenance_features.select_dtypes(include=[np.number])
        patt_numeric = patterns_features.select_dtypes(include=[np.number])
        
        # Combine datasets
        combined_features = pd.concat([maint_numeric, patt_numeric], ignore_index=True)
        combined_features.to_csv(output_dir / "features_for_modeling.csv", index=False)
        
        click.echo(f"✅ Engineered features saved to {output_dir}/")
        click.echo(f"📊 Maintenance features: {maintenance_features.shape}")
        click.echo(f"📊 Patterns features: {patterns_features.shape}")
        click.echo(f"📊 Combined features: {combined_features.shape}")
        
    except Exception as e:
        click.echo(f"❌ Error during feature engineering: {e}")


@cli.command()
def status():
    """Show project status."""
    click.echo("📋 Project Status")
    click.echo("=" * 50)
    
    # Check if key files exist
    key_files = [
        "config/config.yaml",
        "requirements.txt", 
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_data_preprocessing.ipynb",
        "notebooks/03_feature_engineering.ipynb",
        "notebooks/04_prediction_modelling.ipynb",
        "notebooks/05_lstm_modeling.ipynb",
        "src/data/data_loader.py",
        "src/features/feature_engineering.py",
        "src/models/prediction.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            click.echo(f"✅ {file_path}")
        else:
            click.echo(f"❌ {file_path}")
    
    # Check data files
    click.echo("\n📊 Data Files:")
    data_files = [
        "archive/EV_Predictive_Maintenance_Dataset_15min.csv",
        "archive (1)/rare_user.csv",
        "archive (1)/moderate_user.csv",
        "archive (1)/heavy_user.csv",
        "archive (1)/daily_user.csv"
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / 1024 / 1024
            click.echo(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            click.echo(f"❌ {file_path}")
    
    # Check processed/engineered data
    click.echo("\n🔧 Processed Data:")
    processed_files = [
        "data/processed/maintenance_cleaned.csv",
        "data/processed/patterns_cleaned.csv",
        "data/features/features_for_modeling.csv"
    ]
    
    for file_path in processed_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / 1024 / 1024
            click.echo(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            click.echo(f"❌ {file_path}")
    
    # Check models
    click.echo("\n🤖 Models:")
    model_files = [
        "models/rul_rf_model.joblib",
        "models/lstm_rul_model.h5"
    ]
    
    for file_path in model_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / 1024 / 1024
            click.echo(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            click.echo(f"❌ {file_path}")


@cli.command()
@click.argument('notebook_name')
def notebook(notebook_name):
    """Launch a specific notebook."""
    notebook_map = {
        'explore': 'notebooks/01_data_exploration.ipynb',
        'preprocess': 'notebooks/02_data_preprocessing.ipynb',
        'features': 'notebooks/03_feature_engineering.ipynb',
        'model': 'notebooks/04_prediction_modelling.ipynb',
        'lstm': 'notebooks/05_lstm_modeling.ipynb'
    }
    
    notebook_path = notebook_map.get(notebook_name)
    if not notebook_path:
        click.echo(f"❌ Unknown notebook: {notebook_name}")
        click.echo("Available notebooks: " + ", ".join(notebook_map.keys()))
        return
    
    if not Path(notebook_path).exists():
        click.echo(f"❌ Notebook not found: {notebook_path}")
        return
    
    click.echo(f"📓 Launching {notebook_path}")
    try:
        import subprocess
        subprocess.run(["jupyter", "notebook", notebook_path])
    except Exception as e:
        click.echo(f"Error: {e}")
        click.echo(f"Please run manually: jupyter notebook {notebook_path}")


if __name__ == "__main__":
    cli()
