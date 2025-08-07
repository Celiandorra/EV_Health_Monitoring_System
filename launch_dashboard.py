#!/usr/bin/env python3
"""
EV Health Dashboard Launcher

Quick launcher script for the EV Health Monitoring Dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['streamlit', 'plotly', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ Packages installed successfully!")
    
    return True

def check_data():
    """Check if dashboard data is available."""
    data_files = ['dashboard_data.csv', 'final_predictions.csv']
    
    for file in data_files:
        if Path(file).exists():
            print(f"✅ Found data file: {file}")
            return True
    
    print("⚠️  No dashboard data found.")
    print("Run predictions first: python main.py predict --input-file data/features/features_for_modeling.csv")
    
    # Check if we can create sample data
    if Path("data/features/features_for_modeling.csv").exists():
        print("📊 Creating sample dashboard data...")
        create_sample_data()
        return True
    
    return False

def create_sample_data():
    """Create sample data for dashboard."""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Load features data
        df = pd.read_csv("data/features/features_for_modeling.csv")
        
        # Create mock health scores
        np.random.seed(42)
        health_scores = np.random.beta(2, 5, len(df))  # Skewed towards lower scores
        
        # Add health data
        df['predicted_health_score'] = health_scores
        df['health_category'] = pd.cut(health_scores, 
                                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=['critical', 'poor', 'fair', 'good', 'excellent'])
        
        # Add timestamps
        base_time = datetime.now() - timedelta(days=30)
        df['Timestamp'] = [base_time + timedelta(hours=i*0.01) for i in range(len(df))]
        
        # Sample for performance
        dashboard_df = df.sample(n=min(5000, len(df)), random_state=42).reset_index(drop=True)
        dashboard_df.to_csv('dashboard_data.csv', index=False)
        
        print(f"✅ Created dashboard data with {len(dashboard_df)} records")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching EV Health Monitoring Dashboard...")
    print("📊 Dashboard will open at: http://localhost:8502")
    print("🔄 Press Ctrl+C to stop the dashboard")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'dashboard.py',
            '--server.headless', 'false',
            '--server.port', '8502',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function."""
    print("🚗 EV Health Monitoring Dashboard Launcher")
    print("=" * 50)
    
    # Check if dashboard.py exists
    if not Path("dashboard.py").exists():
        print("❌ dashboard.py not found in current directory")
        return
    
    # Check requirements
    print("🔧 Checking requirements...")
    if not check_requirements():
        return
    
    # Check data
    print("📊 Checking data availability...")
    if not check_data():
        print("❌ No data available for dashboard")
        print("Please run predictions first or ensure data files exist")
        return
    
    print("✅ All checks passed!")
    print()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
