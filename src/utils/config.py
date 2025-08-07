"""
Configuration loader for the EV Health Monitoring project.
"""

import yaml
import os
from pathlib import Path


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to config.yaml in the config directory
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get(self, key, default=None):
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_paths(self):
        """Get data paths configuration."""
        return self.config.get('data', {})
    
    def get_model_config(self):
        """Get model configuration."""
        return self.config.get('models', {})
    
    def get_features_config(self):
        """Get feature engineering configuration."""
        return self.config.get('features', {})
    
    def get_user_profiles_config(self):
        """Get user profiling configuration."""
        return self.config.get('user_profiles', {})
    
    def get_visualization_config(self):
        """Get visualization configuration."""
        return self.config.get('visualization', {})
    
    def get_evaluation_config(self):
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})


# Global configuration instance
config = Config()
