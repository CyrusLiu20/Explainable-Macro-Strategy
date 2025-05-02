import yaml
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import inspect


def filter_valid_kwargs(cls, kwargs):
    """
    Filter out unexpected keyword arguments for a class's __init__ method.

    Args:
    - cls: The class to inspect.
    - kwargs (dict): Dictionary of keyword arguments.

    Returns:
    - dict: A dictionary containing only valid keyword arguments for the class's __init__ method.
    """
    # Get the parameters of the class's __init__ method
    init_params = inspect.signature(cls.__init__).parameters
    
    # Filter kwargs to include only valid parameters
    valid_kwargs = {
        key: value for key, value in kwargs.items()
        if key in init_params
    }
    
    return valid_kwargs


class BaseConfigLoader:
    def __init__(self, config_path):
        """
        Base class for loading and processing YAML configuration files.

        Args:
        - config_path (str or Path): Path to the configuration YAML file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._extract_sections()

    def _load_config(self):
        """Load YAML configuration file."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _extract_sections(self):
        """Extract top-level configurations dynamically."""
        for section, value in self.config.items():
            setattr(self, section, value)

    def _extract_attributes(self, config_section):
        """Dynamically extract attributes from a configuration section."""
        for key, value in config_section.items():
            setattr(self, key, value)

    def get_config(self):
        """Return all configuration parameters as a dictionary."""
        return {key: value for key, value in vars(self).items()}
    

class BacktestConfigurationLoader(BaseConfigLoader):

    def __init__(self, config_path):
        """Derived class for openfoam-specific configurations."""
        super().__init__(config_path)

        # Extract second level attributes
        self._extract_attributes(self.macro_news_aggregation)
        self._extract_attributes(self.backtest)
        self._extract_attributes(self.file_paths)

        self.data_root = Path(self.data_root)
        self.results_path = Path(self.results_path)
        self.news_path = self.data_root / "ProcessedData/MacroNews.csv"
        self.mapping_csv = self.data_root / "MacroIndicators/indicator_mapping.csv"
        self.macro_csv_list = [
            self.data_root / "ProcessedData/MacroIndicatorDaily.csv",
            self.data_root / "ProcessedData/MacroIndicatorWeekly.csv",
            self.data_root / "ProcessedData/MacroIndicatorMonthly.csv",
            self.data_root / "ProcessedData/MacroIndicatorQuarterly.csv"
        ]
