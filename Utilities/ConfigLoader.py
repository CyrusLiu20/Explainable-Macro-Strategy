import yaml
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd

# Define the BacktestConfig dataclass
@dataclass
class BacktestConfig:
    # Macro News Aggregation Parameters
    max_retries: int
    chunk_size: int
    verbose: bool
    filter_agent: bool
    prompt_num_relevance: str

    # Backtesting Configuration Parameters
    asset: str
    model: str
    has_system_prompt: bool
    ticker: str

    # File Paths and Data Management
    data_root: Path
    output_path: Path
    dates: list
    last_periods_list: list

    # These fields depend on `data_root`, so they should be set in __post_init__
    news_path: Path = field(init=False)
    mapping_csv: Path = field(init=False)
    macro_csv_list: list = field(init=False)

    def __post_init__(self):
        self.news_path = self.data_root / "ProcessedData/MacroNews.csv"
        self.mapping_csv = self.data_root / "MacroIndicators/indicator_mapping.csv"
        self.macro_csv_list = [
            self.data_root / "ProcessedData/MacroIndicatorDaily.csv",
            self.data_root / "ProcessedData/MacroIndicatorWeekly.csv",
            self.data_root / "ProcessedData/MacroIndicatorMonthly.csv",
            self.data_root / "ProcessedData/MacroIndicatorQuarterly.csv"
        ]

def load_config(config_file_path: str) -> BacktestConfig:
    # Open the YAML file and load its contents
    with open(config_file_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Map the loaded data to the BacktestConfig dataclass
    backtest_config = BacktestConfig(
        max_retries=config_data['macro_news_aggregation']['max_retries'],
        chunk_size=config_data['macro_news_aggregation']['chunk_size'],
        verbose=config_data['macro_news_aggregation']['verbose'],
        filter_agent=config_data['macro_news_aggregation']['filter_agent'],
        prompt_num_relevance=config_data['macro_news_aggregation']['prompt_num_relevance'],
        asset=config_data['backtest']['asset'],
        model=config_data['backtest']['model'],
        has_system_prompt=config_data['backtest']['has_system_prompt'],
        ticker=config_data['backtest']['ticker'],
        data_root=Path(config_data['file_paths']['data_root']),
        output_path=Path(config_data['file_paths']['output_path']),
        dates=config_data['dates'],
        last_periods_list=config_data['last_periods']
    )

    return backtest_config