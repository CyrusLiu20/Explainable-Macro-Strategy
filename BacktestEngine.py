import argparse
from pickle import STRING
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from Backtest import MacroAggregator, check_file_paths
from Backtest import NewsDrivenFramework
from LLMAgent.InstructionPrompt import *
from DataPipeline import write_mapping

# Configurations and variables for backtesting
@dataclass
class BacktestConfig:

    # Macro News Aggregation Parameters
    max_retries: int = 3 # Number of retries for the FilterAgent
    chunk_size: int = 15 # Discretise news chunk in within the date range
    verbose: bool = True # Print out the relevant news in news aggregation with FilterAgent
    filter_agent: bool = True # Whether to use LLM to filter news (or import from csv)
    prompt_num_relevance: str = "1-2" # Number of relevance news to select in each chunk

    # Backtesting Configuration Parameters
    asset: str = "US 10-year Treasury bonds"
    model: str = "deepseek-r1:8b"
    has_system_prompt: bool = False # Whether the LLM model has a system message
    ticker: str = "IEF"

    data_root: Path = Path("DataPipeline/Data")
    output_path: Path = Path("Backtest/AggregatedData/AggregatedNews.csv")
    # dates: list = field(default_factory=lambda: ["2022-12-13"])
    dates: list = field(
        default_factory=lambda: pd.date_range("2022-04-02", "2022-04-03")
        .strftime("%Y-%m-%d")
        .tolist()
    )
    last_periods_list: list = field(default_factory=lambda: [10, 4, 6, 4]) # Daily, Weekly, Monthly, Quarterly

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

def main(aggregate: bool, backtest: bool):

    backtest_config = BacktestConfig()
    files_exist = check_file_paths([*backtest_config.macro_csv_list, backtest_config.news_path, backtest_config.mapping_csv])
    write_mapping(folder_path=backtest_config.data_root/"MacroIndicators")

    if aggregate and files_exist:

      macro_aggregator = MacroAggregator(config=backtest_config, 
                                         current_date="N/A")
      macro_aggregator.aggregate_news(filter_dates=backtest_config.dates, 
                                     filter_agent=backtest_config.filter_agent, 
                                     max_retries=backtest_config.max_retries,
                                     chunk_size=backtest_config.chunk_size)

    if backtest:

      news_driven_framework = NewsDrivenFramework(config=backtest_config)
      backtest_results = news_driven_framework.backtest(["2022-12-13"])

      print(backtest_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run macro data aggregation")
    parser.add_argument("--aggregate", action="store_true", default=False, help="Run data aggregation process")
    parser.add_argument("--backtest", action="store_true", default=False, help="Run Backtest Framework")
    args = parser.parse_args()

    main(aggregate=args.aggregate, backtest=args.backtest)



