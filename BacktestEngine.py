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
    asset: str = "US 10-year Treasury bonds"
    model: str = "deepseek-r1:32b"
    has_system_prompt: bool = False # Whether the LLM model has a system message
    ticker: str = "IEF"
    data_root: Path = Path("DataPipeline/Data")
    output_path: Path = Path("Backtest/AggregatedData/AggregatedNews.csv")
    dates: list = field(default_factory=lambda: ["2022-12-13"])
    last_periods_list: list = field(default_factory=lambda: [10, 4, 6, 4])

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

def main(aggregate: bool):
    backtest_config = BacktestConfig()
    files_exist = check_file_paths([*backtest_config.macro_csv_list, backtest_config.news_path, backtest_config.mapping_csv])

    if aggregate and files_exist:
        # aggregator = MacroAggregator(
        #     news_path=backtest_config.news_path,
        #     asset=backtest_config.asset,
        #     model=backtest_config.model,
        #     output_path=backtest_config.output_path,
        #     macro_csv_list=backtest_config.macro_csv_list,
        #     mapping_csv=backtest_config.mapping_csv,
        #     current_date=backtest_config.dates[0],
        #     last_periods_list=backtest_config.last_periods_list
        # )

        # event_prompt = aggregator.aggregate_all(filter_dates=backtest_config.dates, filter_agent=False)

      news_driven_framework = NewsDrivenFramework(backtest_config)
      backtest_results = news_driven_framework.backtest(["2022-12-13"])

      print(backtest_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run macro data aggregation")
    parser.add_argument("--aggregate", action="store_true", default=False, help="Run data aggregation process")
    args = parser.parse_args()

    main(aggregate=args.aggregate)



