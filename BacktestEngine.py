import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from Backtest import MacroAggregator, check_file_paths
from LLMAgent.InstructionPrompt import *
from DataPipeline import write_mapping

@dataclass
class Config:
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
    config = Config()
    check_file_paths([*config.macro_csv_list, config.news_path, config.mapping_csv])

    if aggregate:
        aggregator = MacroAggregator(
            news_path=config.news_path,
            asset="US 10-year Treasury bonds",
            model="deepseek-r1:8b",
            output_path=config.output_path,
            macro_csv_list=config.macro_csv_list,
            mapping_csv=config.mapping_csv,
            current_date=config.dates[0],
            last_periods_list=config.last_periods_list
        )

        event_prompt = aggregator.aggregate_all(filter_dates=config.dates, filter_agent=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run macro data aggregation")
    parser.add_argument("--aggregate", action="store_true", default=False, help="Run data aggregation process")
    args = parser.parse_args()

    main(aggregate=args.aggregate)