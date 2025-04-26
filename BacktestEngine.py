import argparse
from pickle import STRING
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from Backtest import MacroAggregator, check_file_paths
from Backtest import NewsDrivenFramework
from LLMAgent.InstructionPrompt import *
from DataPipeline import write_mapping
from Utilities import load_config

def main(aggregate: bool, backtest: bool):

    # backtest_config = BacktestConfig()
    backtest_config = load_config("Config.yaml")
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
      backtest_results = news_driven_framework.backtest()

      print(backtest_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run macro data aggregation")
    parser.add_argument("--aggregate", action="store_true", default=False, help="Run data aggregation process")
    parser.add_argument("--backtest", action="store_true", default=False, help="Run Backtest Framework")
    args = parser.parse_args()

    main(aggregate=args.aggregate, backtest=args.backtest)



