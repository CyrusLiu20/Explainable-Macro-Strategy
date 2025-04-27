import argparse
from pickle import STRING
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from Backtest import MacroAggregator, check_file_paths
from Backtest import NewsDrivenFramework
from LLMAgent.InstructionPrompt import *
from DataPipeline import write_mapping
from Utilities import BacktestConfigurationLoader, filter_valid_kwargs

def main(backtest: bool):

    # backtest_config = load_config("Config.yaml")
    backtest_config_loader = BacktestConfigurationLoader(config_path="Config.yaml")
    backtest_config = backtest_config_loader.get_config()

    files_exist = check_file_paths([*backtest_config_loader.macro_csv_list, backtest_config_loader.news_path, backtest_config_loader.mapping_csv])
    write_mapping(folder_path=backtest_config_loader.data_root/"MacroIndicators")

    if backtest:

      aggregator_kwargs = filter_valid_kwargs(MacroAggregator, backtest_config)
      aggregator = MacroAggregator(**aggregator_kwargs)

      backtest_kwargs = filter_valid_kwargs(NewsDrivenFramework, backtest_config)
      news_driven_framework = NewsDrivenFramework(aggregator=aggregator, **backtest_kwargs)
      backtest_results = news_driven_framework.backtest()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run macro data aggregation")
    parser.add_argument("--backtest", action="store_true", default=False, help="Run Backtest Framework")
    args = parser.parse_args()

    main(backtest=args.backtest)



