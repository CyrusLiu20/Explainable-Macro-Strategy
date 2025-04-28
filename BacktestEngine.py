import argparse
from pickle import STRING
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from Backtest import MacroAggregator, check_file_paths
from Backtest import NewsDrivenStrategy, DebateDrivenStrategy
from LLMAgent.InstructionPrompt import *
from DataPipeline import write_mapping
from Utilities import BacktestConfigurationLoader, filter_valid_kwargs

def main(multi_agent: bool, config_path: str):

    ##################################### Load Backtest Configuration #####################################
    backtest_config_loader = BacktestConfigurationLoader(config_path=config_path)
    backtest_config = backtest_config_loader.get_config()

    files_exist = check_file_paths([*backtest_config_loader.macro_csv_list, backtest_config_loader.news_path, backtest_config_loader.mapping_csv])
    write_mapping(folder_path=backtest_config_loader.data_root/"MacroIndicators")
    ##################################### Load Backtest Configuration #####################################



    ##################################### Strategy Backtest #####################################
    aggregator_kwargs = filter_valid_kwargs(MacroAggregator, backtest_config)
    aggregator = MacroAggregator(**aggregator_kwargs)

    if not multi_agent:
      backtest_kwargs = filter_valid_kwargs(NewsDrivenStrategy, backtest_config)
      news_driven_strategy = NewsDrivenStrategy(aggregator=aggregator, **backtest_kwargs)
      backtest_results = news_driven_strategy.backtest()
    else:
      backtest_kwargs = filter_valid_kwargs(DebateDrivenStrategy, backtest_config)
      debate_driven_strategy = DebateDrivenStrategy(aggregator=aggregator, **backtest_kwargs)
      backtest_results = debate_driven_strategy.backtest()
    ##################################### Strategy Backtest #####################################


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Explainable Macro Strategy Backtest")
  parser.add_argument("-m", "--multi-agent", action="store_true", default=False, help="Run Multi-Agent Backtest Strategy")
  parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file (YAML)")
  args = parser.parse_args()

  main(multi_agent=args.multi_agent, config_path=args.config)



