from Backtest import ETFBacktest
from Utilities import BacktestConfigurationLoader, filter_valid_kwargs

if __name__ == "__main__":

    config_path = "multi_agent_config.yaml"
    backtest_config_loader = BacktestConfigurationLoader(config_path=config_path)
    backtest_config = backtest_config_loader.get_config()


    # Initialize backtester
    backtest_kwargs = filter_valid_kwargs(ETFBacktest, backtest_config)
    backtester = ETFBacktest(**backtest_kwargs)
    backtester.run_backtest()