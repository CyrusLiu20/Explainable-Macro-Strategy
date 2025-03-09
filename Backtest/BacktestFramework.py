import pandas as pd

from Backtest import MacroAggregator
from LLMAgent import TradingAgent
from LLMAgent.InstructionPrompt import *


class NewsDrivenFramework:
    """A framework for backtesting trading strategies driven by news and macroeconomic data."""

    def __init__(self, config):
        """Initialize the framework with the given configuration."""
        self.config = config
        self.agent = TradingAgent(
            asset=self.config.asset,
            ticker=self.config.ticker,
            name="NewsDrivenAgent",
            logger_name="NewsDrivenAgent",
            model=self.config.model,
            style="risk_neutral",
            risk_tolerance="medium",
            has_system_prompt = self.config.has_system_prompt,
        )

    def backtest(self, date_range):
        """Run a backtest over the specified date range.

        Args:
            date_range (iterable): A list or range of dates to backtest.

        Returns:
            pd.DataFrame: A DataFrame containing the backtest results with columns:
                           ["Date", "Prediction", "Explanation"].
        """
        results = []

        for date in date_range:
            # Aggregate data for the current date
            # aggregator = MacroAggregator(
            #     news_path=self.config.news_path,
            #     asset=self.config.asset,
            #     model=self.config.model,
            #     output_path=self.config.output_path,
            #     macro_csv_list=self.config.macro_csv_list,
            #     mapping_csv=self.config.mapping_csv,
            #     current_date=date,
            #     last_periods_list=self.config.last_periods_list,
            # )
            aggregator = MacroAggregator(config=self.config, current_date=date)

            input_prompt = aggregator.aggregate_all(filter_dates=self.config.dates, filter_agent=self.config.filter_agent, chunk_size=self.Config.chunk_size)

            # Get trading decision from the agent
            prediction, explanation = self.agent.get_trading_decision(input_prompt)

            # Store results for the current date
            results.append({"Date": date, "Prediction": prediction, "Explanation": explanation})

        # Convert results to a DataFrame
        return pd.DataFrame(results, columns=["Date", "Prediction", "Explanation"])