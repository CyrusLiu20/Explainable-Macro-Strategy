import pandas as pd
import time

from Backtest import MacroAggregator
from LLMAgent import TradingAgent
from LLMAgent.InstructionPrompt import *
from Utilities import logger

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
        self.log = logger(name="NewsDrivenFramework", log_file=f"Logs/backtest.log")

    def backtest(self):
        """Run a backtest over the specified date range.

        Args:
            date_range (iterable): A list or range of dates to backtest.

        Returns:
            pd.DataFrame: A DataFrame containing the backtest results with columns:
                           ["Date", "Prediction", "Explanation"].
        """
        open("Logs/backtest.log", "w").close()
        results = []

        date_range = pd.date_range(start=self.config.dates[0], end=self.config.dates[1])
        for date in date_range:

            # Aggregate data for the current date
            self.log.info(f"Running backtest for date: {date}")
            aggregator = MacroAggregator(config=self.config, current_date=date)

            input_prompt = aggregator.aggregate_all(filter_dates=[date], filter_agent=self.config.filter_agent, chunk_size=self.config.chunk_size)

            # Get trading decision from the agent
            start_time = time.time()
            prediction, explanation = self.agent.get_trading_decision(input_prompt)
            elapsed_time = time.time() - start_time

            # Log trading decision
            self.log.info(f"Prediction: {prediction}")
            self.log.info(f"Explanation: {explanation}")
            self.log.info(f"Decision time: {elapsed_time:.2f} seconds")


            # Store results for the current date
            results.append({"Date": date, "Prediction": prediction, "Explanation": explanation})

        # Convert results to a DataFrame
        return pd.DataFrame(results, columns=["Date", "Prediction", "Explanation"])