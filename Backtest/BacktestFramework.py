import pandas as pd
import multiprocessing
import time
import os

from Backtest import MacroAggregator
from LLMAgent import TradingAgent
from LLMAgent.InstructionPrompt import *
from Utilities import logger


# Mapping market sentiment to trading decision
def sentiment_to_decision(prediction):
    sentiment_map = {
        "Strongly Bullish": 2,
        "Bullish": 1,
        "Slightly Bullish": 0.5,
        "Flat": 0,
        "Fluctuating": 0,
        "Slightly Bearish": -0.5,
        "Bearish": -1,
        "Strongly Bearish": -2
    }

    return sentiment_map.get(prediction, 0) # Default to 0 if the prediction is not recognized


class NewsDrivenFramework:

    def __init__(self, dates: list, filter_agent: bool, chunk_size: int, asset: str, 
                 ticker: str, model_aggregate: str, model_trading: str, trading_system_prompt: bool, 
                 results_path: str, aggregator: MacroAggregator):
        """Initialize the framework with the given parameters."""
        self.asset = asset
        self.ticker = ticker
        self.model_aggregate = model_aggregate
        self.model_trading = model_trading
        self.dates = dates
        self.filter_agent = filter_agent
        self.chunk_size = chunk_size
        self.trading_system_prompt = trading_system_prompt
        self.results_path = results_path

        self.aggregator = aggregator

        self.name = "NewsDrivenAgent"
        self.logger_name = "backtest"
        self.style = "risk_neutral"
        self.risk_tolerance = "medium"

        # Decision making agent for trading
        self.agent = TradingAgent(
            asset=self.asset,
            ticker=self.ticker,
            name=self.name,
            logger_name=self.logger_name,
            model=self.model_trading,
            style=self.style,
            risk_tolerance=self.risk_tolerance,
            has_system_prompt=self.trading_system_prompt,
        )

        # Set up logging
        self.log = logger(name="NewsDrivenFramework", log_file=f"Logs/backtest.log")

    def single_day_backtest(self, date, aggregator, filter_agent, chunk_size, agent):
        """Run backtest for a single date."""
        log = logger(name="NewsDrivenFramework", log_file=f"Logs/backtest.log")
        results = []
        
        # Aggregate data for the current date
        log.info(f"Running backtest for date: [{date}]")

        # Measure aggregation time
        log.info(f"Aggregating data for {date.strftime('%Y-%m-%d')}...")
        aggregation_start_time = time.time()
        
        aggregator.set_current_date(current_date=date)  # To filter only the current date
        input_prompt = aggregator.aggregate_all(
            filter_dates=[date],
            filter_agent=filter_agent,
            chunk_size=chunk_size
        )
        
        aggregation_elapsed_time = time.time() - aggregation_start_time
        log.info(f"Finished aggregating data for {date.strftime('%Y-%m-%d')} (Took {aggregation_elapsed_time:.2f} seconds)")

        # Get trading decision from the agent
        start_time = time.time()
        prediction, explanation = agent.get_trading_decision(input_prompt)
        decision = sentiment_to_decision(prediction=prediction)
        elapsed_time = time.time() - start_time

        # Log trading decision
        log.info(f"Prediction: {prediction} | Trading decision {decision} for date: [{date}]")
        log.info(f"Decision time: {elapsed_time:.2f} seconds for date: [{date}]")

        # Store results for the current date
        results.append({"Date": date, "Prediction": prediction, "Decision": decision, "Explanation": explanation})

        return results

    def backtest(self):
        """Run a backtest over the specified date range with multiprocessing."""
        open("Logs/backtest.log", "w").close()
        results = []

        date_range = pd.date_range(start=self.dates[0], end=self.dates[1])

        # Set up a pool of workers for parallel execution
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # Distribute the work across multiple processes
            results = pool.starmap(self.single_day_backtest, [
                (date, self.aggregator, self.filter_agent, self.chunk_size, self.agent) 
                for date in date_range
            ])

        # Flatten the list of results and convert to DataFrame
        flat_results = [item for sublist in results for item in sublist]
        results_df = pd.DataFrame(flat_results, columns=["Date", "Prediction", "Decision", "Explanation"])
        
        self.save_results(results_df)

        return results_df
    

    def save_results(self, df: pd.DataFrame):

        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        df.to_csv(self.results_path, index=False)
        self.log.info(f"Results saved to {self.results_path}")