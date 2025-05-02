import pandas as pd
import multiprocessing
import time
import os
from datetime import timedelta

from Backtest import MacroAggregator
from LLMAgent import TradingAgent, MultiAgentNetwork
from LLMAgent.InstructionPrompt import *
from Utilities import logger


# Single Agent Strategy
class NewsDrivenStrategy:

    def __init__(self, dates: list, filter_agent: bool, chunk_size: int, num_processes: int, asset: str, 
                 ticker: str, lookback_period: int, model_aggregate: str, model_trading: str, trading_system_prompt: bool, 
                 results_path: str, chat_history_path, aggregator: MacroAggregator):
        """Initialize the strategy with the given parameters."""
        self.asset = asset
        self.ticker = ticker
        self.lookback_period = lookback_period
        self.model_aggregate = model_aggregate
        self.model_trading = model_trading
        self.dates = dates
        self.filter_agent = filter_agent
        self.chunk_size = chunk_size
        self.num_processes = num_processes
        self.trading_system_prompt = trading_system_prompt
        self.results_path = results_path
        self.chat_history_path = chat_history_path

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
            chat_history_path=self.chat_history_path,
        )

        # Set up logging
        self.log = logger(name="NewsDrivenStrategy", log_file=f"Logs/backtest.log")

    def single_day_backtest(self, date, lookback_period, aggregator, filter_agent, chunk_size, agent):
        """Run backtest for a single date."""
        log = logger(name="NewsDrivenStrategy", log_file=f"Logs/backtest.log")
        results = []
        
        # Aggregate data for the current date
        log.info(f"Running backtest for date: [{date}]")

        # Measure aggregation time
        log.info(f"Aggregating data for {date.strftime('%Y-%m-%d')}...")
        aggregation_start_time = time.time()
        start_date = date - timedelta(days=lookback_period)
        filter_dates = [start_date + timedelta(days=i) for i in range(lookback_period + 1)]

        aggregator.set_current_date(current_date=date)  # To filter only the current date
        input_prompt = aggregator.aggregate_all(
            filter_dates=[filter_dates],
            filter_agent=filter_agent,
            chunk_size=chunk_size
        )
        
        aggregation_elapsed_time = time.time() - aggregation_start_time
        log.info(f"Finished aggregating data for {date.strftime('%Y-%m-%d')} (Took {aggregation_elapsed_time:.2f} seconds)")

        # Get trading decision from the agent
        start_time = time.time()

        # prediction, explanation = "N/A", "N/A" # Debugging purposes
        prediction, explanation = agent.get_trading_decision(input_prompt)
        decision = sentiment_to_decision(prediction=prediction)
        elapsed_time = time.time() - start_time

        # Log trading decision
        log.info(f"Prediction: {prediction} | Trading decision {decision} for date: [{date}]")
        log.info(f"Decision time: {elapsed_time:.2f} seconds for date: [{date}]")

        # Store backtest results for the current date and chat_history
        results.append({"Date": date, "Agent": self.name, "Prediction": prediction, "Decision": decision, "Explanation": explanation})
        agent.save_chat_history(date=date)

        return results

    def backtest(self):
        """Run a backtest over the specified date range with optional multiprocessing."""
        open("Logs/backtest.log", "w").close()
        results = []

        date_range = pd.date_range(start=self.dates[0], end=self.dates[1])

        self.log.info(f"Starting backtesting with {self.num_processes} processes and {self.lookback_period} lookback periods")

        if self.num_processes == 1:
            # Serial processing
            for date in date_range:
                day_results = self.single_day_backtest(date, self.lookback_period, self.aggregator, self.filter_agent, self.chunk_size, self.agent)
                results.append(day_results)
        else:
            # Parallel processing
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.starmap(self.single_day_backtest, [
                    (date, self.lookback_period, self.aggregator, self.filter_agent, self.chunk_size, self.agent) 
                    for date in date_range
                ])

        # Flatten the list of results and convert to DataFrame
        flat_results = [item for sublist in results for item in sublist]
        results_df = pd.DataFrame(flat_results, columns=["Date", "Agent", "Prediction", "Decision", "Explanation"])

        self.save_results(results_df)

        return results_df
    

    # Concatenate new results to the old results csv (duplicate entries will be replaced)
    def save_results(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        if os.path.exists(self.results_path) and os.path.getsize(self.results_path) > 0:
            existing_df = pd.read_csv(self.results_path)
            
            # Ensure 'Date' is of type datetime in both DataFrames
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=["Date"], keep="last")
            merged_df = merged_df.sort_values(by="Date")
            merged_df.to_csv(self.results_path, index=False)
            self.log.info(f"Results updated and saved to {self.results_path}")
        else:
            df.to_csv(self.results_path, index=False)
            self.log.info(f"Results saved to {self.results_path}")






# Multi Agent Strategy
class DebateDrivenStrategy:

    def __init__(self, dates: list, filter_agent: bool, chunk_size: int, num_processes: int, 
                 max_rounds: int, asset: str, lookback_period: int,
                 ticker: str, model_aggregate: str, model_trading: str, trading_system_prompt: bool, 
                 results_path: str, aggregator: MacroAggregator):
        """Initialize the strategy with the given parameters."""
        self.asset = asset
        self.ticker = ticker
        self.lookback_period = lookback_period
        self.model_aggregate = model_aggregate
        self.model_trading = model_trading
        self.dates = dates
        self.filter_agent = filter_agent
        self.chunk_size = chunk_size
        self.num_processes = num_processes
        self.max_rounds = max_rounds
        self.trading_system_prompt = trading_system_prompt
        self.results_path = results_path

        self.aggregator = aggregator

        self.name = ["RiskAverseAgent", "RiskNeutralAgent", "RiskSeekingAgent"]
        self.logger_name = ["backtest", "backtest", "backtest"]
        self.style = ["Risk Averse", "Risk Neutral","Risk Seeking"]
        self.risk_tolerance = ["low", "medium", "high"]

        # Decision making agent for trading
        self.network = MultiAgentNetwork(
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
        self.log = logger(name="DebateDrivenStrategy", log_file=f"Logs/backtest.log")

    def single_day_backtest(self, date, lookback_period, aggregator, filter_agent, chunk_size, network):
        """Run backtest for a single date."""
        log = logger(name="DebateDrivenStrategy", log_file=f"Logs/backtest.log")
        results = []
        
        # Aggregate data for the current date
        log.info(f"Running backtest for date: [{date}]")

        # Measure aggregation time
        log.info(f"Aggregating data for {date.strftime('%Y-%m-%d')}...")
        aggregation_start_time = time.time()
        start_date = date - timedelta(days=lookback_period)
        filter_dates = [start_date + timedelta(days=i) for i in range(lookback_period + 1)]

        aggregator.set_current_date(current_date=date)  # To filter only the current date
        input_prompt = aggregator.aggregate_all(
            filter_dates=[filter_dates],
            filter_agent=filter_agent,
            chunk_size=chunk_size
        )
        
        aggregation_elapsed_time = time.time() - aggregation_start_time
        log.info(f"Finished aggregating data for {date.strftime('%Y-%m-%d')} (Took {aggregation_elapsed_time:.2f} seconds)")

        # Get trading decision from the agent
        start_time = time.time()
        final_opinions = network.get_trading_decision(input_prompt=input_prompt, max_rounds=self.max_rounds)

        elapsed_time = time.time() - start_time
        log.info(f"Decision time: {elapsed_time:.2f} seconds for date: [{date}]")

        results = self._extract_final_opinions(date=date, final_opinions=final_opinions, log=log)

        return results

    def backtest(self):
        """Run a backtest over the specified date range with optional multiprocessing."""
        open("Logs/backtest.log", "w").close()
        results = []

        date_range = pd.date_range(start=self.dates[0], end=self.dates[1])

        self.log.info(f"Starting backtesting with {self.num_processes} processes")

        if self.num_processes == 1:
            # Serial processing
            for date in date_range:
                day_results = self.single_day_backtest(date, self.lookback_period, self.aggregator, self.filter_agent, self.chunk_size, self.network)
                results.append(day_results)
        else:
            # Parallel processing
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.starmap(self.single_day_backtest, [
                    (date, self.lookback_period, self.aggregator, self.filter_agent, self.chunk_size, self.agent) 
                    for date in date_range
                ])

        # Flatten the list of results and convert to DataFrame
        flat_results = [item for sublist in results for item in sublist]
        results_df = pd.DataFrame(flat_results, columns=["Date", "Agent", "Prediction", "Decision", "Explanation"])

        self.save_results(results_df)

        return results_df
    
    def _extract_final_opinions(self, date, final_opinions: dict, log: logger):

        results = []
        # Save the final opinions for each agent
        for agent_name, opinion in final_opinions.items():
            prediction = opinion['prediction']
            explanation = opinion['explanation']
            decision = sentiment_to_decision(prediction=prediction)

            # Log the final opinion of the agent
            log.info(f"Agent: {agent_name} | Prediction: {prediction} | Explanation: {explanation}")

            # Store the results for the current date
            results.append({
                "Date": date,
                "Agent": agent_name,
                "Prediction": prediction,
                "Decision": decision,
                "Explanation": explanation,
            })    

        return results
    
    # Concatenate new results to the old results csv (duplicate entries will be replaced)
    def save_results(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        if os.path.exists(self.results_path) and os.path.getsize(self.results_path) > 0:
            existing_df = pd.read_csv(self.results_path)
            
            # Ensure 'Date' is of type datetime in both DataFrames
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Merge and remove duplicates based on both 'Date' and 'Agent'
            merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=["Date", "Agent"], keep="last")
            merged_df = merged_df.sort_values(by="Date")
            
            # Save the updated results to the CSV file
            merged_df.to_csv(self.results_path, index=False)
            self.log.info(f"Results updated and saved to {self.results_path}")
        else:
            # If the file doesn't exist, save the new results directly
            df.to_csv(self.results_path, index=False)
            self.log.info(f"Results saved to {self.results_path}")

