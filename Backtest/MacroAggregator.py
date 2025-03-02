import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *
from LLMAgent.BaseAgent import BaseAgent
from LLMAgent.MacroAgent import FilterAgent

class MacroAggregator:
    def __init__(self, news_path, asset, model, output_path="filtered_news.csv", verbose=True):
        """
        Initializes the NewsAggregator.
        
        :param news_path: Path to the macro news CSV file.
        :param asset: The asset related to the news.
        :param model: The model used for filtering news.
        :param save_path: Path to save the filtered news CSV.
        :param verbose: Boolean flag to control printing of news details.
        """
        self.news_path = news_path
        self.asset = asset
        self.model = model
        self.output_path = output_path
        self.verbose = verbose
        self.agent = FilterAgent(name="FilterAgent", asset=self.asset, model=self.model)
        self.log = logger(name="MacroAggregator", log_file=f"Logs/macro_aggregator.log")

        self.agent.start_ollama_server()

    def aggregate_news(self, filter_dates, max_retries=5):
        """
        Processes and concatenates impactful news for all given dates and saves it as a CSV.

        :param filter_dates: List of dates to filter news.
        :param max_retries: Maximum number of retries for failed filtering attempts.
        :return: Concatenated DataFrame of impactful news.
        """
        all_news = []
        news_chunks = format_macro_news(self.news_path, filter_dates=filter_dates)

        for i, chunk in enumerate(news_chunks):
            attempts = 0
            status_flag = False
            impactful_news = pd.DataFrame()

            while attempts < max_retries and not status_flag:
                impactful_news, status_flag = self.agent.filter_news(chunk)
                attempts += 1

                if not status_flag:
                    self.log.warning(f"Retrying filter_news for chunk {i+1} (Attempt {attempts}/{max_retries})")

            all_news.append(impactful_news)

            if self.verbose:
                print(f"=== News Chunk {i+1} ===")
                for _, row in impactful_news.iterrows():
                    print(f"Title: {row['Title']}")
                    print(f"Relevance: {row['Relevance']}\n")

        self.final_df = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()

        # Save the filtered news DataFrame as a CSV file
        if not self.final_df.empty:
            self.final_df.to_csv(self.output_path, index=False)
            print(f"Filtered news saved to {self.output_path}.")
        else:
            print("No filtered news to save.")

        return self.final_df

    def aggregate_indicators(self, macro_csv, mapping_csv, current_date, last_periods=4):
        pass


if __name__ == "__main__":
  # Example usage:
  news_path = "DataPipeline/Data/ProcessedData/MacroNews.csv"
  output_path = "Backtest/AggregatedData/FilteredNews.csv"
  start_date = "2022-12-13"
  end_date = "2022-12-15"
  dates = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
  # dates = ["2022-12-13", "2022-12-14", "2022-12-15"]  # Example list of dates
  aggregator = MacroAggregator(news_path, asset="US 10-year Treasury bonds", model="deepseek-r1:8b",  output_path=output_path)
  impactful_news_df = aggregator.aggregate_news(dates, max_retries=4)

  print(impactful_news_df)
