import pandas as pd
import textwrap
import sys
import os
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utilities.Logger import logger
from LLMAgent.InstructionPrompt import *
from LLMAgent import FilterAgent
from procoder.functional import format_prompt
from procoder.prompt import NamedBlock

# Recent Macroeconomic News Block
MACROECONOMIC_NEWS_PROMPT = NamedBlock(
    name="Recent Macroeconomic News",
    content=textwrap.dedent("""\
        ```
        Below is a summary of recent macroeconomic news articles and key events
        on {current_date} that may impact financial markets.

        {news_chunk}
        ```
    """)
)

class MacroAggregator:

    def __init__(self, config: dataclass, current_date: str):
        """
        Initializes the MacroAggregator.
        
        :param config: An instance of MacroAggregatorConfig containing all parameters.
        """
        self.config = config
        self.news_path = config.news_path
        self.prompt_num_relevance = config.prompt_num_relevance
        self.asset = config.asset
        self.model = config.model
        self.output_path = config.output_path
        self.verbose = config.verbose
        self.macro_csv_list = config.macro_csv_list or []
        self.last_periods_list = config.last_periods_list or []
        self.mapping_csv = config.mapping_csv

        self.current_date = current_date
        self.agent = FilterAgent(name="FilterAgent", asset=self.asset, prompt_num_relevance=self.prompt_num_relevance, model=self.model)
        self.log = logger(name="MacroAggregator", log_file=f"Logs/macro_aggregator.log")


    def aggregate_news(self, filter_dates=None, filter_agent=False, max_retries=3, chunk_size=15):
        """
        Loads the filtered news CSV and returns entries matching the given list of dates.
        If filter_agent is True, it calls aggregate_news_llm instead of loading from the file.

        :param filter_dates: List of dates to filter the news.
        :param filter_agent: Flag to control whether to use aggregate_news_llm instead of loading from file.
        :param max_retries: Maximum number of retries for failed filtering attempts.
        :param chunk_size: Number of news items to process in each chunk.
        :return: DataFrame containing filtered news for the specified dates.
        """
        if filter_agent:
            self.log.info("Using aggregate_news_llm instead of loading from file...")
            print(filter_dates)
            return self.aggregate_news_llm(filter_dates=filter_dates, max_retries=max_retries, chunk_size=chunk_size)

        try:
            news_chunk = format_macro_news(csv_file=self.output_path, filter_dates=filter_dates, chunk_size=10)
            self.log.info(f"Loaded filtered news from {self.output_path}")
            return news_chunk
        except FileNotFoundError:
            self.log.warning(f"File not found: {self.output_path}")
        except Exception as e:
            self.log.error(f"Error loading CSV: {e}")

        return pd.DataFrame()

    def aggregate_news_llm(self, filter_dates, max_retries=3, chunk_size=15):
        """
        Processes and concatenates impactful news for all given dates and saves it as a CSV.

        :param filter_dates: List of dates to filter news.
        :param max_retries: Maximum number of retries for failed filtering attempts.
        :return: Concatenated DataFrame of impactful news.
        """
        all_news = []
        news_chunks = format_macro_news(self.news_path, filter_dates=filter_dates, chunk_size=chunk_size)

        for i, chunk in enumerate(news_chunks):
            attempts = 0
            status_flag = False
            impactful_news = pd.DataFrame()
            
            while attempts < max_retries and not status_flag:
                self.log.info()
                impactful_news, status_flag = self.agent.filter_news(chunk)
                attempts += 1
                if not status_flag:
                    self.log.warning(f"Retrying filter_news for chunk {i+1} (Attempt {attempts}/{max_retries})")

                self.log.info(f"Received {len(impactful_news)} impactful news items for chunk {i+1} (Attempt {attempts}/{max_retries})")

            all_news.append(impactful_news)

            if self.verbose:
                self.log.info(f"=== News Chunk {i+1} ===")
                for _, row in impactful_news.iterrows():
                    self.log.info(f"Title: {row['Title']}")
                    self.log.info(f"Relevance: {row['Relevance']}\n")

        new_df = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()
        self.save_news_chunks(self.output_path, new_df)

        return format_macro_news(self.output_path)


    def aggregate_indicators(self):
        """
        Aggregates macro indicators based on the initialized parameters.

        :return: Combined text of formatted macro indicators.
        """
        indicator_text = []
        for macro_csv, last_periods in zip(self.macro_csv_list, self.last_periods_list):
            indicator_text.append(format_macro_indicator(macro_csv, self.mapping_csv, self.current_date, last_periods=last_periods))

        combined_indicator_text = "\n\n".join(indicator_text)
        # self.log.debug(combined_indicator_text)

        return combined_indicator_text
    

    def aggregate_all(self, filter_dates=None, filter_agent=False, max_retries=3, chunk_size=15):
        """
        Combines macro indicators and filtered news into a single aggregated output.

        :param filter_dates: List of dates to filter news.
        :param max_retries: Maximum number of retries for failed filtering attempts.
        :return: A dictionary containing aggregated macro indicators and news.
        """
        self.log.info("Aggregating macro indicators and news...")
        
        # Aggregate macro indicators
        indicator_text = self.aggregate_indicators()
        
        # Aggregate news using LLM
        news_chunk = self.aggregate_news(filter_dates=filter_dates, filter_agent=filter_agent, max_retries=max_retries, chunk_size=chunk_size)
        news_text = format_prompt(MACROECONOMIC_NEWS_PROMPT,{"current_date": self.current_date, "news_chunk": news_chunk[0]})

        # Combine both aggregations
        aggregated_output = f"Macro Indicators:\n{indicator_text}\n\n{news_text}"
        
        self.log.debug(aggregated_output)
        self.log.info(f"Aggregation complete for {self.current_date}.")
        return aggregated_output


    def save_news_chunks(self, output_path, new_df):
        """
        Saves the filtered news to the specified output_path. If a file already exists, it merges new entries,
        removes duplicates, and updates the file. Handles potential format mismatches during merging.

        :param output_path: Path to the output CSV file.
        :param new_df: DataFrame containing the newly filtered news.
        """
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                self.log.info(f"Loaded existing news data from {output_path}.")
            except Exception as e:
                self.log.error(f"Error loading existing CSV: {e}")
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()

        try:
            # Attempt to merge new and existing data, remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates()
        except Exception as e:
            self.log.error(f"Error merging new and existing news data: {e}")
            self.log.warning("Saving only the new filtered news instead.")
            combined_df = new_df  # Fall back to saving only the new data

        if not combined_df.empty:
            combined_df.to_csv(output_path, index=False)
            self.log.info(f"Updated filtered news saved to {output_path}.")
        else:
            self.log.warning("No filtered news to save.")


log = logger(name="FileChecker", log_file=f"Logs/file_checker.log")

def check_file_paths(file_paths):
    """
    Checks whether the given list of CSV file paths exist, are valid CSV files, and are not empty.
    
    Args:
        file_paths (list of str): List of file paths to check.
    
    Returns:
        None
    """
    
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            log.warning(f"File does not exist: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                log.warning(f"File is empty: {file_path}")

        except Exception as e:
            log.error(f"Invalid CSV format for {file_path}: {e}")
            return False

    log.info(f"All file paths are valid and non-empty")
    return True