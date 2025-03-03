import pandas as pd
import textwrap
import sys
import os

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
    def __init__(self, news_path, asset, model, output_path="filtered_news.csv", verbose=True,
                 macro_csv_list=None, last_periods_list=None, mapping_csv=None, current_date=None, last_periods=4):
        """
        Initializes the MacroAggregator.
        
        :param news_path: Path to the macro news CSV file.
        :param asset: The asset related to the news.
        :param model: The model used for filtering news.
        :param output_path: Path to save the filtered news CSV.
        :param verbose: Boolean flag to control printing of news details.
        :param macro_csv_list: List of paths to macro indicator CSV files.
        :param last_periods_list: List of periods to consider for each macro indicator.
        :param mapping_csv: Path to the mapping CSV file for macro indicators.
        :param current_date: The current date for filtering indicators.
        :param last_periods: Default number of periods to consider for indicators.
        """
        self.news_path = news_path
        self.asset = asset
        self.model = model
        self.output_path = output_path
        self.verbose = verbose
        self.macro_csv_list = macro_csv_list or []
        self.last_periods_list = last_periods_list or []
        self.mapping_csv = mapping_csv
        self.current_date = current_date
        self.agent = FilterAgent(name="FilterAgent", asset=self.asset, model=self.model)
        self.log = logger(name="MacroAggregator", log_file=f"Logs/macro_aggregator.log")

    def aggregate_news(self, filter_dates=None, filter_agent=False, max_retries=3):
        """
        Loads the filtered news CSV and returns entries matching the given list of dates.
        If the file is not found or use_aggregation flag is set, aggregate_news_llm is called to fetch data.

        :param filter_dates: List of dates to filter the news.
        :param filter_agent: Flag to control whether to use aggregate_news_llm if file is not found.
        :param max_retries: Maximum number of retries for failed filtering attempts.
        :return: DataFrame containing filtered news for the specified dates.
        """
        try:
            news_chunk = format_macro_news(csv_file=self.output_path, filter_dates=filter_dates, chunk_size=1e1)
            self.log.info(f"Loaded filtered news from {self.output_path}")
        except FileNotFoundError:
            self.log.warning(f"File not found: {self.output_path}")
            if filter_agent:
                self.log.info("Aggregating news instead, as the file is missing...")
                return self.aggregate_news_llm(filter_dates=filter_dates, max_retries=max_retries)
            else:
                return pd.DataFrame()
        except Exception as e:
            self.log.error(f"Error loading CSV: {e}")
            return pd.DataFrame()

        return news_chunk

    def aggregate_news_llm(self, filter_dates, max_retries=3):
        """
        Processes and concatenates impactful news for all given dates and saves it as a CSV.

        :param filter_dates: List of dates to filter news.
        :param max_retries: Maximum number of retries for failed filtering attempts.
        :return: Concatenated DataFrame of impactful news.
        """
        all_news = []
        news_chunks = format_macro_news(self.news_path, filter_dates=filter_dates, chunk_size=15)

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
                self.log.info(f"=== News Chunk {i+1} ===")
                for _, row in impactful_news.iterrows():
                    self.log.info(f"Title: {row['Title']}")
                    self.log.info(f"Relevance: {row['Relevance']}\n")

        self.final_df = pd.concat(all_news, ignore_index=True) if all_news else pd.DataFrame()

        if not self.final_df.empty:
            self.final_df.to_csv(self.output_path, index=False)
            self.log.info(f"Filtered news saved to {self.output_path}.")
        else:
            self.log.warning("No filtered news to save.")

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
    

    def aggregate_all(self, filter_dates=None, filter_agent=False, max_retries=3):
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
        news_chunk = self.aggregate_news(filter_dates=filter_dates, filter_agent=filter_agent, max_retries=max_retries)
        news_text = format_prompt(MACROECONOMIC_NEWS_PROMPT,{"current_date": self.current_date, "news_chunk": news_chunk[0]})

        # Combine both aggregations
        aggregated_output = f"Macro Indicators:\n{indicator_text}\n\n{news_text}"
        
        self.log.debug(aggregated_output)
        self.log.info(f"Aggregation complete for {self.current_date}.")
        return aggregated_output


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

    log.info(f"All file paths are valid and non-empty")