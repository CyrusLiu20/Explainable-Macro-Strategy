import os
import pandas as pd
from alpha_vantage.alphaintelligence import AlphaIntelligence
import json

from Utilities.Logger import scraper_logger  # Assuming you have a custom logger utility.

class AlphaVantageScraper:
    def __init__(self, api_key=None, config_file="alpha_vantage_config.json", log_file="alpha_vantage_scraper.log"):
        """
        Initialize the scraper with an API key for Alpha Vantage API.
        :param api_key: API key for Alpha Vantage. If not provided, it will be fetched from environment variables.
        :param config_file: Path to the JSON configuration file for topics and time frames.
        :param log_file: Path to the log file.
        """
        self.api_key = api_key or os.getenv('ALPHAVANTAGE_API_KEY')
        self.ai = AlphaIntelligence(key=self.api_key, output_format="pandas")
        self.config_file = config_file
        self.topics_config = self.load_config()
        self.name = "Alpha Vantage Scraper"

        # Setup logging
        self.logger = scraper_logger(name=self.name, log_file=log_file)

    def load_config(self):
        """
        Load topics configuration from a JSON file.
        :return: List of dictionaries containing the topics configuration.
        """
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration file {self.config_file}: {e}")
            return []

    def fetch_news_sentiment(self, topics, time_from, time_to, sort='LATEST', limit=50):
        """
        Fetches news sentiment from Alpha Vantage API.
        :param topics: Topics to fetch news for.
        :param time_from: Start time for the news data (YYYYMMDDTHHMM).
        :param time_to: End time for the news data (YYYYMMDDTHHMM).
        :param sort: Sorting order (default is 'LATEST').
        :param limit: The number of records to fetch.
        :return: A Pandas DataFrame with the news sentiment data.
        """
        try:
            news_data = self.ai.get_news_sentiment(topics=topics, time_from=time_from, time_to=time_to, sort=sort, limit=limit)
            if not news_data[0].empty:
                self.logger.info(f"Successfully fetched {news_data[0].shape[0]} records for {topics}.")
                return news_data[0]
            else:
                self.logger.warning(f"No data returned for {topics}.")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching news sentiment for {topics}: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df, folder_path, file_name):
        """
        Saves the DataFrame to a CSV file.
        :param df: The DataFrame to be saved.
        :param folder_path: The folder path to save the file.
        :param file_name: The name of the file to save.
        """
        try:
            df.to_csv(os.path.join(folder_path, file_name), index=True)
            self.logger.info(f"Data saved successfully to {os.path.join(folder_path, file_name)}.")
        except Exception as e:
            self.logger.error(f"Error saving {file_name}: {e}")

    def scrape_and_save_all(self, folder_path):
        """
        Fetches and saves all topics specified in the JSON config file.
        """
        if not self.topics_config:
            self.logger.error("No topics found in configuration file.")
            return

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            self.logger.debug(f"Directory '{folder_path}' created.")

        for topic in self.topics_config:
            topic_name = topic["topic"]
            time_from = topic["time_from"]
            time_to = topic["time_to"]
            file_name = topic["file_name"]

            df = self.fetch_news_sentiment(topic_name, time_from, time_to)
            if not df.empty:
                self.save_to_csv(df, folder_path, file_name)
