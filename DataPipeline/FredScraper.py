import os
import json
import pandas as pd
from fredapi import Fred

from Utilities.Logger import logger

class FredDataScraper:
    def __init__(self, api_key=None, config_file="fred_config.json", log_file="scraper.log"):
        """
        Initialize the scraper with an API key for the FRED API.
        :param api_key: API key for the FRED API. If not provided, it will be fetched from environment variables.
        :param config_file: Path to the JSON configuration file for series data.
        :param log_file: Path to the log file.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.fred = Fred(api_key=self.api_key)
        self.config_file = config_file
        self.series_config = self.load_config()
        self.name = "Fred Data Scrapper"

        # Setup logging
        self.logger = logger(name=self.name, log_file=log_file)

    def load_config(self):
        """
        Load series configuration from a JSON file.
        :return: Dictionary containing the series configuration.
        """
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration file {self.config_file}: {e}")
            return []

    def fetch_series(self, series_id, start_date, end_date):
        """
        Fetches a time series from the FRED API.
        :param series_id: The ID of the FRED series (e.g., 'SP500').
        :param start_date: Start date for the data (YYYY-MM-DD).
        :param end_date: End date for the data (YYYY-MM-DD).
        :return: A Pandas DataFrame with the series data.
        """
        try:
            # self.logger.info(f"Fetching data for {series_id} from {start_date} to {end_date}...")
            series = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            if series is None:
                self.logger.warning(f"No data returned for {series_id}.")
                return pd.DataFrame()

            df = series.to_frame(name=series_id)  # Convert to DataFrame and set column name
            df.index.name = "Date"  # Name the index
            self.logger.info(f"Successfully fetched {df.shape[0]} rows for {series_id}.")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {series_id}: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df, folder_path, file_name):
        """
        Saves the DataFrame to a CSV file.
        :param df: The DataFrame to be saved.
        :param file_name: The name of the file to save.
        """
        try:
            file_path = os.path.join(folder_path,file_name)
            df.to_csv(file_path,index=True)
            self.logger.info(f"Data saved successfully to {folder_path+file_name}.")
        except Exception as e:
            self.logger.error(f"Error saving {folder_path+file_name}: {e}")

    def scrape_and_save_all(self, folder_path):
        """
        Fetches and saves all series specified in the JSON config file.
        """
        if not self.series_config:
            self.logger.error("No series found in configuration file.")
            return

        for series in self.series_config:
            series_id = series["series_id"]
            start_date = series["start_date"]
            end_date = series["end_date"]
            frequency = series["frequency"]
            file_name = series["file_name"]

            df = self.fetch_series(series_id, start_date, end_date)
            if not df.empty:

                folder_frequency_path = os.path.join(folder_path,frequency)
                if not os.path.exists(folder_frequency_path):
                    os.makedirs(folder_frequency_path)
                    self.logger.debug(f"Directory '{folder_frequency_path}' created.")

                self.save_to_csv(df, folder_frequency_path, file_name)