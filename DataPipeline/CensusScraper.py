import os
import requests
import pandas as pd
import json
from Utilities.Logger import logger  # Import the custom logger

class CensusDataScraper:
    def __init__(self, api_key=None, config_file="datasets_config.json", log_file="census_scraper.log"):
        """
        Initialize the scraper with an API key and a configuration file for datasets.
        :param api_key: API key for Census API.
        :param config_file: Path to the JSON configuration file for datasets.
        :param log_file: Path to the log file.
        """
        self.logger = logger("CensusDataScraper", log_file)  # Initialize logger
        self.api_key = api_key or os.getenv('CENSUS_API_KEY')
        self.config_file = config_file
        self.datasets = self.load_datasets_config()

    def load_datasets_config(self):
        """
        Load the datasets configuration from a JSON file.
        :return: List of datasets.
        """
        try:
            with open(self.config_file, 'r') as f:
                datasets = json.load(f)
            self.logger.info(f"Successfully loaded dataset configuration from {self.config_file}.")
            return datasets
        except Exception as e:
            self.logger.error(f"Error loading dataset configuration: {e}")
            return []

    def construct_params(self, time_range, dataset_specific_params):
        """
        Construct the request parameters.
        :param time_range: Time range for the dataset.
        :param dataset_specific_params: Dictionary containing specific dataset parameters.
        :return: Constructed parameters.
        """
        url = dataset_specific_params["url"]
        params_dict = {
            "get": dataset_specific_params["get"],
            "time": f'from {time_range[0]} to {time_range[1]}',
            "for": dataset_specific_params["geo_level"],
            "key": self.api_key
        }
        self.logger.debug(f"Constructed request parameters for {url}: {params_dict}")
        return url, params_dict

    def download_dataset(self, url, params, folder_path, file_name):
        """
        Download the dataset and save it as a CSV file.
        :param url: The URL for the API request.
        :param params: Request parameters.
        :param folder_path: Folder path where the file will be saved.
        :param file_name: Name of the file to save.
        """
        self.logger.info(f"Requesting data from {url} for {file_name}...")
        response = requests.get(url, params=params)

        if response.status_code == 200:
            try:
                data = response.json()
                df = pd.DataFrame(data[1:], columns=data[0])
                output_path = os.path.join(folder_path, file_name)
                df.to_csv(output_path, index=True)
                self.logger.info(f"Data saved successfully to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to process response JSON for {file_name}: {e}")
        else:
            self.logger.error(f"Error {response.status_code} while requesting {file_name}: {response.text}")

    def scrape_datasets(self, folder_path, time_range):
        """
        Scrape and save datasets based on the configurations loaded from the JSON file.
        :param folder_path: Folder where the CSV files will be saved.
        :param time_range: Tuple specifying the time range (start year, end year).
        """
        self.logger.info(f"Starting US Census Bureau dataset scraping")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            self.logger.debug(f"Directory '{folder_path}' created.")

        for dataset in self.datasets:
            try:
                url, params = self.construct_params(time_range, dataset)
                self.download_dataset(url, params, folder_path, dataset["file_name"])
            except Exception as e:
                self.logger.error(f"Failed to scrape dataset {dataset['file_name']}: {e}")

        self.logger.info("US Census Bureau Dataset scraping completed.")
