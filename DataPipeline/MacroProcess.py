import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utilities.Logger import logger
from DataPipeline.DataProcessor import DataProcessor

class NewsDataProcessor(DataProcessor):
    def __init__(self, folder_path, output_folder_path, log_file):
        """Initialize the FiscalDataProcessor with specific column configurations."""
        super().__init__(
            folder_path=folder_path,
            output_folder_path=output_folder_path,
            log_file=log_file,
            date_column="time_published",
            date_format="%Y%m%dT%H%M%S",
            columns_to_keep=["time_published", "title", "summary", "source", "topics"],
            rename_columns={"time_published": "date"}
        )

        self.prefixes = ["fiscal", "economic", "market"]
        self.output_file = "MacroNews"

    def process_data(self):
        """Run the general data processing pipeline for fiscal data."""
        return super().process_data(file_name=self.output_file, prefix=self.prefixes)
    
class IndicatorDataProcessor(DataProcessor):
    def __init__(self, folder_path, output_folder_path, log_file):
        """Initialize the FiscalDataProcessor with specific column configurations."""
        super().__init__(
            folder_path=folder_path,
            output_folder_path=output_folder_path,
            log_file=log_file,
            date_column="time_published",
            date_format="%Y%m%dT%H%M%S",
            columns_to_keep=["time_published", "title", "summary", "source", "topics"],
            rename_columns={"time_published": "date"}
        )

        self.prefixes = ["indicator"]
        self.output_file = "MacroIndicator"

    def process_data(self):
        """Run the general data processing pipeline for Indicator data."""
        return super().process_data(file_name=self.output_file, prefix=self.prefixes)



# Define file paths
folder_path = "DataPipeline/Data/MacroNews"
output_folder_path = "DataPipeline/Data/ProcessedData"
log_file = "Logs/DataProcessor.log"

# Instantiate and process fiscal data
processor = NewsDataProcessor(folder_path, output_folder_path, log_file)
processed_data = processor.process_data()
