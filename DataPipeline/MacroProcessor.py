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
            rename_columns={"time_published": "Date"}
        )

        self.prefixes = ["fiscal", "monetary", "macro"]
        self.output_file = "MacroNews"

    def process_data(self, prefixes=""):
        """General pipeline to process data."""
        csv_files = super().get_csv_files(prefixes=self.prefixes)
        if not csv_files:
            return

        df = super().read_and_concatenate_csvs(csv_files)
        df = super().remove_duplicates(df)
        df = super().handle_missing_dates(df)
        df = super().process_columns(df)
        super().save_processed_data(df, filename=f"{self.output_file}.csv")
        super().find_missing_date_ranges(df)
        
        return df

class IndicatorDataProcessor(DataProcessor):
    def __init__(self, frequency, folder_path, output_folder_path, log_file):

        self.frequency = frequency
        self.prefixes = ["indicator"]
        self.output_file = "MacroIndicator"
        self.folder_path = os.path.join(folder_path,frequency)

        """Initialize the FiscalDataProcessor with specific column configurations."""
        super().__init__(
            folder_path=self.folder_path,
            output_folder_path=output_folder_path,
            log_file=log_file,
            date_column="Date",
            date_format="%Y%m%d",
            columns_to_keep=None,
            rename_columns=None
        )

    def process_data(self, prefixes=""):
        """General pipeline to process data."""
        csv_files = super().get_csv_files(prefixes=self.prefixes)
        if not csv_files:
            return

        df = super().read_and_concatenate_csvs_horizontally(csv_files, merge_on=self.date_column)
        df = super().remove_duplicates(df)
        df = df.dropna(subset=df.columns.difference(['date']), how='all')
        super().save_processed_data(df, filename=f"{self.output_file}"+f"{self.frequency}.csv")
        super().find_missing_date_ranges(df)
        
        return df



if __name__ == "__main__":
    # Define file paths
    folder_path = "DataPipeline/Data/MacroNews"
    output_folder_path = "DataPipeline/Data/ProcessedData"
    log_file = "Logs/news_data_processor.log"

    # Instantiate and process news data
    news_processor = NewsDataProcessor(folder_path, output_folder_path, log_file)
    processed_data = news_processor.process_data()



    # Define file paths
    folder_path = "DataPipeline/Data/MacroIndicators"
    output_folder_path = "DataPipeline/Data/ProcessedData"
    log_file = "Logs/indicators_data_processor.log"
    frequencies = ["Daily", "Weekly", "Monthly", "Quarterly"]

    for frequency in frequencies:
        # Instantiate and process indicator data
        indicator_processor = IndicatorDataProcessor(frequency=frequency, folder_path=folder_path, output_folder_path=output_folder_path, log_file=log_file)
        processed_data = indicator_processor.process_data()
