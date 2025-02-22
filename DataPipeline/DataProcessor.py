import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utilities.Logger import logger

class FiscalDataProcessor:
    def __init__(self, folder_path, output_folder_path, log_file):
        self.folder_path = folder_path
        self.output_folder_path = output_folder_path
        self.log_file = log_file
        self.log = logger("FiscalDataProcessor", log_file)
    
    def get_fiscal_files(self):
        """Retrieve all fiscal CSV files from the given folder."""
        csv_files = [f for f in os.listdir(self.folder_path) if f.startswith("fiscal") and f.endswith(".csv")]
        if not csv_files:
            self.log.warning("No fiscal CSV files found in the directory.")
        return csv_files

    def read_and_concatenate_csvs(self, csv_files):
        """Read and concatenate all fiscal CSV files into one DataFrame."""
        self.log.info(f"Found {len(csv_files)} fiscal CSV files.")
        return pd.concat(
            (pd.read_csv(os.path.join(self.folder_path, file)) for file in csv_files),
            ignore_index=True
        )

    def remove_duplicates(self, df):
        """Remove duplicate rows and log the count."""
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        self.log.info(f"Total records after removing duplicates: {len(df)}")
        self.log.info(f"Number of duplicate entries removed: {duplicates_removed}")
        return df

    def handle_missing_dates(self, df):
        """Convert 'time_published' to datetime and handle missing dates."""
        df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
        missing_dates = df["time_published"].isna().sum()
        if missing_dates > 0:
            self.log.warning(f"Number of entries with missing or unparseable dates: {missing_dates}")
        return df

    def process_columns(self, df):
        """Select relevant columns and rename 'time_published' to 'date'."""
        df = df[["time_published", "title", "summary", "source", "topics"]].rename(columns={"time_published": "date"})
        return df

    def save_processed_data(self, df):
        """Ensure output folder exists and save the processed data to CSV."""
        os.makedirs(self.output_folder_path, exist_ok=True)
        output_file_path = os.path.join(self.output_folder_path, "fiscal_data.csv")
        df.to_csv(output_file_path, index=False)
        self.log.info(f"Processed data saved to {output_file_path}")
    
    def process_fiscal_data(self):
        """Main method to process fiscal data."""
        csv_files = self.get_fiscal_files()
        if not csv_files:
            return
        
        # Read and concatenate the CSV files
        df = self.read_and_concatenate_csvs(csv_files)
        
        # Process the DataFrame
        self.log.info(f"Total records before removing duplicates: {len(df)}")
        df = self.remove_duplicates(df)
        df = self.handle_missing_dates(df)
        df = self.process_columns(df)
        
        # Save the processed data
        self.save_processed_data(df)
        
        return df

# Define file paths
folder_path = "DataPipeline/Data/MacroNews"
output_folder_path = "DataPipeline/Data/ProcessedData"
log_file = "Logs/DataProcessor.log"

# Instantiate and process fiscal data
processor = FiscalDataProcessor(folder_path, output_folder_path, log_file)
processed_data = processor.process_fiscal_data()
