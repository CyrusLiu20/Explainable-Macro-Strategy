import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utilities.Logger import logger

class DataProcessor:
    def __init__(self, folder_path, output_folder_path, log_file, 
                 date_column=None, date_format=None, columns_to_keep=None, rename_columns=None):
        """
        General data processor.

        Parameters:
        - folder_path: Path to input CSVs
        - output_folder_path: Path to save processed data
        - log_file: Log file path
        - date_column: Name of the datetime column (if applicable)
        - date_format: Expected format for datetime parsing
        - columns_to_keep: List of columns to retain
        - rename_columns: Dictionary for renaming columns {old_name: new_name}
        """
        self.folder_path = folder_path
        self.output_folder_path = output_folder_path
        self.log_file = log_file
        self.date_column = date_column
        self.date_format = date_format
        self.columns_to_keep = columns_to_keep
        self.rename_columns = rename_columns or {}
        self.log = logger(self.__class__.__name__, log_file)

    def get_csv_files(self, prefixes=None):
        """Retrieve all CSV files, optionally filtered by a list of prefixes."""
        if prefixes is None:
            prefixes = [""]  # Default to all CSVs if no prefixes provided

        csv_files = [
            f for f in os.listdir(self.folder_path)
            if f.endswith(".csv") and any(f.startswith(prefix) for prefix in prefixes)
        ]

        if not csv_files:
            self.log.warning(f"No CSV files found in {self.folder_path} with prefixes {prefixes}.")

        return csv_files

    def read_and_concatenate_csvs(self, csv_files):
        """Read and concatenate CSV files."""
        self.log.info(f"Found {len(csv_files)} CSV files.")
        return pd.concat(
            (pd.read_csv(os.path.join(self.folder_path, file)) for file in csv_files),
            ignore_index=True
        )
    
    def read_and_concatenate_csvs_horizontally(self, csv_files, merge_on="Date"):
        """Read and concatenate CSV files horizontally (merge on a common column)."""
        self.log.info(f"Found {len(csv_files)} CSV files for horizontal merging.")
        
        dfs = [pd.read_csv(os.path.join(self.folder_path, file)) for file in csv_files]
        
        if not dfs:
            self.log.warning("No CSV files loaded for horizontal merging.")
            return None

        # Merge all DataFrames on the specified column
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = merged_df.merge(df, on=merge_on, how="outer", suffixes=("", "_dup"))
        
        # Log any duplicate column names after merging
        duplicate_cols = [col for col in merged_df.columns if col.endswith("_dup")]
        if duplicate_cols:
            self.log.warning(f"Duplicate columns detected after merging: {duplicate_cols}")

        return merged_df


    def remove_duplicates(self, df, macro_news=False):
        """Remove duplicate rows, with an option to filter by specific columns if macro_news is True."""
        initial_count = len(df)
        
        if macro_news:
            required_cols = {"summary", "time_published", "title", "source"}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                self.log.error(f"Missing columns in DataFrame: {missing_cols}")
                return df  # Return the original DataFrame without modification
            
            df = df.drop_duplicates(subset=required_cols)
        else:
            df = df.drop_duplicates()
        
        df = df.reset_index(drop=True)  # Reset index to maintain consistency
        duplicates_removed = initial_count - len(df)
        self.log.info(f"Number of duplicate entries removed: {duplicates_removed}")
        
        return df

    def handle_missing_dates(self, df):
        """Convert the specified column to datetime and handle missing dates."""
        if self.date_column and self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format, errors="coerce")
            missing_dates = df[self.date_column].isna().sum()
            if missing_dates > 0:
                self.log.warning(f"Number of entries with missing or unparseable dates in '{self.date_column}': {missing_dates}")
        return df

    def process_columns(self, df):
        """Select relevant columns and rename them."""
        if self.columns_to_keep:
            df = df[self.columns_to_keep]
        df = df.rename(columns=self.rename_columns)
        df.columns = df.columns.str.capitalize() # Capitalise First Letter of all column names
        return df

    def find_missing_date_ranges(self, df):

        if self.rename_columns is not None:
            return

        """Identify missing date periods in the dataset."""
        if self.rename_columns[next(iter(self.rename_columns))] in df.columns:
            df["date_only"] = df[self.rename_columns[next(iter(self.rename_columns))]].dt.date  # Extract only the date
            min_date = df["date_only"].min()
            max_date = df["date_only"].max()
            full_date_range = pd.date_range(start=min_date, end=max_date, freq="D").date

            present_dates = set(df["date_only"])
            missing_dates = sorted(set(full_date_range) - present_dates)

            if missing_dates:
                missing_periods = []
                start_date = missing_dates[0]

                for i in range(1, len(missing_dates)):
                    if (missing_dates[i] - missing_dates[i - 1]).days > 1:
                        end_date = missing_dates[i - 1]
                        missing_periods.append((start_date, end_date))
                        start_date = missing_dates[i]

                missing_periods.append((start_date, missing_dates[-1]))  # Add last period

                self.log.warning(f"Found {len(missing_periods)} missing date ranges between {min_date} and {max_date}.")
                for start, end in missing_periods:
                    self.log.warning(f"Missing period: {start} to {end}")
            else:
                self.log.info("No missing date periods found in the dataset.")

    def save_processed_data(self, df, filename="processed_data.csv"):
        """Save the processed data to CSV."""
        os.makedirs(self.output_folder_path, exist_ok=True)
        output_file_path = os.path.join(self.output_folder_path, filename)
        df.to_csv(output_file_path, index=False)
        self.log.info(f"Processed data saved to {output_file_path}")

