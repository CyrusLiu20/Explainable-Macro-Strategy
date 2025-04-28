# A very short script to manually delete unncessary data

import pandas as pd

def deduplicate_data(file_path: str):
    # Load the CSV
    df = pd.read_csv(file_path)
    original_len = len(df)
    print(f"Original number of entries: {original_len}")

    # Sort by Relevance length in descending order
    df_sorted = df.sort_values(by="Relevance", key=lambda x: x.str.len(), ascending=False)

    # Remove duplicates based on Date, Source, Title
    df_deduped = df_sorted.drop_duplicates(subset=["Date", "Source", "Title", "Summary"], keep='first')

    deleted_entries = original_len - len(df_deduped)
    return df_deduped, deleted_entries

def main():
    file_path = "Backtest/AggregatedData/AggregatedNews.csv"

    df_deduped, deleted_entries = deduplicate_data(file_path)

    # Save the cleaned data back to the original file
    df_deduped.to_csv(file_path, index=False)

    print(f"Number of entries deleted: {deleted_entries}")
    print(f"Modified number of entries: {len(df_deduped)}")

if __name__ == "__main__":
    main()
