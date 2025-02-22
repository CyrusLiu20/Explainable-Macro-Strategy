import json
import os
from datetime import datetime, timedelta
from Utilities.Logger import logger

# Initialize logger
log = logger("split_config", "logs/split_config.log")

def split_time_range(time_from, time_to, months=3):
    """Split the time range into three-month intervals."""
    date_format = "%Y%m%dT%H%M"
    start = datetime.strptime(time_from, date_format)
    end = datetime.strptime(time_to, date_format)
    
    periods = []
    while start < end:
        next_period = start + timedelta(days=months * 30)  # Approximate months with 30 days
        if next_period > end:
            next_period = end
        periods.append((start.strftime(date_format), next_period.strftime(date_format)))
        start = next_period
    
    return periods

def process_config(input_file, output_file, selected_types=None, date_range=None):
    """Process the configuration file, filtering by type and date range."""
    if not os.path.exists(input_file):
        log.error(f"Input file {input_file} not found.")
        return
    
    with open(input_file, "r") as f:
        config = json.load(f)
    
    new_config = []
    for entry in config:
        if selected_types and entry["topic"] not in selected_types:
            continue
        
        file_name = entry["file_name"].replace(".csv", "")  # Remove .csv from filename
        time_from = entry["time_from"]
        time_to = entry["time_to"]
        
        # Override time range if date_range is specified
        if date_range:
            time_from, time_to = date_range
        
        time_splits = split_time_range(time_from, time_to, months=3)
        
        for i, (start, end) in enumerate(time_splits):
            new_entry = entry.copy()
            new_entry["time_from"] = start
            new_entry["time_to"] = end
            new_entry["file_name"] = f"{file_name}_{start}_{end}.csv"
            new_config.append(new_entry)
    
    with open(output_file, "w") as f:
        json.dump(new_config, f, indent=2)
    
    log.info(f"New configuration saved to {output_file}")
    log.info(f"Total number of chunks: {len(new_config)}")
