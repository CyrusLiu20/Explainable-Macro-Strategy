from DataPipeline.CensusScraper import CensusDataScraper
from DataPipeline.FredScraper import FredDataScraper
from DataPipeline.AlphaVantageScraper import AlphaVantageScraper

from DataPipeline.Data.MacroIndicators.IndicatorMapping import write_mapping
import DataPipeline.Config.SplitTime as splittime

from pathlib import Path

if __name__ == "__main__":
    
    data_root = Path("DataPipeline/Data")
    config_root = Path("DataPipeline/Config")
    log_root = Path("DataPipeline/LogFiles")
    time_range = (2000, 2024)
    
    # Write mapping for macro indicators
    write_mapping(folder_path=data_root / "MacroIndicators")
    
    # Initialize and run the FRED data scraper
    fred_scraper = FredDataScraper(
        config_file=config_root / "fred_config.json",
        log_file=log_root / "fred_scraper.log"
    )
    fred_scraper.scrape_and_save_all(data_root / "MacroIndicators")
    
    # Census Data Scraper
    # census_scraper = CensusDataScraper(
    #     config_file=config_root / "census_config.json",
    #     log_file=log_root / "census_scraper.log"
    # )
    # census_scraper.scrape_datasets(data_root, time_range)
    
    # Process AlphaVantage config
    # input_file = config_root / "alphavantage_config_orig.json"
    # output_file = config_root / "alphavantage_config.json"
    # selected_types = ["economy_macro"]
    # date_range = ("20220101T0130", "20250220T0130")
    # splittime.process_config(input_file, output_file, selected_types, date_range, months=2)
    
    # AlphaVantage Data Scraper
    # alphavantage_scraper = AlphaVantageScraper(
    #     config_file=config_root / "alphavantage_config.json",
    #     log_file=log_root / "alphavantage_scraper.log"
    # )
    # alphavantage_scraper.scrape_and_save_all(data_root / "MacroNews")


    