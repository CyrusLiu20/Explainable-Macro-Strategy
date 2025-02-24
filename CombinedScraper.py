from DataPipeline.CensusScraper import CensusDataScraper
from DataPipeline.FredScraper import FredDataScraper
from DataPipeline.AlphaVantageScraper import AlphaVantageScraper

from DataPipeline.Data.MacroIndicators.IndicatorMapping import write_mapping
import DataPipeline.Config.SplitTime as splittime
from DataPipeline.MacroProcessor import NewsDataProcessor, IndicatorDataProcessor

import argparse
from pathlib import Path

def main(scrape: bool, process: bool):
    data_root = Path("DataPipeline/Data")
    config_root = Path("DataPipeline/Config")
    log_root = Path("DataPipeline/LogFiles")
    time_range = (2000, 2024)

    # Write mapping for macro indicators
    write_mapping(folder_path=data_root / "MacroIndicators")

    if scrape:

        # Initialize and run the FRED data scraper
        fred_scraper = FredDataScraper(
            config_file=config_root / "fred_config.json",
            log_file=log_root / "fred_scraper.log"
        )
        fred_scraper.scrape_and_save_all(data_root / "MacroIndicators")

        # Process AlphaVantage config
        input_file = config_root / "alphavantage_config_orig.json"
        output_file = config_root / "alphavantage_config.json"
        selected_types = ["economy_monetary"]
        date_range = ("20220101T0130", "20250220T0130")
        splittime.process_config(input_file, output_file, selected_types=selected_types, date_range=date_range, months=1)

        # AlphaVantage Data Scraper
        alphavantage_scraper = AlphaVantageScraper(
            config_file=config_root / "alphavantage_config.json",
            log_file=log_root / "alphavantage_scraper.log"
        )
        alphavantage_scraper.scrape_and_save_all(data_root / "MacroNews")

    if process:
        # Process news data
        news_processor = NewsDataProcessor(
            folder_path=data_root / "MacroNews",
            output_folder_path=data_root / "ProcessedData",
            log_file=log_root / "news_data_processor.log"
        )
        news_processor.process_data()

        # Process macro indicators data
        frequencies = ["Daily", "Weekly", "Monthly", "Quarterly"]
        for frequency in frequencies:
            indicator_processor = IndicatorDataProcessor(
                frequency=frequency,
                folder_path=data_root / "MacroIndicators",
                output_folder_path=data_root / "ProcessedData",
                log_file=log_root / "indicators_data_processor.log"
            )
            indicator_processor.process_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data scraping and/or processing")
    parser.add_argument("--scrape", action="store_true", default=False, help="Scrape data from sources")
    parser.add_argument("--process", action="store_true", default=False, help="Process scraped data")
    args = parser.parse_args()

    main(scrape=args.scrape, process=args.process)



    