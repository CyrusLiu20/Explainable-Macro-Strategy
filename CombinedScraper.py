from DataPipeline.CensusScraper import CensusDataScraper
from DataPipeline.FredScraper import FredDataScraper
from DataPipeline.AlphaVantageScraper import AlphaVantageScraper

import DataPipeline.Config.SplitTime as splittime


if __name__ == "__main__":

    folder_path = "DataPipeline/Data/"
    time_range = (2000, 2024)

    # # US Census Bureau Data Scraper
    # census_scraper = CensusDataScraper(config_file="DataPipeline/Config/census_config.json",
    #                                    log_file="DataPipeline/LogFiles/census_scraper.log")
    # census_scraper.scrape_datasets(folder_path, time_range)

    # Federal Reserve Bank of St Louis Economical Data Scraper
    folder_path = "DataPipeline/Data/MacroIndicators/"
    fred_scraper = FredDataScraper(config_file="DataPipeline/Config/fred_config.json",
                                   log_file="DataPipeline/LogFiles/fred_scraper.log")
    fred_scraper.scrape_and_save_all(folder_path)

    # # Define input and output files
    # input_file = "DataPipeline/Config/alphavantage_config_orig.json"
    # output_file = "DataPipeline/Config/alphavantage_config.json"
    # selected_types = ["economy_macro"]
    # date_range = ("20220101T0130", "20250220T0130")


    # splittime.process_config(input_file, output_file, selected_types=selected_types, date_range=date_range, months=2)

    # folder_path = "DataPipeline/Data/MacroNews/"
    # alphavantage_scraper = AlphaVantageScraper(config_file="DataPipeline/Config/alphavantage_config.json",
    #                                            log_file="DataPipeline/LogFiles/alphavantage_scraper.log")
    # alphavantage_scraper.scrape_and_save_all(folder_path)