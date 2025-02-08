from DataPipeline.CensusScraper import CensusDataScraper
from DataPipeline.FredScraper import FredDataScraper

if __name__ == "__main__":

    folder_path = "DataPipeline/Data/"
    time_range = (2000, 2024)

    # US Census Bureau Data Scraper
    census_scraper = CensusDataScraper(config_file="DataPipeline/Config/census_config.json",
                                       log_file="DataPipeline/LogFiles/census_scraper.log")
    census_scraper.scrape_datasets(folder_path, time_range)

    # Federal Reserve Bank of St Louis Economical Data Scraper
    fred_scraper = FredDataScraper(config_file="DataPipeline/Config/fred_config.json",
                                   log_file="DataPipeline/LogFiles/fred_scraper.log")
    fred_scraper.scrape_and_save_all(folder_path)

