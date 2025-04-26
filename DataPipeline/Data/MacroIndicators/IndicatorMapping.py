import os
import csv
from Utilities.Logger import logger

def write_mapping(folder_path=None, output_filename="indicator_mapping.csv"):

    header_row = ["Series ID", "Renamed Series", "Delay (Days)", "Trade Frequency"]
    series_data = [
        ("SP500",           "S&P 500",                      0,      "Daily"),
        ("GDP",             "US GDP",                       30,     "Quarterly"),
        ("CPIAUCSL",        "US Urban CPI",                 14,     "Monthly"),
        ("UNRATE",          "US Unemployment Rate",         7,      "Monthly"),
        ("MICH",            "UoM Inflation Expect",         21,     "Monthly"),
        ("BOPGSTB",         "US Trade Balance",             60,     "Quarterly"),
        ("WHLSLRIMSA",      "Wholesale Inventories",        28,     "Monthly"),
        ("RSXFS",           "Advanced Retail Sales",        14,     "Monthly"),
        ("BUSINV",          "Total Business Inventories",   28,     "Monthly"),
        ("TCU",             "Capacity Utilization",         21,     "Monthly"),
        ("UMCSENT",         "UoM Consumer Sentiment",       21,     "Monthly"),
        ("DAUTOSAAR",       "Domestic Vehicle Sales",       21,     "Monthly"),
        ("DGORDER",         "Durable Goods Orders",         21,     "Monthly"),
        ("IR",              "Import Price Index",           14,     "Monthly"),
        ("ICSA",            "Initial Jobless Claims",       2,      "Weekly"),
        ("PAYEMS",          "Total Nonfarm Payrolls",       7,      "Monthly"),
        ("FEDFUNDS",        "Federal Funds Rates",          30,     "Monthly"),
        ("DGS10",           "10Y Treasury Yield",           0,      "Daily"),
        ("DGS5",            "5Y Treasury Yield",            0,      "Daily"),
        ("DGS2",            "2Y Treasury Yield",            0,      "Daily"),
        ("DGS30",           "30Y Treasury Yield",           0,      "Daily"),
        ("A939RX0Q048SBEA", "Real GDP per Capita",          30,     "Quarterly"),
        ("GDPC1",           "Real GDP",                     30,     "Quarterly"),
        ("PPIACO",          "PPI",                          14,     "Monthly"),
        ("HNFSEPUSSA",      "New Home Sales",               28,     "Monthly"),
        ("DEXUSEU",         "EUR/USD",                      0,      "Daily"),
        ("DEXJPUS",         "USD/JPY",                      0,      "Daily"),
        ("DEXUSUK",         "GBP/USD",                      0,      "Daily")
    ]

    log = logger(name="IndicatorMapper", log_file="Logs/scraper.log")
    file_path = os.path.join(folder_path,output_filename)
    with open(file_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header_row)
        for row in series_data:
            writer.writerow([row[0], row[1], row[2], row[3]])
    log.info(f"CSV file '{file_path}' has been written successfully.")

if __name__ == "__main__":
    write_mapping()
