import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager



class ETFBacktest:
    def __init__(self, results_folder_path, backtest_names, backtest_results, price_data_path, leverage=2, start_date="2023-01-01", end_date="2024-01-01"):

        self.results_folder_path = results_folder_path
        self.backtest_names = backtest_names
        self.backest_results = backtest_results
        self.price_data_path = price_data_path
        self.leverage = leverage
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self, results_path):

        # Load and process decisions data
        decisions = pd.read_csv(results_path, parse_dates=["Date"])
        decisions.set_index("Date", inplace=True)
        
        # Truncate to date range
        start_date, end_date = self.start_date, self.end_date
        decisions = decisions.loc[start_date:end_date]
        decisions = decisions[["Decision"]]
        decisions.reset_index(inplace=True)
        
        # Apply leverage
        decisions["Decision"] = decisions["Decision"].apply(
            lambda x: self.leverage if x > 0 else (-self.leverage if x < 0 else 0)
        )
        
        # Compute median decision for general multi-agent backtest results
        decisions = decisions.groupby("Date", as_index=False)["Decision"].median()

        # Load and process price data
        price_data = pd.read_csv(self.price_data_path)
        price_data["Date"] = pd.to_datetime(price_data["Date"])
        price_data = price_data[(price_data["Date"] >= start_date) & 
                                        (price_data["Date"] <= end_date)]
        price_data = price_data.sort_values("Date")
        
        return decisions, price_data


    def prepare_backtest_data(self, decisions, price_data):
        """
        Prepare the data for backtesting by merging decisions with price data.
        """
        if price_data is None or decisions is None:
            raise ValueError("Please load data first using load_data()")
            
        # Calculate returns
        adj_close = price_data["Price Close"]
        returns = adj_close.pct_change()
        
        # Create base dataframe
        data = pd.DataFrame({
            "Date": price_data["Date"],
            "Price": adj_close,
            "Return": returns
        })
        
        # Ensure Date format matches for joining
        decisions["Date"] = pd.to_datetime(decisions["Date"])
        
        # Merge on Date
        data = pd.merge(data, decisions, on="Date", how="left")
        data = data.sort_values("Date")
        
        # Shift the position forward by one day to avoid look-ahead bias
        data["Position"] = data["Decision"].shift(1)
        data["Strategy Return"] = data["Position"] * data["Return"]
        data = data.dropna(subset=["Strategy Return"])
        
        # Calculate cumulative returns
        data["Cumulative Return"] = (1 + data["Return"]).cumprod()
        data["Cumulative Strategy Return"] = (1 + data["Strategy Return"]).cumprod()

        return data

    def run_backtest(self):

        data_list = []
        for backest_result in self.backest_results:

            results_path = os.path.join(self.results_folder_path, backest_result)
            print(results_path)
            decisions, price_data = self.load_data(results_path=results_path)
            data = self.prepare_backtest_data(decisions=decisions, price_data=price_data)
            data_list.append(data)

        self.plot_cumulative_returns(data_list=data_list)

    # Plot the price series of the underlying asset.
    def plot_price_series(self, price_data, filename="price_series.png"):

        if price_data is None:
            raise ValueError("Please load data first using load_data()")
            
        plt.figure(figsize=(12, 6))
        plt.plot(price_data["Date"], price_data["Price Close"], 
                label="Close Price", linewidth=3)
        
        plt.title("iShares 7-10 Year Treasury Bond ETF (IEF) Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = os.path.join(self.results_folder_path, filename)
        plt.savefig(filename)
        plt.close()
        
    # Plot the cumulative returns of the strategy vs the underlying asset.
    def plot_cumulative_returns(self, data_list, filename="strategy_return.png"):
        plt.figure(figsize=(12, 6))
        
        plt.plot(data_list[0]["Date"], data_list[0]["Cumulative Return"], 
                label="IEF asset data", linewidth=3)

        for i, data in enumerate(data_list):
            # Create unique labels for each curve
            label_strategy = f"{self.backtest_names[i]}"
            
            plt.plot(data["Date"], data["Cumulative Strategy Return"], 
                    label=label_strategy, linewidth=3)
        

        plt.title("Cumulative Return vs Cumulative Strategy Return", 
                fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=14, fontweight='bold')
        plt.ylabel("Cumulative Return", fontsize=14, fontweight='bold')
        

        plt.legend(fontsize=12, prop=font_manager.FontProperties(weight='bold'))
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(self.results_folder_path, filename)
        plt.savefig(filename)
        plt.close()