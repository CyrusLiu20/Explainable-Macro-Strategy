import numpy as np
import pandas as pd
from scipy.stats import linregress


class Stats():

    def __init__(self, date, strategy_return, benchmark_return, position, name="Strategy"):

        self.strategy_return = strategy_return
        self.benchmark_return = benchmark_return
        self.position = position
        self.name = name

        self.signal = pd.DataFrame({
            'Strategy_Daily_Return': np.array(strategy_return),
            'Benchmark_Daily_Return': np.array(benchmark_return),
            'Position': np.array(position)
        }, index=np.array(date))

        self.benchmark = True
        self.trim = False
        self.trim_range = None

    def cagr(self, returns, rf=0.0, compounded=True, periods=252):
        total = (returns.add(1).prod(axis=0) - 1) if compounded else np.sum(total, axis=0)
        years = (returns.index[-1] - returns.index[0]).days / periods

        res = abs(total + 1.0) ** (1.0 / years) - 1

        return res

    def sharpe(self, returns, rf=0.028, periods=252, annualize=True, smart=False):
        if rf != 0 and periods is None:
            raise Exception("Must provide periods if rf != 0")

        rf_daily = np.power(1 + rf, 1.0 / periods) - 1.0
        returns_std = returns.std(ddof=1)
        res = (returns - rf_daily).mean() / returns_std

        if annualize:
            return res * np.sqrt(1 if periods is None else periods)

        return res

    def mdd(self, returns):

        return_cumulative = (1 + returns).cumprod()
        running_max = return_cumulative.cummax()
        drawdown = (running_max - return_cumulative) / running_max

        # Maximum Drawdown (MDD)
        res = drawdown.max()
        return res

    def calmar(self, returns):

        cagr_ratio = self.cagr(returns,periods=252)
        max_dd = self.mdd(returns)
        return cagr_ratio / abs(max_dd)

    def omega(self, returns, rf=0.0, required_return=0.0, periods=252):

        rf_daily = np.power(1 + rf, 1.0 / periods) - 1.0
        returns_excess = returns - rf_daily

        if periods == 1:
            return_threshold = required_return
        else:
            return_threshold = (1 + required_return) ** (1.0 / periods) - 1

        returns_less_thresh = returns_excess - return_threshold
        numer = returns_less_thresh[returns_less_thresh > 0.0].sum()
        denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum()

        return numer / denom

    def sortino(self, returns, rf=0.0, return_target=0.0, periods=252, annualize=True):

        rf_daily = np.power(1 + rf, 1.0 / periods) - 1.0
        returns_excess = returns - rf_daily

        downside = np.sqrt((returns_excess[returns_excess  < return_target] ** 2).sum() / len(returns_excess ))

        res = returns_excess.mean() / downside

        if annualize:
            return res * np.sqrt(1 if periods is None else periods)

        return res

    def trade_reversals(self, positions=None) -> int:

        # Return 0 if benchmarking is enabled and no positions are provided
        if positions is None:
            return 0

        # Calculate position changes (trade reversals)
        trade_reversals = positions.diff().abs().fillna(0)
        return int(trade_reversals.sum()/2)

    def compute_cumulative_return(self, daily_returns):

        return (1 + daily_returns).cumprod() - 1

    def yearly_pnl(self, returns):

        yearly_pnl = returns.resample('Y').sum()
        return yearly_pnl

    def win_rate(self, returns):

        wins = (returns > 0).sum()
        win_rate = wins / len(returns)
        return win_rate

    def annualized_volatility(self, returns, periods=252):

        daily_volatility = returns.std(ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(periods)
        return annualized_volatility

    def alpha(self, returns, returns_benchmark, periods=252, rf=0.028):

        rf_daily = np.power(1 + rf, 1.0 / periods) - 1.0

        excess_returns = returns - rf_daily
        benchmark_excess_returns = returns_benchmark - rf_daily

        beta = self.beta(returns, returns_benchmark)
        daily_alpha = (excess_returns.mean() - beta * benchmark_excess_returns.mean())
        annual_alpha = daily_alpha * periods

        return annual_alpha

    def beta(self, returns, returns_benchmark):

        cov_matrix = np.cov(returns, returns_benchmark)
        return cov_matrix[0, 1] / cov_matrix[1, 1]

    def r_squared(self, returns, returns_benchmark):

        _, _, r, _, _ = linregress(returns, returns_benchmark)
        return r**2

    def var(self, returns, confidence_level=0.95):

        return np.percentile(returns, (1 - confidence_level) * 100)

    def cvar(self, returns, confidence_level=0.95):

        var = self.var(returns, confidence_level)

        return returns[returns <= var].mean()

    def kurtosis(self, returns):

        return returns.kurtosis()

    def skewness(self, returns):

        return returns.skew()

    def total_return(self, returns):

        return (1 + returns).prod() - 1

    def _compute_stats(self, returns, returns_benchmark=None, positions=None):

        # aligned = pd.DataFrame({'returns': returns, 'returns_benchmark': returns_benchmark}).dropna()
        # returns = aligned['returns']
        # returns_benchmark = aligned['returns_benchmark']

        stats = {
            'CAGR': self.cagr(returns, periods=252),
            'Total Return': self.total_return(returns),
            'Sharpe Ratio': self.sharpe(returns),
            'Sortino Ratio': self.sortino(returns),
            'Calmar Ratio': self.calmar(returns),
            'Omega Ratio': self.omega(returns),
            'Alpha': self.alpha(returns, returns_benchmark) if returns_benchmark is not None else None,
            'Beta': self.beta(returns, returns_benchmark) if returns_benchmark is not None else None,
            'R Squared': self.r_squared(returns, returns_benchmark) if returns_benchmark is not None else None,
            'Maximum Drawdown': self.mdd(returns),
            'Volatility': self.annualized_volatility(returns),
            'VaR': self.var(returns),
            'CVaR': self.cvar(returns),
            'Kurtosis': self.kurtosis(returns),
            'Skewness': self.skewness(returns),
            'Win Rate': self.win_rate(returns),
            'Trade Reversals': self.trade_reversals(positions),
        }

        # If no benchmark data is provided
        stats = {key: value for key, value in stats.items() if value is not None}

        return stats

    def display_stats(self):
        returns = self.signal['Strategy_Daily_Return']
        positions = self.signal['Position']
        returns, positions = self._trim_daterange(returns), self._trim_daterange(positions)

        if self.benchmark is None:
            strategy_stats = self._compute_stats(returns=returns, positions=positions)
        else:
            returns_benchmark = self.signal['Benchmark_Daily_Return']
            returns_benchmark = self._trim_daterange(returns_benchmark)
            strategy_stats = self._compute_stats(returns=returns, returns_benchmark=returns_benchmark, positions=positions)

        # Format numbers based on size
        def format_number(value, precision=4):
            # If the value is an integer or large, display it as is
            if isinstance(value, int) or abs(value) >= 10:
                return f'{value}'.rjust(8)
            else:
                return f'{value:.{precision}f}'.rjust(8)

        # If benchmark data is available, compute benchmark stats
        if self.benchmark is not None:
            returns_benchmark = self.signal['Benchmark_Daily_Return']
            returns_benchmark = self._trim_daterange(returns_benchmark)

            benchmark_stats = self._compute_stats(returns=returns_benchmark, returns_benchmark=returns_benchmark, positions=None)

            # Calculate maximum lengths for formatting
            keys = list(strategy_stats.keys())
            max_key_length = max(len(key) for key in keys)
            max_value_length = max(len(format_number(value)) for value in strategy_stats.values())

            # Print the performance table
            print(f'{"="*10} Trade Performance {"="*10}')
            header = f'{"Metric".ljust(max_key_length)} | {self.name.rjust(max_value_length)} | {"Benchmark".rjust(max_value_length)}'
            print(header)
            print('-' * len(header))

            # Print each stat with adjusted number formatting
            for key in keys:
                strategy_value = format_number(strategy_stats[key]).rjust(max_value_length)
                benchmark_value = format_number(benchmark_stats.get(key, "N/A")) if key in benchmark_stats else "N/A".rjust(max_value_length)
                print(f'{key.ljust(max_key_length)} | {strategy_value} |  {benchmark_value}')

            print(f'{"="*10} Trade Performance {"="*10}')
        else:
            # Print strategy stats only if benchmark data is not available
            print(f'=================Trade Performance=================')
            for key, value in strategy_stats.items():
                print(f'{key} : {format_number(value)}')
            print(f'=================Trade Performance=================')


    def set_trim_range(self, trim_range):
        self.trim = True
        self.trim_range = trim_range

    def _trim_daterange(self, series):
        # Trim the series to the specified date range.
        if self.trim_range is not None:
            return series[(series.index >= self.trim_range[0]) & (series.index <= self.trim_range[1])]
        return series
    

# Example use case
if __name__ == "__main__":

    # Generate synthetic data
    n_days = 100
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    benchmark_return = np.random.normal(loc=0.0005, scale=0.01, size=n_days)
    strategy_return = benchmark_return + np.random.normal(loc=0.0002, scale=0.005, size=n_days)
    positions = np.random.choice([-1, 0, 1], size=n_days)

    stats = Stats(strategy_return=strategy_return, 
                benchmark_return=benchmark_return, 
                position=positions,
                date=dates,
                name="News Driven Strategy")

    stats.display_stats()