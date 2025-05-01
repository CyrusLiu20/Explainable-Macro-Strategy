import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from Utilities.Logger import logger

class BondBacktest:
    def __init__(self, start_date, end_date, initial_cash=1000000):
        """
        Initialize the bond backtest system
        
        Parameters:
        - start_date: Starting date for the simulation
        - end_date: Ending date for the simulation
        - initial_cash: Initial cash position
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.dates = pd.date_range(start=self.start_date, end=self.end_date)
        
        # Initialize positions and tracking
        self.initial_cash = initial_cash
        self.cash_position = pd.Series(initial_cash, index=self.dates, dtype=float)
        self.bond_position = pd.Series(0, index=self.dates, dtype=float)
        self.bond_holdings = {}  # {bond_id: {'quantity': qty, 'price': price}}
        self.bond_values = pd.Series(0, index=self.dates, dtype=float)
        self.transactions = []
        
        # Results tracking
        self.daily_pnl = pd.Series(0, index=self.dates, dtype=float)
        self.cumulative_pnl = pd.Series(0, index=self.dates, dtype=float)
        
        # Market data
        self.yields = None
        self.bond_prices = {}

        # Trade history
        self.trades = {}
    
        self.log = logger(name="BondBacktest", log_file=f"Logs/backtest.log")

    def load_market_data(self, yields_data):
        """
        Load market yield data
        
        Parameters:
        - yields_data: DataFrame with dates as index and yields as columns for different tenors
        """
        self.yields = yields_data.reindex(self.dates, method='ffill')
    
    def calculate_bond_price(self, par_value, coupon_rate, maturity_years, yield_rate, days_per_year=365):
        """
        Calculate clean price of a bond using yield
        
        Parameters:
        - par_value: Bond's face value
        - coupon_rate: Annual coupon rate (decimal)
        - maturity_years: Years to maturity
        - yield_rate: Current yield rate (decimal)
        - days_per_year: Days per year for calculation
        
        Returns:
        - Clean price of the bond
        """
        if yield_rate == coupon_rate:
            return par_value
        
        frequency = 2  # Semi-annual coupons
        periods = maturity_years * frequency
        rate_per_period = yield_rate / frequency
        coupon_per_period = (par_value * coupon_rate) / frequency
        
        # Present value of coupons
        pv_coupons = coupon_per_period * (1 - (1 + rate_per_period) ** -periods) / rate_per_period
        
        # Present value of principal
        pv_principal = par_value / ((1 + rate_per_period) ** periods)
        
        # Clean price
        price = pv_coupons + pv_principal
        
        return price
    
    def update_bond_prices(self, date):
        """
        Update prices for all bonds in the portfolio based on current yields
        
        Parameters:
        - date: Current date for price update
        """
        if self.yields is None:
            raise ValueError("Market yield data not loaded. Call load_market_data first.")
        
        current_yields = self.yields.loc[date]
        
        for bond_id, details in self.bond_holdings.items():
            tenor = details['tenor']
            coupon = details['coupon']
            remaining_years = details['maturity_date'] - date
            remaining_years = remaining_years.days / 365
            
            if remaining_years <= 0:
                # Bond matured
                price = 100  # Par value
            else:
                # Interpolate yield based on tenor
                if tenor in current_yields:
                    yield_rate = current_yields[tenor]
                else:
                    # Simple linear interpolation between closest tenors
                    avail_tenors = [float(t) for t in current_yields.index if isinstance(t, (int, float))]
                    if not avail_tenors:
                        raise ValueError(f"No valid tenors found in yield data for date {date}")
                    
                    lower_tenor = max([t for t in avail_tenors if t <= tenor], default=min(avail_tenors))
                    upper_tenor = min([t for t in avail_tenors if t >= tenor], default=max(avail_tenors))
                    
                    if lower_tenor == upper_tenor:
                        yield_rate = current_yields[lower_tenor]
                    else:
                        lower_yield = current_yields[lower_tenor]
                        upper_yield = current_yields[upper_tenor]
                        yield_rate = lower_yield + (tenor - lower_tenor) * (upper_yield - lower_yield) / (upper_tenor - lower_tenor)
                
                price = self.calculate_bond_price(100, coupon, remaining_years, yield_rate)
            
            self.bond_prices[bond_id] = price
    
    def execute_trade(self, date, bond_id, quantity, price=None, tenor=5, coupon=0.03):
        """
        Execute a bond trade
        
        Parameters:
        - date: Trade date
        - bond_id: Identifier for the bond
        - quantity: Number of bonds (positive for buy, negative for sell)
        - price: Price per bond (if None, will use market price)
        - tenor: Bond tenor in years (for new bonds)
        - coupon: Bond coupon rate (for new bonds)
        """
        date = pd.to_datetime(date)
        
        # If the bond is new, add it to holdings
        if bond_id not in self.bond_holdings and quantity > 0:
            maturity_date = date + timedelta(days=int(tenor * 365))
            self.bond_holdings[bond_id] = {
                'quantity': 0,
                'tenor': tenor,
                'coupon': coupon,
                'issue_date': date,
                'maturity_date': maturity_date
            }
        
        # Update bond prices
        self.update_bond_prices(date)
        
        # Use market price if not specified
        if price is None:
            if bond_id in self.bond_prices:
                price = self.bond_prices[bond_id]
            else:
                # Calculate price for new bond
                price = self.calculate_bond_price(
                    100, coupon, tenor, self.yields.loc[date, tenor]
                )
        
        # Calculate trade cash flow (negative for buys, positive for sells)
        trade_value = -quantity * price
        
        # Update positions
        if bond_id in self.bond_holdings:
            self.bond_holdings[bond_id]['quantity'] += quantity
            
            # Remove bond if quantity is 0
            if self.bond_holdings[bond_id]['quantity'] == 0:
                del self.bond_holdings[bond_id]
        
        # Update cash position
        idx = self.cash_position.index.get_indexer([date], method='ffill')[0]
        self.cash_position.iloc[idx:] += trade_value
        
        # Record transaction
        self.transactions.append({
            'date': date,
            'bond_id': bond_id,
            'quantity': quantity,
            'price': price,
            'value': -trade_value,
            'type': 'buy' if quantity > 0 else 'sell'
        })
        
        # Recalculate bond position value
        self._calculate_positions(date)
    
    def process_bond_maturity(self, date, bond_id):
        """
        Process the maturity of a bond
        
        Parameters:
        - date: Maturity date
        - bond_id: Identifier for the maturing bond
        """
        if bond_id not in self.bond_holdings:
            return
        
        bond = self.bond_holdings[bond_id]
        if bond['maturity_date'] == date:
            # Return principal at par
            quantity = bond['quantity']
            principal_value = quantity * 100  # Par value is 100
            
            # Update cash
            idx = self.cash_position.index.get_indexer([date], method='ffill')[0]
            self.cash_position.iloc[idx:] += principal_value
            
            # Remove bond from holdings
            del self.bond_holdings[bond_id]
            
            # Record transaction
            self.transactions.append({
                'date': date,
                'bond_id': bond_id,
                'quantity': -quantity,
                'price': 100,
                'value': principal_value,
                'type': 'maturity'
            })
    
    def process_coupon_payment(self, date):
        """
        Process coupon payments for all bonds on the given date
        
        Parameters:
        - date: Payment date to check for coupons
        """
        total_coupon = 0
        
        for bond_id, details in list(self.bond_holdings.items()):
            issue_date = details['issue_date']
            maturity_date = details['maturity_date']
            
            # Semi-annual coupon payment
            payment_dates = pd.date_range(
                start=issue_date + timedelta(days=180),
                end=maturity_date,
                freq='180D'
            )
            
            if date in payment_dates:
                quantity = details['quantity']
                coupon_rate = details['coupon']
                coupon_payment = quantity * 100 * (coupon_rate / 2)  # Semi-annual payment
                total_coupon += coupon_payment
                
                # Record transaction
                self.transactions.append({
                    'date': date,
                    'bond_id': bond_id,
                    'quantity': 0,
                    'price': 0,
                    'value': coupon_payment,
                    'type': 'coupon'
                })
        
        if total_coupon > 0:
            # Update cash position
            idx = self.cash_position.index.get_indexer([date], method='ffill')[0]
            self.cash_position.iloc[idx:] += total_coupon
    
    def _calculate_positions(self, date):
        """
        Calculate the total value of bond positions
        
        Parameters:
        - date: Date for calculation
        """
        total_value = 0
        
        for bond_id, details in self.bond_holdings.items():
            quantity = details['quantity']
            if bond_id in self.bond_prices:
                price = self.bond_prices[bond_id]
                total_value += quantity * price
        
        idx = self.bond_position.index.get_indexer([date], method='ffill')[0]
        self.bond_values.iloc[idx:] = total_value
        self.bond_position.iloc[idx:] = sum(
            details['quantity'] for details in self.bond_holdings.values()
        )
    
    def run_backtest(self):
        """
        Run the full backtest simulation
        """
        for date in self.dates:
            # Check for bond maturities
            for bond_id in list(self.bond_holdings.keys()):
                if self.bond_holdings[bond_id]['maturity_date'] == date:
                    self.process_bond_maturity(date, bond_id)

            # Process coupon payments
            self.process_coupon_payment(date)
            
            # Update bond prices and positions
            self.update_bond_prices(date)
            self._calculate_positions(date)
            
            # Execute trades for the day
            date_string = date.strftime("%Y-%m-%d")
            if date_string in self.trades.keys():
                for trade in self.trades[date_string]:
                    date_string = trade.date
                    bond_id = trade.id
                    qty = trade.qty
                    tenor = trade.tenor
                    coupon = trade.coupon
                    self.execute_trade(date=date_string, bond_id=bond_id, quantity=qty, tenor=tenor, coupon=coupon)

            # Calculate daily P&L
            idx = self.daily_pnl.index.get_indexer([date])[0]
            if idx > 0:
                previous_cash = self.cash_position.iloc[idx-1]
                previous_bonds = self.bond_values.iloc[idx-1]
                current_cash = self.cash_position.iloc[idx]
                current_bonds = self.bond_values.iloc[idx]
                
                daily_pnl = (current_cash + current_bonds) - (previous_cash + previous_bonds)
                self.daily_pnl.iloc[idx] = daily_pnl
                
                self.cumulative_pnl.iloc[idx] = self.cumulative_pnl.iloc[idx-1] + daily_pnl
            else:
                self.cumulative_pnl.iloc[idx] = 0
    
    def generate_summary(self):
        """
        Generate summary statistics of the backtest
        
        Returns:
        - Dictionary with summary statistics
        """
        total_return = self.cumulative_pnl.iloc[-1]
        percent_return = total_return / self.initial_cash * 100
        
        annual_return = (
            (1 + total_return / self.initial_cash) ** 
            (365 / (self.end_date - self.start_date).days) - 
            1
        ) * 100
        
        volatility = self.daily_pnl.std() * np.sqrt(252)
        sharpe_ratio = annual_return / (volatility / self.initial_cash * 100) if volatility != 0 else 0
        
        drawdown = (self.cumulative_pnl.cummax() - self.cumulative_pnl)
        max_drawdown = drawdown.max() / self.initial_cash * 100
        
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return': total_return,
            'percent_return': percent_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_bond_position': self.bond_position.iloc[-1],
            'final_cash_position': self.cash_position.iloc[-1],
            'total_value': self.bond_values.iloc[-1] + self.cash_position.iloc[-1]
        }
    
    def plot_results(self):
        """
        Plot the results of the backtest
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
        
        # Plot bond position (quantity)
        # axes[0].plot(self.bond_position.index, self.bond_position, 'b-', linewidth=2)
        # axes[0].set_title('Bond Position (Quantity)', fontsize=16)
        # axes[0].set_ylabel('Number of Bonds', fontsize=14)

        axes[0].plot(self.bond_values.index, self.bond_values, 'b-', linewidth=2)
        axes[0].set_title('Bond Position (Value)', fontsize=16)
        axes[0].set_ylabel('Bond Value', fontsize=14)
        axes[0].grid(True)
        
        # Plot cash position
        axes[1].plot(self.cash_position.index, self.cash_position, 'g-', linewidth=2)
        axes[1].set_title('Cash Position', fontsize=16)
        axes[1].set_ylabel('Cash ($)', fontsize=14)
        axes[1].grid(True)
        
        # Plot cumulative P&L
        axes[2].plot(self.cumulative_pnl.index, self.cumulative_pnl, 'r-', linewidth=2)
        axes[2].set_title('Cumulative P&L', fontsize=16)
        axes[2].set_ylabel('P&L ($)', fontsize=14)
        axes[2].set_xlabel('Date', fontsize=14)
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig