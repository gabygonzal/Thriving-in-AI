"""
Calculate daily returns for SPY (SPDR S&P 500 ETF Trust)

Formula: Return = (price_t / price_t-1) - 1
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_spy_data(start_date=None, end_date=None):
    """
    Fetch SPY historical data from Yahoo Finance
    
    Args:
        start_date: Start date for data (default: 1 year ago)
        end_date: End date for data (default: today)
    
    Returns:
        pandas DataFrame with SPY data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    # Fetch SPY data
    spy = yf.Ticker("SPY")
    data = spy.history(start=start_date, end=end_date)
    
    return data

def calculate_daily_returns(data):
    """
    Calculate daily returns using the formula: Return = (price_t / price_t-1) - 1
    
    Args:
        data: pandas DataFrame with 'Close' column containing prices
    
    Returns:
        pandas Series with daily returns
    """
    # Use closing prices
    prices = data['Close']
    
    # Calculate daily returns: (price_t / price_t-1) - 1
    daily_returns = (prices / prices.shift(1)) - 1
    
    return daily_returns

def main():
    """
    Main function to fetch SPY data and calculate daily returns
    """
    print("Fetching SPY data from Yahoo Finance...")
    print("-" * 60)
    
    try:
        # Fetch SPY data
        spy_data = fetch_spy_data()
        
        print(f"Successfully fetched {len(spy_data)} days of SPY data")
        print(f"Date range: {spy_data.index[0].date()} to {spy_data.index[-1].date()}")
        print()
        
        # Calculate daily returns
        daily_returns = calculate_daily_returns(spy_data)
        
        # Remove the first row (NaN because there's no previous day)
        daily_returns = daily_returns.dropna()
        
        print("Daily Returns Calculation:")
        print("=" * 60)
        print(f"Formula: Return = (price_t / price_t-1) - 1")
        print()
        print("Recent Daily Returns:")
        print("-" * 60)
        print(daily_returns.tail(10).to_string())
        print()
        
        # Summary statistics
        print("Summary Statistics:")
        print("-" * 60)
        print(f"Total trading days: {len(daily_returns)}")
        print(f"Average daily return: {daily_returns.mean():.4%}")
        print(f"Standard deviation: {daily_returns.std():.4%}")
        print(f"Minimum daily return: {daily_returns.min():.4%}")
        print(f"Maximum daily return: {daily_returns.max():.4%}")
        print()
        
        # Show first few calculations manually for verification
        print("First 5 Days Calculation (for verification):")
        print("-" * 60)
        prices = spy_data['Close']
        for i in range(1, min(6, len(prices))):
            price_t = prices.iloc[i]
            price_t_minus_1 = prices.iloc[i-1]
            return_val = (price_t / price_t_minus_1) - 1
            print(f"Date: {prices.index[i].date()}")
            print(f"  Price_t: ${price_t:.2f}")
            print(f"  Price_t-1: ${price_t_minus_1:.2f}")
            print(f"  Return = ({price_t:.2f} / {price_t_minus_1:.2f}) - 1 = {return_val:.4%}")
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed yfinance:")
        print("  pip install yfinance pandas")

if __name__ == "__main__":
    main()
