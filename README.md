# SPY Daily Returns Calculator

This program calculates the daily returns for SPY (SPDR S&P 500 ETF Trust) using the formula:

**Return = (price_t / price_t-1) - 1**

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the program:
```bash
python spy_daily_returns.py
```

## How it works

1. **Data Fetching**: The program fetches SPY historical data from Yahoo Finance using the `yfinance` library
2. **Return Calculation**: For each day, it calculates the return using the closing price of that day divided by the previous day's closing price, minus 1
3. **Output**: Displays recent daily returns, summary statistics, and verification calculations

## Formula Explanation

- `price_t`: Closing price on day t (today)
- `price_t-1`: Closing price on day t-1 (yesterday)
- `Return = (price_t / price_t-1) - 1`

For example, if SPY closed at $400 yesterday and $402 today:
- Return = ($402 / $400) - 1 = 1.005 - 1 = 0.005 = 0.5%
