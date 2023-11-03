import yfinance as yf
import pandas as pd
import logging
import argparse
from Portfolio import Portfolio

def get_data(stock_symbols: list, start: str, end: str, period="1d"):
    stock_data = {}

    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        stock_history = stock.history(period=period, start=start, end=end)
        stock_data[symbol] = stock_history

    return pd.DataFrame({symbol: data["Close"] for symbol, data in stock_data.items()})

def main():
    # args
    parser = argparse.ArgumentParser(description="Configure a Portfolio object")
    parser.add_argument("--data", type=str, default='stock_data.csv', help="Path to a CSV file containing data")
    parser.add_argument("--intervals", type=int, default=252, help="Number of trading intervals per year(default: 252)")
    parser.add_argument("--risk_free_rate", type=int, default=0, help="Risk free rate (default: 0)")
    parser.add_argument("--minimum_weight", type=int, default=0, help="Minimum weight for assets in the portfolio (default: 0)")
    parser.add_argument("--return_calc", type=str, default='log', help="Accepts either 'log' or 'simple' as the method to compute returns (default: 252)")
    parser.add_argument("--mult", type=float, default=1.2, help="multiplier of max sharp return to limit the frontier (default: 252)")

    args = parser.parse_args()
    # logging
    logging.basicConfig(filename='portfolio.log', level=logging.ERROR)
    # Load data from CSV file
    if args.data:
        data = pd.read_csv(args.data)
        print(data.head())
    else:
        print("No data provided. Exiting.")

    portfolio = Portfolio(data=data,
                          intervals=args.intervals,
                          risk_free_rate=args.risk_free_rate,
                          minimum_weight=args.minimum_weight,
                          return_calc=args.return_calc,
                          mult=args.mult)
    print(pd.DataFrame([portfolio.max_sharp_ratio_portfolio()['x'], stock_symbols]).T)
    print(portfolio.get_efficient_frontier())
    portfolio.plot_efficient_frontier()

if __name__ == '__main__':
    # Define a list of stock symbols
    stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "JNJ", "BRK-B", "JPM",
                     "WMT", "DIS", "PG", "V", "PYPL", "NFLX", "CRM", "CSCO", "INTC", "T",
                     "BA", "MA", "XOM", "PEP", "ADBE", "CMCSA", "KO", "HD", "UNH", "TMUS"]
    # Define the date range for historical data
    start_date = "2015-01-01"
    end_date = "2023-10-01"

    # Read data
    df = get_data(stock_symbols, start_date, end_date)
    df.to_csv('stock_data.csv', index=False)
    main()

