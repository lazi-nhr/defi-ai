# -----------------------------------------------------------------------------
# Binance USDⓈ-M Perpetual Futures Funding Rate Downloader
#
# Description:
# This script downloads the historical funding rate data for all active
# USDⓈ-Margined perpetual futures contracts on Binance for a specified
# date range.
#
# Features:
# - Accepts a start and end date from the command line.
# - Automatically fetches a list of all perpetual symbols.
# - Handles API pagination to retrieve the complete history within the date range.
# - Saves the data for each symbol into a separate, efficiently stored
#   Parquet file.
# - Organizes output files into a structured directory: 'data/binance_funding/'.

# -----------------------------------------------------------------------------

import os
import argparse
import pandas as pd
from binance.client import Client
from datetime import datetime
from tqdm import tqdm
import time

def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description='Download Binance Funding Rate History.')
    parser.add_argument('--start', required=True, type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', required=True, type=str, help='End date in YYYY-MM-DD format')
    return parser

def convert_date_to_ms(date_str):
    """Converts a YYYY-MM-DD date string to a milliseconds Unix timestamp."""
    # --- CHANGE: Added .strip() to remove any leading/trailing whitespace from input ---
    return int(datetime.strptime(date_str.strip(), '%Y-%m-%d').timestamp() * 1000)

def get_all_perpetual_symbols(client):
    """Gets all USDⓈ-M perpetual contract symbols."""
    try:
        exchange_info = client.futures_exchange_info()
        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['contractType'] == 'PERPETUAL' and s['status'] == 'TRADING'
        ]
        print(f"Found {len(symbols)} perpetual trading symbols.")
        return symbols
    except Exception as e:
        print(f"Error fetching exchange info: {e}")
        return []

def fetch_funding_history(client, symbol, start_ms, end_ms):
    """
    Fetches the complete funding rate history for a single symbol, handling API pagination.
    """
    all_funding_data = []
    current_start_time = start_ms
    limit = 1000

    while current_start_time < end_ms:
        try:
            data = client.futures_funding_rate(
                symbol=symbol,
                startTime=current_start_time,
                endTime=end_ms,
                limit=limit
            )
            
            if not data:
                break
            
            all_funding_data.extend(data)
            
            last_record_time = int(data[-1]['fundingTime'])
            current_start_time = last_record_time + 1
            
            if len(data) < limit:
                break

            time.sleep(0.1)

        except Exception as e:
            print(f"An error occurred while fetching funding rate for {symbol}: {e}")
            time.sleep(1)
            break
            
    return all_funding_data

def main():
    """Main execution function."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    start_date_str = args.start
    end_date_str = args.end
    output_dir = "data/binance_funding"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    start_ms = convert_date_to_ms(start_date_str)
    end_ms = convert_date_to_ms(end_date_str) + (24 * 60 * 60 * 1000 - 1)
    
    client = Client()

    symbols = get_all_perpetual_symbols(client)
    if not symbols:
        print("Could not fetch symbols. Exiting.")
        return

    for symbol in tqdm(symbols, desc="Processing symbols"):
        funding_data = fetch_funding_history(client, symbol, start_ms, end_ms)

        if not funding_data:
            continue

        df = pd.DataFrame(funding_data)
        
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        if 'markPrice' in df.columns:
            df['markPrice'] = df['markPrice'].astype(float)

        filename = f"{symbol.replace('USDT', '')}_funding_{start_date_str.strip()}_{end_date_str.strip()}.parquet"
        output_path = os.path.join(output_dir, filename)
        
        try:
            df.to_parquet(output_path, index=False, engine='pyarrow')
        except Exception as e:
            print(f"Error saving file for {symbol}: {e}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Usage Guide
#
#
# How to Run:
#    Execute the script from your terminal, providing the mandatory 'start' and
#    'end' dates as command-line arguments.
#
# Syntax:
#    python binance_funding.py --start YYYY-MM-DD --end YYYY-MM-DD
#
# Example:
#    To download all funding rate data from January 1, 2024, to August 22, 2024:
#    python binance_funding.py --start 2024-01-01 --end 2024-08-22
#
# Understanding Symbol Names:
#    - Standard contracts (e.g., 'BTC') refer to the BTC/USDT pair.
#    - Multiple contracts (e.g., '1000SHIB') refer to pairs like 1000SHIB/USDT,
#      where the trading unit is 1000 SHIB tokens.
#    - USDC-margined contracts will have 'USDC' in their name (e.g., 'BTCUSDC').
# -----------------------------------------------------------------------------
