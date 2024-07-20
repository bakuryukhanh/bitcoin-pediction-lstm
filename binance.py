import requests
import pandas as pd
from datetime import datetime


def fetch_binance_data(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["date_string"] = df["timestamp"].apply(
        lambda x: datetime.utcfromtimestamp(x / 1000).strftime("%Y-%m-%d")
    )
    df["Date"] = pd.to_numeric(df["timestamp"])
    df["Close"] = df["close"].astype(float)
    return df


fetch_binance_data()
