"""
Data ingestion pipeline for Binance BTCUSDT margin data.
Downloads historical OHLCV data and stores it as Parquet files.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from binance.client import Client
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

load_dotenv()

TIMEFRAME_MAP = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
}

TIMEFRAME_HISTORY = {
    "1m": "short_tf_years",
    "5m": "short_tf_years",
    "15m": "short_tf_years",
    "1h": "long_tf_years",
    "4h": "long_tf_years",
    "1d": "long_tf_years",
}


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    """Load project settings from YAML config."""
    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)
    # Resolve env vars for API keys
    for key in ["api_key", "api_secret"]:
        val = settings.get("binance", {}).get(key, "")
        if val.startswith("${") and val.endswith("}"):
            env_var = val[2:-1]
            settings["binance"][key] = os.environ.get(env_var, "")
    return settings


def get_binance_client(settings: dict) -> Client:
    """Create a Binance client from settings."""
    api_key = settings["binance"]["api_key"]
    api_secret = settings["binance"]["api_secret"]
    if not api_key or not api_secret:
        logger.warning("No Binance API keys found. Using public endpoints (rate-limited).")
        return Client("", "")
    return Client(api_key, api_secret)


# Estimated candles per year for progress bar estimation
_CANDLES_PER_YEAR = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


def _estimate_total_candles(interval_str: str, years: float) -> int:
    """Estimate total candles for a given interval and years."""
    # Reverse-lookup from Binance interval constant to our key
    reverse_map = {v: k for k, v in TIMEFRAME_MAP.items()}
    tf_key = reverse_map.get(interval_str, "1h")
    return int(_CANDLES_PER_YEAR.get(tf_key, 8760) * years)


def fetch_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: Optional[str] = None,
    estimated_total: int = 0,
) -> pd.DataFrame:
    """
    Fetch historical klines from Binance with progress bar.

    Args:
        client: Binance client
        symbol: Trading pair (e.g. "BTCUSDT")
        interval: Kline interval (e.g. Client.KLINE_INTERVAL_1HOUR)
        start_date: Start date string (e.g. "1 Jan 2022")
        end_date: End date string (optional)
        estimated_total: Estimated total candles for progress bar

    Returns:
        DataFrame with OHLCV data
    """
    reverse_map = {v: k for k, v in TIMEFRAME_MAP.items()}
    tf_label = reverse_map.get(interval, interval)

    logger.info(f"Fetching {symbol} {tf_label} from {start_date} to {end_date or 'now'}")

    # Manual pagination with progress bar
    all_klines = []
    limit = 1000

    # Parse start date to timestamp ms
    from dateutil import parser as dateparser_util
    start_dt = dateparser_util.parse(start_date)
    start_ts = int(start_dt.timestamp() * 1000)

    end_ts = None
    if end_date:
        end_dt = dateparser_util.parse(end_date)
        end_ts = int(end_dt.timestamp() * 1000)
    else:
        end_ts = int(datetime.utcnow().timestamp() * 1000)

    pbar = tqdm(
        total=estimated_total if estimated_total > 0 else None,
        desc=f"  📥 {symbol} {tf_label}",
        unit=" candles",
        bar_format="{desc} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    current_ts = start_ts
    while current_ts < end_ts:
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_ts,
                endTime=end_ts,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"API error, retrying in 5s: {e}")
            time.sleep(5)
            continue

        if not klines:
            break

        all_klines.extend(klines)
        pbar.update(len(klines))

        # Move to next batch
        last_open_time = klines[-1][0]
        current_ts = last_open_time + 1

        if len(klines) < limit:
            break

        time.sleep(0.05)  # Small delay to avoid rate limiting

    pbar.close()

    if not all_klines:
        logger.warning(f"No data returned for {symbol} {tf_label}")
        return pd.DataFrame()

    df = pd.DataFrame(
        all_klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    df["trades"] = df["trades"].astype(int)
    df = df.drop(columns=["ignore"])
    df = df.set_index("open_time")
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")]

    logger.info(f"✅ {symbol} {tf_label}: {len(df):,} candles downloaded")
    return df


def save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to Parquet file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    logger.info(f"Saved {len(df)} rows to {filepath}")


def load_parquet(filepath: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from Parquet file if it exists."""
    if Path(filepath).exists():
        df = pd.read_parquet(filepath, engine="pyarrow")
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    return None


def get_parquet_path(raw_dir: str, symbol: str, timeframe: str) -> str:
    """Get the Parquet file path for a given symbol and timeframe."""
    return os.path.join(raw_dir, f"{symbol}_{timeframe}.parquet")


def ingest_timeframe(
    client: Client,
    symbol: str,
    timeframe: str,
    raw_dir: str,
    years: int,
) -> pd.DataFrame:
    """
    Ingest data for a single timeframe with incremental updates.

    If data already exists, only fetches new data since the last timestamp.
    """
    filepath = get_parquet_path(raw_dir, symbol, timeframe)
    interval = TIMEFRAME_MAP[timeframe]

    existing_df = load_parquet(filepath)

    if existing_df is not None and len(existing_df) > 0:
        last_ts = existing_df.index.max()
        start_date = (last_ts + timedelta(minutes=1)).strftime("%d %b %Y %H:%M:%S")
        logger.info(f"Incremental update from {start_date}")
        estimated = 0  # Unknown for incremental
    else:
        start_date = (datetime.utcnow() - timedelta(days=365 * years)).strftime("%d %b %Y")
        logger.info(f"Full download from {start_date}")
        estimated = _estimate_total_candles(interval, years)

    new_df = fetch_klines(client, symbol, interval, start_date, estimated_total=estimated)

    if existing_df is not None and len(new_df) > 0:
        df = pd.concat([existing_df, new_df])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
    elif existing_df is not None:
        df = existing_df
    else:
        df = new_df

    if len(df) > 0:
        save_parquet(df, filepath)

    return df


def ingest_all(settings: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """
    Ingest all timeframes for all symbols defined in settings.

    Returns:
        Dictionary mapping timeframe -> DataFrame (for default symbol)
        Use ingest_all_symbols() for multi-asset.
    """
    if settings is None:
        settings = load_settings()

    client = get_binance_client(settings)
    symbol = settings["data"]["symbol"]
    raw_dir = settings["data"]["raw_dir"]
    history = settings["data"]["history"]
    timeframes = settings["data"]["timeframes"]

    results = {}
    total_tf = len(timeframes)
    for i, tf in enumerate(timeframes, 1):
        years_key = TIMEFRAME_HISTORY[tf]
        yrs = history[years_key]
        est = _CANDLES_PER_YEAR.get(tf, 8760) * yrs
        logger.info(f"\n{'='*50}")
        logger.info(f"[{i}/{total_tf}] Ingesting {symbol} {tf} ({yrs} years, ~{est:,.0f} candles)")
        logger.info(f"{'='*50}")
        try:
            df = ingest_timeframe(client, symbol, tf, raw_dir, yrs)
            results[tf] = df
            logger.info(f"✅ {tf}: {len(df):,} candles | {df.index.min()} → {df.index.max()}")
        except Exception as e:
            logger.error(f"❌ Failed to ingest {tf}: {e}")
            results[tf] = pd.DataFrame()
        time.sleep(1)  # Rate limiting

    return results


def ingest_all_symbols(settings: Optional[dict] = None) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Ingest all timeframes for ALL symbols defined in settings.

    Returns:
        Nested dict: symbol -> timeframe -> DataFrame
    """
    if settings is None:
        settings = load_settings()

    symbols = settings["data"].get("symbols", [settings["data"]["symbol"]])
    client = get_binance_client(settings)
    raw_dir = settings["data"]["raw_dir"]
    history = settings["data"]["history"]
    timeframes = settings["data"]["timeframes"]

    all_data = {}
    total = len(symbols) * len(timeframes)
    step = 0

    for symbol in symbols:
        all_data[symbol] = {}
        for tf in timeframes:
            step += 1
            years_key = TIMEFRAME_HISTORY[tf]
            yrs = history[years_key]
            est = _CANDLES_PER_YEAR.get(tf, 8760) * yrs
            logger.info(f"\n[{step}/{total}] Ingesting {symbol} {tf} ({yrs}y, ~{est:,.0f} candles)")
            try:
                df = ingest_timeframe(client, symbol, tf, raw_dir, yrs)
                all_data[symbol][tf] = df
                logger.info(f"✅ {symbol} {tf}: {len(df):,} candles")
            except Exception as e:
                logger.error(f"❌ Failed {symbol} {tf}: {e}")
                all_data[symbol][tf] = pd.DataFrame()
            time.sleep(1)

    return all_data


def load_all_data(settings: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """
    Load all previously ingested data from Parquet files (default symbol).

    Returns:
        Dictionary mapping timeframe -> DataFrame
    """
    if settings is None:
        settings = load_settings()

    symbol = settings["data"]["symbol"]
    raw_dir = settings["data"]["raw_dir"]
    timeframes = settings["data"]["timeframes"]

    results = {}
    for tf in timeframes:
        filepath = get_parquet_path(raw_dir, symbol, tf)
        df = load_parquet(filepath)
        if df is not None:
            results[tf] = df
        else:
            logger.warning(f"No data found for {tf} at {filepath}")

    return results


def load_all_symbols_data(settings: Optional[dict] = None) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all previously ingested data for ALL symbols.

    Returns:
        Nested dict: symbol -> timeframe -> DataFrame
    """
    if settings is None:
        settings = load_settings()

    symbols = settings["data"].get("symbols", [settings["data"]["symbol"]])
    raw_dir = settings["data"]["raw_dir"]
    timeframes = settings["data"]["timeframes"]

    all_data = {}
    for symbol in symbols:
        all_data[symbol] = {}
        for tf in timeframes:
            filepath = get_parquet_path(raw_dir, symbol, tf)
            df = load_parquet(filepath)
            if df is not None:
                all_data[symbol][tf] = df

    # Log summary
    for symbol, tf_data in all_data.items():
        total = sum(len(df) for df in tf_data.values())
        logger.info(f"📊 {symbol}: {len(tf_data)} TFs, {total:,} total candles")

    return all_data


if __name__ == "__main__":
    settings = load_settings()
    ingest_all(settings)
