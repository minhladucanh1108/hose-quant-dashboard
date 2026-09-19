
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    ema_regime: int = 200
    vol_window: int = 20
    vol_trigger: float = 1.50
    cross_lookback: int = 3
    stop_loss: float = 0.05
    take_profit: float = 0.15
    min_hold_sessions: int = 2
    max_hold_sessions: int = 10
    buy_fee: float = 0.0015
    sell_fee: float = 0.0025


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        # This function expects one ticker only.
        if len(set(x.columns.get_level_values(0))) == 1:
            x.columns = x.columns.get_level_values(-1)
        else:
            raise ValueError("normalize_ohlcv received multiple tickers")
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    x = x[required].copy()
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=required[:4]).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100).where(avg_loss.ne(0), 100)


def add_features(df: pd.DataFrame, cfg: StrategyConfig = StrategyConfig()) -> pd.DataFrame:
    x = normalize_ohlcv(df)
    x["EMA20"] = x["Close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    x["EMA200"] = x["Close"].ewm(span=cfg.ema_regime, adjust=False).mean()
    x["VOL_MA20"] = x["Volume"].rolling(cfg.vol_window, min_periods=cfg.vol_window).mean()
    x["VOL_RATIO"] = x["Volume"] / x["VOL_MA20"].replace(0, np.nan)
    x["RSI14"] = _rsi(x["Close"], 14)

    # Real crossover event, not merely EMA20 > EMA50.
    x["BULL_CROSS"] = (x["EMA20"] > x["EMA50"]) & (x["EMA20"].shift(1) <= x["EMA50"].shift(1))
    x["CROSS_RECENT"] = (
        x["BULL_CROSS"].astype(int)
        .rolling(cfg.cross_lookback, min_periods=1)
        .max()
        .astype(bool)
    )
    x["ABOVE_MA200"] = x["Close"] > x["EMA200"]
    x["EMA_TREND"] = x["EMA20"] > x["EMA50"]
    x["VOL_CONFIRM"] = x["VOL_RATIO"] >= cfg.vol_trigger

    # Quantified Wyckoff-like event proxies. These are research features,
    # not claims of a canonical Wyckoff phase classification.
    low20_prev = x["Low"].rolling(20, min_periods=20).min().shift(1)
    high20_prev = x["High"].rolling(20, min_periods=20).max().shift(1)
    high60_prev = x["High"].rolling(60, min_periods=60).max().shift(1)
    low60_prev = x["Low"].rolling(60, min_periods=60).min().shift(1)
    range60 = (high60_prev - low60_prev) / x["Close"].replace(0, np.nan)

    x["WY_SPRING"] = (
        (x["Low"] < low20_prev * 0.995)
        & (x["Close"] > low20_prev)
        & (x["Close"] > x["Open"])
        & (x["VOL_RATIO"] >= 1.0)
    )
    x["WY_SOS"] = (
        (x["Close"] > high20_prev)
        & (x["Close"] > x["Open"])
        & (x["VOL_RATIO"] >= 1.30)
    )
    x["WY_LPS"] = (
        (x["EMA20"] > x["EMA50"])
        & (x["Low"] <= x["EMA20"] * 1.015)
        & (x["Close"] >= x["EMA20"])
        & (x["Close"] > x["Open"])
        & (x["VOL_RATIO"].between(0.40, 0.95))
    )
    x["WY_BASE"] = (range60 <= 0.25) & (x["Close"] >= low60_prev + 0.45 * (high60_prev - low60_prev))
    x["WY_EVENT"] = x[["WY_SPRING", "WY_SOS", "WY_LPS"]].any(axis=1)

    # Explainable research ranking score. Not a calibrated return probability.
    score = pd.Series(0.0, index=x.index)
    score += np.where(x["EMA_TREND"], 20.0, 0.0)
    score += np.where(x["ABOVE_MA200"], 10.0, 0.0)
    score += np.where(x["CROSS_RECENT"], 20.0, 0.0)
    score += np.where(x["VOL_CONFIRM"], 15.0, np.where(x["VOL_RATIO"] >= 1.1, 7.5, 0.0))
    score += np.where(x["RSI14"].between(50, 70), 10.0, np.where(x["RSI14"].between(45, 75), 5.0, 0.0))
    score += np.where(x["WY_SOS"], 20.0, 0.0)
    score += np.where(x["WY_SPRING"], 15.0, 0.0)
    score += np.where(x["WY_LPS"], 10.0, 0.0)
    score += np.where(x["WY_BASE"], 5.0, 0.0)
    x["RESEARCH_SCORE"] = score.clip(0, 100)

    return x


def signal_mask(x: pd.DataFrame, mode: str, cfg: StrategyConfig = StrategyConfig()) -> pd.Series:
    mode = mode.upper()
    if mode == "EMA_VOL":
        return x["CROSS_RECENT"] & x["VOL_CONFIRM"]
    if mode == "EMA_VOL_MA200":
        return x["ABOVE_MA200"] & x["CROSS_RECENT"] & x["VOL_CONFIRM"]
    if mode == "WYCKOFF":
        return x["WY_EVENT"]
    if mode == "COMPOSITE":
        # Technical/Wyckoff event trigger. Sentiment is applied as an overlay outside
        # this price-only engine so that price and news evidence can be audited separately.
        return (x["WY_SOS"] | x["WY_SPRING"] | (x["CROSS_RECENT"] & x["VOL_CONFIRM"]))
    raise ValueError(f"Unknown mode: {mode}")


def explain_latest(x: pd.DataFrame) -> Dict[str, object]:
    if x.empty:
        return {}
    r = x.iloc[-1]
    events = []
    if bool(r.get("WY_SPRING", False)):
        events.append("Spring proxy")
    if bool(r.get("WY_SOS", False)):
        events.append("SOS proxy")
    if bool(r.get("WY_LPS", False)):
        events.append("LPS proxy")
    return {
        "date": x.index[-1],
        "close": float(r["Close"]),
        "above_ma200": bool(r["ABOVE_MA200"]),
        "ema_trend": bool(r["EMA_TREND"]),
        "cross_recent": bool(r["CROSS_RECENT"]),
        "vol_ratio": float(r["VOL_RATIO"]) if pd.notna(r["VOL_RATIO"]) else np.nan,
        "rsi14": float(r["RSI14"]) if pd.notna(r["RSI14"]) else np.nan,
        "wyckoff_event": ", ".join(events) if events else "None",
        "research_score": float(r["RESEARCH_SCORE"]),
    }


def backtest_trades(
    df: pd.DataFrame,
    mode: str,
    cfg: StrategyConfig = StrategyConfig(),
) -> pd.DataFrame:
    """
    Trade-level evaluator:
    - signal is observed at close t
    - entry occurs at next available session open t+1
    - no exit before min_hold_sessions
    - conservative same-day stop/TP ambiguity: stop wins
    - fees included
    This is NOT a portfolio CAGR engine; overlapping trades across tickers are not capital-constrained here.
    """
    x = add_features(df, cfg)
    sig = signal_mask(x, mode, cfg).fillna(False)
    rows = []
    i = max(cfg.ema_regime + 5, 205)
    n = len(x)

    while i < n - 1:
        if not bool(sig.iloc[i]):
            i += 1
            continue

        signal_i = i
        entry_i = i + 1
        entry = float(x["Open"].iloc[entry_i])
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue

        stop = entry * (1 - cfg.stop_loss)
        tp = entry * (1 + cfg.take_profit)
        last_i = min(entry_i + cfg.max_hold_sessions, n - 1)
        first_exit_i = min(entry_i + cfg.min_hold_sessions, last_i)

        exit_i = last_i
        exit_px = float(x["Close"].iloc[last_i])
        reason = "TIME"

        for j in range(first_exit_i, last_i + 1):
            low = float(x["Low"].iloc[j])
            high = float(x["High"].iloc[j])
            stop_hit = low <= stop
            tp_hit = high >= tp

            if stop_hit and tp_hit:
                exit_i, exit_px, reason = j, stop, "STOP_CONSERVATIVE"
                break
            if stop_hit:
                exit_i, exit_px, reason = j, stop, "STOP"
                break
            if tp_hit:
                exit_i, exit_px, reason = j, tp, "TAKE_PROFIT"
                break

        gross = exit_px / entry - 1
        net = (exit_px * (1 - cfg.sell_fee)) / (entry * (1 + cfg.buy_fee)) - 1

        rows.append(
            {
                "signal_date": x.index[signal_i],
                "entry_date": x.index[entry_i],
                "entry_open": entry,
                "exit_date": x.index[exit_i],
                "exit_price": exit_px,
                "holding_sessions": exit_i - entry_i,
                "exit_reason": reason,
                "gross_return": gross,
                "net_return": net,
                "signal_score": float(x["RESEARCH_SCORE"].iloc[signal_i]),
                "signal_above_ma200": bool(x["ABOVE_MA200"].iloc[signal_i]),
                "signal_wy_sos": bool(x["WY_SOS"].iloc[signal_i]),
                "signal_wy_spring": bool(x["WY_SPRING"].iloc[signal_i]),
                "signal_wy_lps": bool(x["WY_LPS"].iloc[signal_i]),
            }
        )

        # Prevent overlapping trades in the same ticker.
        i = max(exit_i + 1, i + 1)

    return pd.DataFrame(rows)


def trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "profit_factor": np.nan,
        }
    r = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_net_return": float(r.mean()),
        "median_net_return": float(r.median()),
        "profit_factor": float(pos / neg) if neg > 0 else np.inf,
    }
