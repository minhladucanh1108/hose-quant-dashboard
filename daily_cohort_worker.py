
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yfinance as yf

from quant_engine import StrategyConfig, add_features, signal_mask
from cohort_engine import freeze_cohort, update_marks_from_history

CONFIG = Path("research_config.json")

UNIVERSE = [
    "VCB.VN","BID.VN","CTG.VN","TCB.VN","MBB.VN","ACB.VN","STB.VN","HDB.VN","VPB.VN","TPB.VN",
    "VIB.VN","LPB.VN","MSB.VN","OCB.VN","SHB.VN","SSI.VN","VCI.VN","VND.VN","HCM.VN","FTS.VN",
    "BSI.VN","CTS.VN","HPG.VN","HSG.VN","NKG.VN","VHM.VN","VIC.VN","VRE.VN","PDR.VN","DIG.VN",
    "DXG.VN","KDH.VN","NLG.VN","FCN.VN","KBC.VN","VCG.VN","FPT.VN","MWG.VN","PNJ.VN","DGW.VN",
    "FRT.VN","CTR.VN","VNM.VN","MSN.VN","SAB.VN","PAN.VN","GAS.VN","PLX.VN","POW.VN","GVR.VN",
    "PVD.VN","NT2.VN","PC1.VN","REE.VN","DPM.VN","DCM.VN","CSV.VN","PHR.VN","VHC.VN","ANV.VN",
    "FMC.VN","HAG.VN","DBC.VN","VJC.VN","GMD.VN","HAH.VN","SCS.VN","BVH.VN"
]


def load_config():
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    return {
        "primary_mode": "EMA_VOL_MA200",
        "vol_trigger": 1.5,
        "cross_lookback": 3,
        "min_hold_sessions": 2,
        "max_hold_sessions": 10,
    }


def fetch_batch(tickers, period="3y", interval="1d"):
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    out = {}
    if raw is None or raw.empty:
        return out

    if len(tickers) == 1:
        out[tickers[0]] = raw.dropna(how="all")
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(map(str, raw.columns.get_level_values(0)))
        lvl1 = set(map(str, raw.columns.get_level_values(1)))
        for t in tickers:
            try:
                if t in lvl0:
                    d = raw[t].dropna(how="all")
                elif t in lvl1:
                    d = raw.xs(t, axis=1, level=1).dropna(how="all")
                else:
                    continue
                if not d.empty:
                    out[t] = d
            except Exception:
                continue
    return out


def main():
    cfg_json = load_config()
    mode = cfg_json.get("primary_mode", "EMA_VOL_MA200")
    cfg = StrategyConfig(
        vol_trigger=float(cfg_json.get("vol_trigger", 1.5)),
        cross_lookback=int(cfg_json.get("cross_lookback", 3)),
        min_hold_sessions=int(cfg_json.get("min_hold_sessions", 2)),
        max_hold_sessions=int(cfg_json.get("max_hold_sessions", 10)),
    )

    price_map = fetch_batch(UNIVERSE, period="3y", interval="1d")
    feature_map = {}
    latest_dates = []

    for t in UNIVERSE:
        d = price_map.get(t)
        if d is None or d.empty:
            continue
        try:
            f = add_features(d, cfg)
            if len(f) < 220:
                continue
            dt = pd.Timestamp(f.index[-1]).date().isoformat()
            latest_dates.append(dt)
            feature_map[t] = f
        except Exception as e:
            print("feature error", t, e)

    if not latest_dates:
        raise RuntimeError("No daily prices loaded")

    cohort_date = max(latest_dates)
    signals = []

    for t, f in feature_map.items():
        dt = pd.Timestamp(f.index[-1]).date().isoformat()
        if dt != cohort_date:
            print("skip stale ticker", t, dt, "latest", cohort_date)
            continue

        try:
            sig = signal_mask(f, mode, cfg)
            if bool(sig.iloc[-1]):
                r = f.iloc[-1]
                signals.append({
                    "ticker": t.replace(".VN",""),
                    "reference_price": float(r["Close"]),
                    "reference_source": "EOD_CLOSE",
                    "signal_score": float(r["RESEARCH_SCORE"]),
                    "above_ma200": bool(r["ABOVE_MA200"]),
                    "wy_sos": bool(r["WY_SOS"]),
                    "wy_spring": bool(r["WY_SPRING"]),
                    "wy_lps": bool(r["WY_LPS"]),
                })
        except Exception as e:
            print("signal error", t, e)

    rows = pd.DataFrame(signals)
    n, created = freeze_cohort(
        rows,
        cohort_date=cohort_date,
        mode=mode,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    print("COHORT DATE :", cohort_date)
    print("MODE        :", mode)
    print("NEW COHORT  :", created)
    print("SIGNALS     :", n)

    history = {}
    for t, d in price_map.items():
        history[t.replace(".VN","")] = d

    marks, summaries = update_marks_from_history(history, max_t=10)
    print("MARK UPSERTS:", marks)
    print("T10 SUMMARIES:", summaries)


if __name__ == "__main__":
    main()
