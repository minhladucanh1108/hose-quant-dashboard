
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

COHORT_FILE = Path("paper_cohorts.csv")
MARK_FILE = Path("paper_marks.csv")
SUMMARY_FILE = Path("paper_t10_summary.csv")


COHORT_COLUMNS = [
    "cohort_date","ticker","mode","reference_price","reference_source",
    "signal_score","above_ma200","wy_sos","wy_spring","wy_lps","frozen_at_utc"
]

MARK_COLUMNS = [
    "cohort_date","ticker","t_index","mark_date","close_price","return_pct"
]

SUMMARY_COLUMNS = [
    "cohort_date","ticker","reference_price","t10_date","t10_close","t10_return_pct","status"
]


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    x = pd.read_csv(path)
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan
    return x[cols].copy()


def load_cohorts() -> pd.DataFrame:
    return _read(COHORT_FILE, COHORT_COLUMNS)


def load_marks() -> pd.DataFrame:
    return _read(MARK_FILE, MARK_COLUMNS)


def load_summary() -> pd.DataFrame:
    return _read(SUMMARY_FILE, SUMMARY_COLUMNS)


def freeze_cohort(rows: pd.DataFrame, cohort_date: str, mode: str, frozen_at_utc: str) -> tuple[int, bool]:
    """
    Freeze one cohort once. If a cohort already exists for cohort_date+mode,
    it is never mutated by later app refreshes.
    """
    cohorts = load_cohorts()
    exists = (
        cohorts["cohort_date"].astype(str).eq(str(cohort_date))
        & cohorts["mode"].astype(str).eq(str(mode))
    ).any()
    if exists:
        return 0, False

    if rows.empty:
        # Write no rows. Caller can log "0-signal day" separately if desired.
        return 0, True

    out = pd.DataFrame({
        "cohort_date": str(cohort_date),
        "ticker": rows["ticker"].astype(str),
        "mode": str(mode),
        "reference_price": pd.to_numeric(rows["reference_price"], errors="coerce"),
        "reference_source": rows.get("reference_source", pd.Series("EOD_CLOSE", index=rows.index)).astype(str),
        "signal_score": pd.to_numeric(rows.get("signal_score", np.nan), errors="coerce"),
        "above_ma200": rows.get("above_ma200", False).astype(bool),
        "wy_sos": rows.get("wy_sos", False).astype(bool),
        "wy_spring": rows.get("wy_spring", False).astype(bool),
        "wy_lps": rows.get("wy_lps", False).astype(bool),
        "frozen_at_utc": str(frozen_at_utc),
    })

    cohorts = pd.concat([cohorts, out], ignore_index=True) if not cohorts.empty else out
    cohorts = cohorts.drop_duplicates(["cohort_date","ticker","mode"], keep="first")
    cohorts.to_csv(COHORT_FILE, index=False)
    return len(out), True


def update_marks_from_history(price_history: dict[str, pd.DataFrame], max_t: int = 10) -> tuple[int, int]:
    """
    T+ is counted in actual ticker trading sessions after the cohort date.
    T+0 = cohort/reference day.
    T+1 = first later trading session close, ...
    Historical cohorts are immutable; marks only append/replace the same key.
    """
    cohorts = load_cohorts()
    marks = load_marks()
    summaries = load_summary()

    new_marks = []
    new_summaries = []

    for _, c in cohorts.iterrows():
        ticker = str(c["ticker"])
        d = price_history.get(ticker)
        if d is None or d.empty or "Close" not in d.columns:
            continue

        x = d.copy()
        x.index = pd.to_datetime(x.index)
        x = x[~x.index.duplicated(keep="last")].sort_index()
        x["Close"] = pd.to_numeric(x["Close"], errors="coerce")
        x = x.dropna(subset=["Close"])

        cohort_dt = pd.Timestamp(c["cohort_date"]).normalize()
        sessions = x[x.index.normalize() >= cohort_dt].copy()
        if sessions.empty:
            continue

        # Require the cohort day itself to exist in the daily history.
        same_day = sessions[sessions.index.normalize() == cohort_dt]
        if same_day.empty:
            continue

        ordered = pd.concat([
            same_day.tail(1),
            sessions[sessions.index.normalize() > cohort_dt]
        ]).head(max_t + 1)

        ref = float(c["reference_price"])
        if not np.isfinite(ref) or ref <= 0:
            continue

        for t_idx, (dt, row) in enumerate(ordered.iterrows()):
            close = float(row["Close"])
            ret = (close / ref - 1.0) * 100.0
            new_marks.append({
                "cohort_date": str(c["cohort_date"]),
                "ticker": ticker,
                "t_index": int(t_idx),
                "mark_date": pd.Timestamp(dt).date().isoformat(),
                "close_price": close,
                "return_pct": ret,
            })

        if len(ordered) >= max_t + 1:
            dt, row = ordered.iloc[max_t].name, ordered.iloc[max_t]
            close = float(row["Close"])
            ret = (close / ref - 1.0) * 100.0
            new_summaries.append({
                "cohort_date": str(c["cohort_date"]),
                "ticker": ticker,
                "reference_price": ref,
                "t10_date": pd.Timestamp(dt).date().isoformat(),
                "t10_close": close,
                "t10_return_pct": ret,
                "status": "WIN" if ret > 0 else ("FLAT" if abs(ret) < 1e-12 else "LOSS"),
            })

    if new_marks:
        nm = pd.DataFrame(new_marks)
        marks = pd.concat([marks, nm], ignore_index=True) if not marks.empty else nm
        marks = marks.drop_duplicates(["cohort_date","ticker","t_index"], keep="last")
        marks = marks.sort_values(["cohort_date","ticker","t_index"])
        marks.to_csv(MARK_FILE, index=False)

    if new_summaries:
        ns = pd.DataFrame(new_summaries)
        summaries = pd.concat([summaries, ns], ignore_index=True) if not summaries.empty else ns
        summaries = summaries.drop_duplicates(["cohort_date","ticker"], keep="last")
        summaries = summaries.sort_values(["cohort_date","ticker"])
        summaries.to_csv(SUMMARY_FILE, index=False)

    return len(new_marks), len(new_summaries)


def cohort_matrix(cohort_date: str) -> pd.DataFrame:
    cohorts = load_cohorts()
    marks = load_marks()

    c = cohorts[cohorts["cohort_date"].astype(str).eq(str(cohort_date))].copy()
    if c.empty:
        return pd.DataFrame()

    m = marks[marks["cohort_date"].astype(str).eq(str(cohort_date))].copy()
    if m.empty:
        c["T+0"] = np.nan
        return c

    ret_pivot = m.pivot(index="ticker", columns="t_index", values="return_pct")
    px_pivot = m.pivot(index="ticker", columns="t_index", values="close_price")

    out = c.set_index("ticker").copy()
    for t in range(0, 11):
        out[f"T+{t} %"] = ret_pivot[t] if t in ret_pivot.columns else np.nan
        out[f"T+{t} Close"] = px_pivot[t] if t in px_pivot.columns else np.nan

    return out.reset_index()
