
import requests

def send_telegram_msg(token, chat_id, text):
    if not token or not chat_id or not text:
        return False
    url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
    payload = {"chat_id": str(chat_id).strip(), "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=8)
        return r.status_code == 200
    except Exception:
        return False


from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from quant_engine import StrategyConfig, add_features, signal_mask, backtest_trades, trade_metrics
from cohort_engine import load_cohorts, load_marks, load_summary, cohort_matrix

st.set_page_config(page_title="HOSE Quant Trading V2", layout="wide")

UNIVERSE = [
    "VCB.VN","BID.VN","CTG.VN","TCB.VN","MBB.VN","ACB.VN","STB.VN","HDB.VN","VPB.VN","TPB.VN",
    "VIB.VN","LPB.VN","MSB.VN","OCB.VN","SHB.VN","SSI.VN","VCI.VN","VND.VN","HCM.VN","FTS.VN",
    "BSI.VN","CTS.VN","HPG.VN","HSG.VN","NKG.VN","VHM.VN","VIC.VN","VRE.VN","PDR.VN","DIG.VN",
    "DXG.VN","KDH.VN","NLG.VN","FCN.VN","KBC.VN","VCG.VN","FPT.VN","MWG.VN","PNJ.VN","DGW.VN",
    "FRT.VN","CTR.VN","VNM.VN","MSN.VN","SAB.VN","PAN.VN","GAS.VN","PLX.VN","POW.VN","GVR.VN",
    "PVD.VN","NT2.VN","PC1.VN","REE.VN","DPM.VN","DCM.VN","CSV.VN","PHR.VN","VHC.VN","ANV.VN",
    "FMC.VN","HAG.VN","DBC.VN","VJC.VN","GMD.VN","HAH.VN","SCS.VN","BVH.VN"
]

MODE_LABELS = {
    "EMA_VOL": "EMA20/50 + Volume (không MA200)",
    "EMA_VOL_MA200": "EMA20/50 + Volume + MA200",
    "WYCKOFF": "Wyckoff event proxies",
    "COMPOSITE": "Composite Technical + Wyckoff",
}

st.title("HOSE Quant Trading V2 · Research Dashboard")
st.caption(
    "Daily-bar research scanner. Signal is formed at close t; executable paper entry is next available open t+1. "
    "Current Yahoo Finance feed is a fallback research source, not an exchange-grade realtime feed."
)

with st.sidebar:
    st.header("Research controls")
    selected_mode = st.selectbox(
        "Strategy variant",
        options=list(MODE_LABELS),
        format_func=lambda x: MODE_LABELS[x],
        index=1,
    )
    vol_trigger = st.slider("Volume trigger / SMA20", 1.0, 2.5, 1.5, 0.05)
    cross_lookback = st.slider("Crossover lookback (sessions)", 1, 10, 3, 1)
    min_hold = st.slider("Minimum holding sessions", 0, 5, 2, 1)
    max_hold = st.slider("Maximum holding sessions", 3, 20, 10, 1)
    st.markdown("---")
    st.caption("Weights/scores are research ranking features, not calibrated return probabilities.")
    st.caption("DNSE integration should replace Yahoo Finance before live execution.")
    st.markdown("---")
    st.subheader("🤖 Cấu hình Telegram Bot")
    tg_token = st.text_input("Bot Token", type="password", help="Token lấy từ @BotFather")
    tg_chat_id = st.text_input("Chat ID", help="Lấy từ @userinfobot")
    auto_tg = st.checkbox("Bật tự động báo tín hiệu T+0", value=True)
    if st.button("🔔 Gửi tin nhắn Test Telegram"):
        if tg_token and tg_chat_id:
            ok = send_telegram_msg(tg_token, tg_chat_id, "🚀 *HOSE Quant V2.1.1*: Kết nối Telegram thành công!")
            if ok:
                st.success("Đã gửi tin nhắn test thành công!")
            else:
                st.error("Gửi thất bại! Kiểm tra lại Token hoặc Chat ID.")
        else:
            st.warning("Vui lòng điền đủ Bot Token và Chat ID.")

cfg = StrategyConfig(
    vol_trigger=vol_trigger,
    cross_lookback=cross_lookback,
    min_hold_sessions=min_hold,
    max_hold_sessions=max_hold,
)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_batch(tickers, period="3y"):
    raw = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
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

@st.cache_data(ttl=300, show_spinner=False)
def load_sentiment():
    p = Path("news_sentiment.csv")
    if not p.exists():
        return pd.DataFrame(columns=["ticker","sentiment_score","article_count","last_title","last_url","updated_at_utc"])
    x = pd.read_csv(p)
    if "ticker" in x.columns:
        x["ticker"] = x["ticker"].astype(str).str.upper().str.strip()
    return x

sent = load_sentiment()
sent_map = sent.set_index("ticker")["sentiment_score"].to_dict() if not sent.empty else {}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_batch(tickers):
    """
    Research-only current-session bar using Yahoo 1-minute fallback.
    This is not exchange-grade realtime and may be delayed/incomplete.
    """
    try:
        raw = yf.download(
            tickers=list(tickers),
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            prepost=False,
        )
    except Exception:
        return {}

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


def append_live_bar(daily: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate current intraday observations into one provisional daily bar and
    append/replace that calendar day. Used only for the T+0 live research view.
    """
    if daily is None or daily.empty:
        return daily
    if intraday is None or intraday.empty:
        return daily.copy()

    d = daily.copy()
    i = intraday.copy()
    if isinstance(i.columns, pd.MultiIndex):
        i.columns = i.columns.get_level_values(0)

    req = ["Open","High","Low","Close","Volume"]
    if not all(c in i.columns for c in req):
        return d

    for c in req:
        i[c] = pd.to_numeric(i[c], errors="coerce")
    i = i.dropna(subset=["Open","High","Low","Close"])
    if i.empty:
        return d

    day = pd.Timestamp(i.index[-1]).date()
    live_row = pd.DataFrame(
        {
            "Open": [float(i["Open"].dropna().iloc[0])],
            "High": [float(i["High"].max())],
            "Low": [float(i["Low"].min())],
            "Close": [float(i["Close"].dropna().iloc[-1])],
            "Volume": [float(i["Volume"].fillna(0).sum())],
        },
        index=[pd.Timestamp(day)],
    )

    d.index = pd.to_datetime(d.index)
    d = d[d.index.normalize() != pd.Timestamp(day)]
    return pd.concat([d, live_row]).sort_index()


def fmt_pct(v):
    return "—" if pd.isna(v) else f"{float(v):+.2f}%"

tabs = st.tabs(["Live T+0", "Frozen cohorts", "Signal detail", "Backtest", "News sentiment", "Audit"])

with tabs[0]:
    st.subheader("Live T+0 research scanner")
    st.caption(
        "Dynamic current-session view. Price/volume use Yahoo 1-minute fallback when available. "
        "The list may change intraday. It is NOT the frozen historical cohort."
    )

    with st.spinner("Loading daily history and current-session bars..."):
        price_map = fetch_batch(tuple(UNIVERSE), "3y")
        intraday_map = fetch_intraday_batch(tuple(UNIVERSE))

    rows = []
    errors = []
    for ticker in UNIVERSE:
        d = price_map.get(ticker)
        if d is None or d.empty:
            errors.append(ticker)
            continue
        try:
            live_d = append_live_bar(d, intraday_map.get(ticker))
            f = add_features(live_d, cfg)
            if len(f) < 220:
                errors.append(ticker)
                continue
            sig = signal_mask(f, selected_mode, cfg)
            r = f.iloc[-1]
            code = ticker.replace(".VN", "")
            sentiment = float(sent_map.get(code, 0.0))
            composite = float(np.clip(r["RESEARCH_SCORE"] + 10.0 * sentiment, 0, 100))
            rows.append(
                {
                    "Ticker": code,
                    "T+0 date": pd.Timestamp(f.index[-1]).date().isoformat(),
                    "Indicative live": float(r["Close"]),
                    "Signal now": bool(sig.iloc[-1]),
                    "Above MA200": bool(r["ABOVE_MA200"]),
                    "EMA20>50": bool(r["EMA_TREND"]),
                    "Cross recent": bool(r["CROSS_RECENT"]),
                    "Vol ratio": float(r["VOL_RATIO"]) if pd.notna(r["VOL_RATIO"]) else np.nan,
                    "RSI14": float(r["RSI14"]) if pd.notna(r["RSI14"]) else np.nan,
                    "Wyckoff": "SOS" if r["WY_SOS"] else ("Spring" if r["WY_SPRING"] else ("LPS" if r["WY_LPS"] else "")),
                    "Technical score": float(r["RESEARCH_SCORE"]),
                    "News sentiment": sentiment,
                    "Composite score": composite,
                    "Source": "Yahoo 1m fallback" if ticker in intraday_map else "Daily fallback",
                }
            )
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    scan = pd.DataFrame(rows)
    if scan.empty:
        st.error("No usable market rows were loaded.")
    else:
        latest = scan["T+0 date"].max()
        coverage = int((scan["T+0 date"] == latest).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe configured", len(UNIVERSE))
        c2.metric("Usable", len(scan))
        c3.metric("T+0 date", latest)
        c4.metric("Coverage latest", f"{coverage}/{len(scan)}")

        st.warning(
            "T+0 is intentionally dynamic. The official historical cohort is frozen only after the daily session "
            "and never changes afterward. Until DNSE is connected, intraday prices are research-only Yahoo fallback."
        )

        signal_rows = scan[scan["Signal now"]].sort_values("Composite score", ascending=False)
        st.markdown(f"#### Current T+0 candidates · {len(signal_rows)}")
        st.dataframe(signal_rows, hide_index=True, width="stretch", height=460)
        if tg_token and tg_chat_id and not signal_rows.empty:
            if st.button("📲 Bắn tín hiệu danh mục T+0 sang Telegram ngay"):
                msg = f"🚨 *HOSE QUANT V2.1.1 — TÍN HIỆU T+0 ({latest})*\n\n"
                for _, s_row in signal_rows.iterrows():
                    msg += f"• *{s_row['Ticker']}* — Giá: `{s_row['Indicative live']:,.0f}`\n"
                    msg += f"  Mẫu hình: `{s_row['Wyckoff'] or 'EMA Cross'}` | Score: `{s_row['Technical score']}`\n"
                    msg += f"  Mục tiêu TP (+15%): `{s_row['Indicative live'] * 1.15:,.0f}` | Cắt lỗ SL (-5%): `{s_row['Indicative live'] * 0.95:,.0f}`\n\n"
                if send_telegram_msg(tg_token, tg_chat_id, msg):
                    st.success("Đã bắn toàn bộ tín hiệu T+0 sang Telegram của bạn!")
                else:
                    st.error("Gửi tin nhắn sang Telegram thất bại.")

        st.markdown("#### Full live-ranked universe")
        st.dataframe(scan.sort_values("Composite score", ascending=False), hide_index=True, width="stretch", height=520)

        if errors:
            st.warning(f"{len(errors)} ticker(s) had missing/invalid data. See Audit tab.")

with tabs[1]:
    st.subheader("Frozen daily cohorts · T+0 to T+10")
    st.caption(
        "Each recommendation day is immutable once frozen. T+1/T+2/... are subsequent trading-session CLOSE prices, "
        "not calendar days. At T+10 each ticker is summarized as WIN/LOSS/FLAT."
    )

    cohorts = load_cohorts()
    marks = load_marks()
    t10 = load_summary()

    if cohorts.empty:
        st.info(
            "No frozen cohort exists yet. The scheduled daily_cohort_worker freezes the first cohort after market close."
        )
    else:
        cohort_dates = sorted(cohorts["cohort_date"].astype(str).unique(), reverse=True)
        selected_cohort = st.selectbox("Recommendation date", cohort_dates)
        cm = cohort_matrix(selected_cohort)

        base = cohorts[cohorts["cohort_date"].astype(str).eq(selected_cohort)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cohort date", selected_cohort)
        c2.metric("Frozen names", len(base))
        c3.metric("Primary mode", str(base["mode"].iloc[0]) if not base.empty else "—")
        done_t10 = int(
            t10["cohort_date"].astype(str).eq(selected_cohort).sum()
        ) if not t10.empty else 0
        c4.metric("T+10 completed", f"{done_t10}/{len(base)}")

        show_cols = ["ticker","reference_price","reference_source","signal_score"]
        for t in [0,1,2,3,5,10]:
            if f"T+{t} %" in cm.columns:
                show_cols.append(f"T+{t} %")
            if f"T+{t} Close" in cm.columns:
                show_cols.append(f"T+{t} Close")

        display = cm[[c for c in show_cols if c in cm.columns]].copy()
        for c in [x for x in display.columns if x.endswith(" %")]:
            display[c] = display[c].map(fmt_pct)
        st.dataframe(display, hide_index=True, width="stretch", height=520)

        if not t10.empty:
            summary = t10[t10["cohort_date"].astype(str).eq(selected_cohort)].copy()
            if not summary.empty:
                st.markdown("#### T+10 final results")
                wins = int((summary["status"] == "WIN").sum())
                losses = int((summary["status"] == "LOSS").sum())
                avg = pd.to_numeric(summary["t10_return_pct"], errors="coerce").mean()
                s1, s2, s3 = st.columns(3)
                s1.metric("Wins", wins)
                s2.metric("Losses", losses)
                s3.metric("Average T+10", "—" if pd.isna(avg) else f"{avg:+.2f}%")
                st.dataframe(summary, hide_index=True, width="stretch")

with tabs[2]:
    st.subheader("Signal detail")
    if "scan" not in locals() or scan.empty:
        st.info("Scanner must load first.")
    else:
        ticker = st.selectbox("Ticker", scan["Ticker"].tolist())
        full_ticker = ticker + ".VN"
        d = price_map.get(full_ticker)
        f = add_features(d, cfg)
        tail = f.tail(120).copy()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=tail.index,
            open=tail["Open"], high=tail["High"], low=tail["Low"], close=tail["Close"],
            name="Price"
        ))
        fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA20"], name="EMA20"))
        fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA50"], name="EMA50"))
        fig.add_trace(go.Scatter(x=tail.index, y=tail["EMA200"], name="EMA200"))
        fig.update_layout(height=620, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

        r = f.iloc[-1]
        info = pd.DataFrame(
            {
                "Field": ["Date","Close","Above MA200","EMA20>EMA50","Cross recent","Volume ratio","RSI14","Wyckoff SOS","Wyckoff Spring","Wyckoff LPS","Research score","News sentiment"],
                "Value": [
                    str(pd.Timestamp(f.index[-1]).date()),
                    f"{float(r['Close']):,.0f}",
                    str(bool(r["ABOVE_MA200"])),
                    str(bool(r["EMA_TREND"])),
                    str(bool(r["CROSS_RECENT"])),
                    f"{float(r['VOL_RATIO']):.2f}" if pd.notna(r["VOL_RATIO"]) else "—",
                    f"{float(r['RSI14']):.1f}" if pd.notna(r["RSI14"]) else "—",
                    str(bool(r["WY_SOS"])),
                    str(bool(r["WY_SPRING"])),
                    str(bool(r["WY_LPS"])),
                    f"{float(r['RESEARCH_SCORE']):.1f}",
                    f"{float(sent_map.get(ticker,0.0)):+.2f}",
                ]
            }
        )
        st.dataframe(info, hide_index=True, width="stretch")

with tabs[3]:
    st.subheader("Real OHLCV trade-level backtest")
    st.warning(
        "This replaces the old synthetic/random backtest. Metrics below are trade-level diagnostics, "
        "not portfolio CAGR. Signal close t → entry next open t+1; fees are included."
    )

    default_bt = ["FPT.VN","HPG.VN","MBB.VN","MWG.VN","SSI.VN","VNM.VN","VPB.VN","VIC.VN","STB.VN","GMD.VN"]
    bt_tickers = st.multiselect("Backtest tickers", UNIVERSE, default=default_bt)

    compare_all = st.checkbox("Compare all 4 variants", value=True)
    if st.button("Run backtest", type="primary"):
        modes = list(MODE_LABELS) if compare_all else [selected_mode]
        metric_rows = []
        all_trades = []

        for mode in modes:
            mode_trades = []
            for t in bt_tickers:
                d = price_map.get(t)
                if d is None or len(d) < 230:
                    continue
                try:
                    tr = backtest_trades(d, mode, cfg)
                    if not tr.empty:
                        tr["ticker"] = t.replace(".VN","")
                        tr["mode"] = mode
                        mode_trades.append(tr)
                except Exception:
                    continue

            combined = pd.concat(mode_trades, ignore_index=True) if mode_trades else pd.DataFrame()
            m = trade_metrics(combined)
            metric_rows.append(
                {
                    "Mode": MODE_LABELS[mode],
                    "Trades": m["trades"],
                    "Win rate": m["win_rate"],
                    "Avg net/trade": m["avg_net_return"],
                    "Median net/trade": m["median_net_return"],
                    "Profit factor": m["profit_factor"],
                }
            )
            if not combined.empty:
                all_trades.append(combined)

        metrics = pd.DataFrame(metric_rows)
        for c in ["Win rate","Avg net/trade","Median net/trade"]:
            metrics[c] = metrics[c].map(lambda x: "—" if pd.isna(x) else f"{100*x:.2f}%")
        metrics["Profit factor"] = metrics["Profit factor"].map(
            lambda x: "—" if pd.isna(x) else ("∞" if np.isinf(x) else f"{x:.2f}")
        )
        st.dataframe(metrics, hide_index=True, width="stretch")

        if all_trades:
            trades = pd.concat(all_trades, ignore_index=True)
            show = trades.sort_values("entry_date", ascending=False).head(300).copy()
            show["net_return"] = show["net_return"].map(lambda x: f"{100*x:.2f}%")
            st.markdown("#### Recent evaluated trades")
            st.dataframe(show, hide_index=True, width="stretch", height=520)

        st.caption(
            "Use this comparison as an ablation test. Do not select the variant with the best in-sample number and call it alpha. "
            "Freeze rules, then validate on a later untouched period."
        )

with tabs[4]:
    st.subheader("News sentiment overlay")
    st.caption(
        "RSS headline/summary baseline. It is intentionally separate from the price trigger so news can be audited "
        "and tested incrementally before it is allowed to change execution."
    )
    if sent.empty:
        st.info("news_sentiment.csv is not present yet. Run news_worker.py or enable the scheduled workflow.")
    else:
        st.dataframe(sent.sort_values(["sentiment_score","article_count"], ascending=[False,False]), hide_index=True, width="stretch")
        st.caption("Baseline sentiment is a lightweight Vietnamese finance lexicon, not a calibrated financial-language model.")

with tabs[5]:
    st.subheader("Audit & methodology")
    audit = [
        ("Data source", "Yahoo Finance fallback, daily bars"),
        ("Universe", f"Curated list: {len(UNIVERSE)} tickers; not the full historical HOSE universe"),
        ("Signal timing", "Close t"),
        ("Paper entry", "Next available session Open t+1"),
        ("Fees", f"Buy {cfg.buy_fee:.2%}, sell {cfg.sell_fee:.2%}"),
        ("Minimum hold", f"{cfg.min_hold_sessions} trading sessions"),
        ("Maximum hold", f"{cfg.max_hold_sessions} trading sessions"),
        ("Stop / TP", f"-{cfg.stop_loss:.0%} / +{cfg.take_profit:.0%}"),
        ("Wyckoff", "Quantified event proxies only; no canonical phase claim"),
        ("Sentiment", "Separate overlay; RSS title/summary lexicon baseline"),
        ("Cohort persistence", "Git-tracked CSV ledgers updated by scheduled worker; historical cohorts immutable"),
        ("Live execution", "Disabled"),
    ]
    st.dataframe(pd.DataFrame(audit, columns=["Field","Value"]), hide_index=True, width="stretch")

    if "errors" in locals() and errors:
        st.markdown("#### Data/load errors")
        st.code("\n".join(map(str, errors[:100])))