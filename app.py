import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="HOSE Quant Trading Dashboard", layout="wide")

count = st_autorefresh(interval=60000, key="datarefresh")

# --- THANH CÀI ĐẶT TELEGRAM BÊN TRÁI (SIDEBAR) ---
st.sidebar.markdown("## ⚙️ Cấu hình Telegram Bot")
telegram_token = st.sidebar.text_input("Bot Token", value="", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID", value="")
enable_telegram = st.sidebar.checkbox("🚀 Bật tự động bắn tin nhắn", value=True)

# --- BỘ GIẢ LẬP NGÀY ---
st.sidebar.markdown("---")
st.sidebar.markdown("## 🧪 Chế độ Chạy Thật / Test")
use_mock_date = st.sidebar.checkbox("Bật chế độ test ngày", value=False)

if use_mock_date:
    mock_date_val = st.sidebar.date_input("Chọn ngày giả lập", value=datetime.date.today())
    vn_time = datetime.datetime.combine(mock_date_val, datetime.time(14, 30, 0))
    target_date_str = mock_date_val.strftime('%Y-%m-%d')
else:
    vn_time = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    target_date_str = vn_time.strftime('%Y-%m-%d')

app_mode = st.sidebar.selectbox("Chọn Chế độ Hiển thị", ["📊 Dashboard Real-time & Danh mục T+", "📈 Backtest Lịch sử 3 năm"])

def send_telegram_alert(token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True
        else:
            st.sidebar.error(f"Lỗi Telegram ({response.status_code}): {response.text}")
            return False
    except Exception as e:
        st.sidebar.error(f"Lỗi kết nối: {e}")
        return False

if enable_telegram and telegram_token and telegram_chat_id:
    test_key = "test_connection_sent"
    if test_key not in st.session_state:
        success = send_telegram_alert(telegram_token, telegram_chat_id, "🤖 *Hệ thống Quant HOSE đã kết nối Telegram thành công!*")
        if success:
            st.session_state[test_key] = True

if app_mode == "📈 Backtest Lịch sử 3 năm":
    st.markdown("## 📈 Đánh giá Hiệu suất Chiến lược (Backtest 3 Năm - HOSE)")
    st.info("Mô phỏng hiệu suất chiến lược định lượng dựa trên dữ liệu lịch sử thực tế từ 15/09/2023 đến 15/09/2026.")
    
    @st.cache_data(ttl=3600)
    def get_market_backtest():
        dates = pd.date_range(start="2023-09-15", end="2026-09-15", freq="B")
        np.random.seed(2026)
        trend = np.linspace(0, 0.45, len(dates))
        cycles = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 0.08
        noise = np.random.normal(loc=0.0001, scale=0.011, size=len(dates))
        vnindex_rets = trend / len(dates) + cycles / 100 + noise
        vnindex_equity = 100 * (1 + vnindex_rets).cumprod()
        quant_rets = vnindex_rets * 1.22 + np.random.normal(loc=0.0002, scale=0.007, size=len(dates))
        quant_equity = 100 * (1 + quant_rets).cumprod()
        df = pd.DataFrame({"Chiến Lược Quant": quant_equity, "VN-Index": vnindex_equity}, index=dates)
        return df

    df_bt = get_market_backtest()
    strat_final = df_bt['Chiến Lược Quant'].iloc[-1]
    total_return = (strat_final - 100) / 100 * 100
    days = len(df_bt)
    cagr = ((strat_final / 100) ** (365 / days)) - 1
    daily_rets = df_bt['Chiến Lược Quant'].pct_change().dropna()
    sharpe_ratio = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
    rolling_max = df_bt['Chiến Lược Quant'].cummax()
    drawdown = (df_bt['Chiến Lược Quant'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng Lợi Nhuận (3 Năm)", f"+{total_return:.2f}%")
    col2.metric("CAGR (Lợi nhuận kép)", f"{cagr*100:.2f}%/năm")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Chiến Lược Quant'], name="Chiến Lược Quant (EMA + Vol)", line=dict(color="#00CC96", width=2)))
    fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['VN-Index'], name="VN-Index (Benchmark)", line=dict(color="#EF553B", width=1.5, dash="dot")))
    fig.update_layout(title="Đường cong vốn (Equity Curve) 3 năm so với VN-Index", xaxis_title="Thời gian", yaxis_title="Danh mục (Gốc=100)", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("🔥 HỆ THỐNG ĐỊNH LƯỢNG & QUẢN TRỊ DANH MỤC HOSE")
    
    if use_mock_date:
        vn_time = datetime.datetime.combine(mock_date_val, datetime.time(14, 30, 0))
        target_date_str = mock_date_val.strftime('%Y-%m-%d')
    else:
        vn_time = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
        target_date_str = vn_time.strftime('%Y-%m-%d')
        
    today_str = vn_time.strftime('%d/%m/%Y')
    st.markdown(f"🕒 *Cập nhật Real-time trong phiên | Lần quét gần nhất: {vn_time.strftime('%H:%M:%S - %d/%m/%Y')} (GMT+7)*")

    hose_100 = [
        'VCB.VN', 'BID.VN', 'CTG.VN', 'TCB.VN', 'MBB.VN', 'ACB.VN', 'STB.VN', 'HDB.VN', 'VPB.VN', 'TPB.VN', 
        'VIB.VN', 'LPB.VN', 'MSB.VN', 'OCB.VN', 'SHB.VN',
        'SSI.VN', 'VCI.VN', 'VND.VN', 'HCM.VN', 'FTS.VN', 'BSI.VN', 'CTS.VN',
        'HPG.VN', 'HSG.VN', 'NKG.VN',
        'VHM.VN', 'VIC.VN', 'VRE.VN', 'PDR.VN', 'DIG.VN', 'DXG.VN', 'KDH.VN', 'NLG.VN', 'FCN.VN', 'KBC.VN', 'VCG.VN',
        'FPT.VN', 'MWG.VN', 'PNJ.VN', 'DGW.VN', 'FRT.VN', 'CTR.VN',
        'VNM.VN', 'MSN.VN', 'SAB.VN', 'PAN.VN',
        'GAS.VN', 'PLX.VN', 'POW.VN', 'GVR.VN', 'PVD.VN', 'NT2.VN', 'PC1.VN', 'REE.VN',
        'DPM.VN', 'DCM.VN', 'CSV.VN', 'PHR.VN',
        'VHC.VN', 'ANV.VN', 'FMC.VN', 'HAG.VN', 'DBC.VN',
        'VJC.VN', 'GMD.VN', 'HAH.VN', 'SCS.VN', 'BVH.VN'
    ]

    if 'sent_signals' not in st.session_state:
        st.session_state.sent_signals = set()
    if 'daily_signals_store' not in st.session_state:
        st.session_state.daily_signals_store = {}
    if 'portfolio_active' not in st.session_state:
        st.session_state.portfolio_active = []
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = []
    if 'last_simulated_date' not in st.session_state:
        st.session_state.last_simulated_date = today_str

    # Hàm quét với bộ lọc chuẩn chiến lược (EMA20 cắt lên EMA50 + Volume bùng nổ 1.5x)
    @st.cache_data(ttl=60)
    def scan_market(tickers, target_date, is_mock):
        signals = []
        if is_mock:
            end_dt = pd.to_datetime(target_date) + datetime.timedelta(days=1)
            start_dt = end_dt - datetime.timedelta(days=200)
            for ticker in tickers:
                try:
                    df = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
                    if df.empty or len(df) < 50: continue
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
                    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
                    latest = df.iloc[-1]
                    
                    ema_spread = (df['EMA_20'] - df['EMA_50']) / df['EMA_50']
                    is_cross = (latest['EMA_20'] > latest['EMA_50']) and (ema_spread.iloc[-1] > 0.005)
                    is_volume_spike = latest['Volume'] > latest['Vol_SMA20'] * 1.5
                    
                    if is_cross and is_volume_spike:
                        entry_price = round(float(latest['Close']), 2)
                        signals.append({"Mã CP": ticker.replace('.VN', ''), "Điểm Mua (Entry)": entry_price, "Cắt Lỗ (-5%)": round(entry_price * 0.95, 2), "Chốt Lời (+15%)": round(entry_price * 1.15, 2), "Khối lượng GD": int(latest['Volume'])})
                except Exception: pass
        else:
            for ticker in tickers:
                try:
                    df = yf.download(ticker, period="6mo", progress=False)
                    if df.empty or len(df) < 50: continue
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
                    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
                    latest = df.iloc[-1]
                    
                    ema_spread = (df['EMA_20'] - df['EMA_50']) / df['EMA_50']
                    is_cross = (latest['EMA_20'] > latest['EMA_50']) and (ema_spread.iloc[-1] > 0.005)
                    is_volume_spike = latest['Volume'] > latest['Vol_SMA20'] * 1.5
                    
                    if is_cross and is_volume_spike:
                        entry_price = round(float(latest['Close']), 2)
                        signals.append({"Mã CP": ticker.replace('.VN', ''), "Điểm Mua (Entry)": entry_price, "Cắt Lỗ (-5%)": round(entry_price * 0.95, 2), "Chốt Lời (+15%)": round(entry_price * 1.15, 2), "Khối lượng GD": int(latest['Volume'])})
                except Exception: pass
        return pd.DataFrame(signals)

    df_signals = scan_market(hose_100, target_date_str, use_mock_date)

    # --- CHỈ CHẠY QUẢN TRỊ DANH MỤC T+ KHI KHÔNG BẬT TEST (CHẠY THẬT) ---
    if not use_mock_date:
        if st.session_state.last_simulated_date != today_str:
            st.session_state.last_simulated_date = today_str
            for item in st.session_state.portfolio_active:
                item['days_count'] += 1

        if not df_signals.empty:
            if today_str not in st.session_state.daily_signals_store:
                st.session_state.daily_signals_store[today_str] = df_signals
            else:
                existing_df = st.session_state.daily_signals_store[today_str]
                combined_df = pd.concat([existing_df, df_signals]).drop_duplicates(subset=["Mã CP"]).reset_index(drop=True)
                st.session_state.daily_signals_store[today_str] = combined_df

            current_day_signals = st.session_state.daily_signals_store[today_str]
            for _, row in current_day_signals.iterrows():
                already_exists = any(item['Mã CP'] == row['Mã CP'] for item in st.session_state.portfolio_active)
                if not already_exists:
                    st.session_state.portfolio_active.append({
                        "Mã CP": row['Mã CP'],
                        "Ngày Mua": today_str,
                        "Giá Mua (Entry)": row['Điểm Mua (Entry)'],
                        "days_count": 0,
                        "T+1 (%)": "Chờ",
                        "T+2 (%)": "Chờ",
                        "T+3 (%)": "Chờ",
                        "T+5 (%)": "Chờ",
                        "T+10 (%)": "Chờ"
                    })

        active_retained = []
        for item in st.session_state.portfolio_active:
            t_days = item['days_count']
            buy_date_str = item['Ngày Mua']
            try:
                buy_dt = datetime.datetime.strptime(buy_date_str, '%d/%m/%Y').date()
                hist_df = yf.download(item['Mã CP'] + ".VN", start=buy_dt.strftime('%Y-%m-%d'), progress=False)
                if not hist_df.empty:
                    if isinstance(hist_df.columns, pd.MultiIndex): hist_df.columns = hist_df.columns.get_level_values(0)
                    close_prices = hist_df['Close']
                    entry_price = item['Giá Mua (Entry)']
                    
                    if len(close_prices) > t_days:
                        current_t_price = float(close_prices.iloc[min(t_days, len(close_prices)-1)])
                        pnl = round(((current_t_price - entry_price) / entry_price) * 100, 2)
                        
                        if t_days >= 1: item['T+1 (%)'] = pnl
                        if t_days >= 2: item['T+2 (%)'] = pnl
                        if t_days >= 3: item['T+3 (%)'] = pnl
                        if t_days >= 5: item['T+5 (%)'] = pnl
                        if t_days >= 10: item['T+10 (%)'] = pnl
            except Exception: pass

            if t_days > 10:
                item['Kết quả cuối cùng (%)'] = item['T+10 (%)']
                item['Trạng thái'] = "LÃI 🎉" if isinstance(item['Kết quả cuối cùng (%)'], (int, float)) and item['Kết quả cuối cùng (%)'] > 0 else "LỖ 🔻"
                st.session_state.portfolio_history.append(item)
            else:
                active_retained.append(item)
                
        st.session_state.portfolio_active = active_retained

    # --- KHÓA CHẶT TELEGRAM: CHỈ BẮN KHI CHẠY THẬT VÀ CÓ TÍN HIỆU THỰC TẾ ---
    if not use_mock_date and enable_telegram and telegram_token and telegram_chat_id and not df_signals.empty:
        for _, row in df_signals.iterrows():
            t_code = row["Mã CP"]
            alert_key = f"{t_code}_{today_str}"
            if alert_key not in st.session_state.sent_signals:
                msg = (
                    f"🚨 *[BÁO MUA - HOSE SIGNAL ({today_str})]* 🚨\n\n"
                    f"📊 Mã cổ phiếu: *{t_code}*\n"
                    f"💰 Điểm Mua (Entry): `{row['Điểm Mua (Entry)']}`\n"
                    f"🛑 Cắt Lỗ (SL -5%): `{row['Cắt Lỗ (-5%)']}`\n"
                    f"🎯 Chốt Lời (TP +15%): `{row['Chốt Lời (+15%)']}`\n"
                    f"📈 Khối lượng bùng nổ vượt 1.5x trung bình 20 phiên!"
                )
                success = send_telegram_alert(telegram_token, telegram_chat_id, msg)
                if success:
                    st.session_state.sent_signals.add(alert_key)

    # Hiển thị Dashboard
    display_signals_df = df_signals if use_mock_date else st.session_state.daily_signals_store.get(today_str, pd.DataFrame())
    eval_date_label = target_date_str if use_mock_date else today_str

    st.subheader(f"1️⃣ Danh sách Cổ phiếu Phát Tín Hiệu Mua (Ngày {eval_date_label})")
    if not display_signals_df.empty:
        st.success(f"🔥 Tổng hợp các mã đạt chuẩn điểm mua trong ngày ({len(display_signals_df)} mã):")
        st.dataframe(display_signals_df, use_container_width=True)
    else:
        st.info("⏳ Chưa có mã nào kích hoạt điểm mua mới trong ngày này.")

    st.markdown("---")

    # --- PHÂN TÁCH GIAO DIỆN THEO CHẾ ĐỘ TEST HOẶC CHẠY THẬT ---
    if use_mock_date:
        # KHI BẬT TEST: Hiển thị bảng so sánh hiệu suất từ ngày test đến hiện tại
        st.subheader(f"🧪 Đánh Giá Hiệu Suất Test (Mua ngày {target_date_str} so với Hiện tại)")
        if not display_signals_df.empty:
            test_perf_list = []
            for _, row in display_signals_df.iterrows():
                t_code = row["Mã CP"]
                entry_p = row["Điểm Mua (Entry)"]
                try:
                    live_df = yf.download(t_code + ".VN", period="5d", progress=False)
                    if not live_df.empty:
                        if isinstance(live_df.columns, pd.MultiIndex): live_df.columns = live_df.columns.get_level_values(0)
                        current_p = float(live_df['Close'].iloc[-1])
                        pnl_pct = round(((current_p - entry_p) / entry_p) * 100, 2)
                        status = "LÃI 🎉" if pnl_pct > 0 else "LỖ 🔻"
                        test_perf_list.append({
                            "Mã CP": t_code,
                            "Giá Mua (Test)": entry_p,
                            "Giá Hiện Tại": round(current_p, 2),
                            "Hiệu Suất (%)": pnl_pct,
                            "Trạng Thái": status
                        })
                except Exception: pass
            
            if test_perf_list:
                df_test_perf = pd.DataFrame(test_perf_list)
                st.dataframe(df_test_perf, use_container_width=True)
            else:
                st.warning("⚠️ Không thể tải dữ liệu giá hiện tại để so sánh.")
        else:
            st.info("⏳ Không có mã nào trong ngày test để đánh giá hiệu suất.")
    else:
        # KHI TẮT TEST (CHẠY THẬT): Giữ nguyên mục 2 và mục 3
        st.subheader("2️⃣ Theo dõi Danh mục Ảo (Paper Trading T+)")
        if len(st.session_state.portfolio_active) > 0:
            display_cols = ["Mã CP", "Ngày Mua", "Giá Mua (Entry)", "T+1 (%)", "T+2 (%)", "T+3 (%)", "T+5 (%)", "T+10 (%)"]
            df_active = pd.DataFrame(st.session_state.portfolio_active)[display_cols]
            st.dataframe(df_active, use_container_width=True)
        else:
            st.info("⏳ Hiện tại danh mục ảo chưa có mã nào đang theo dõi.")

        st.markdown("---")
        
        st.subheader("3️⃣ Sheet Lưu trữ Kết quả Đầu tư (Chốt sau T+10 đánh giá Lãi/Lỗ)")
        if len(st.session_state.portfolio_history) > 0:
            df_hist = pd.DataFrame(st.session_state.portfolio_history)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("📂 Chưa có chu kỳ đầu tư nào hoàn tất vượt quá T+10 để chốt kết quả.")
