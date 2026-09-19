
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import re
import requests
import feedparser
import pandas as pd

OUT = Path("news_sentiment.csv")

FEEDS = {
    "CafeF - Chứng khoán": "https://cafef.vn/thi-truong-chung-khoan.rss",
    "CafeF - Doanh nghiệp": "https://cafef.vn/doanh-nghiep.rss",
    "CafeF - Vĩ mô": "https://cafef.vn/vi-mo-dau-tu.rss",
    "VnExpress - Kinh doanh": "https://vnexpress.net/rss/kinh-doanh.rss",
}

TICKERS = [
    "VCB","BID","CTG","TCB","MBB","ACB","STB","HDB","VPB","TPB","VIB","LPB","MSB","OCB","SHB",
    "SSI","VCI","VND","HCM","FTS","BSI","CTS","HPG","HSG","NKG","VHM","VIC","VRE","PDR","DIG","DXG",
    "KDH","NLG","FCN","KBC","VCG","FPT","MWG","PNJ","DGW","FRT","CTR","VNM","MSN","SAB","PAN","GAS",
    "PLX","POW","GVR","PVD","NT2","PC1","REE","DPM","DCM","CSV","PHR","VHC","ANV","FMC","HAG","DBC",
    "VJC","GMD","HAH","SCS","BVH"
]

ALIASES = {
    "FPT": ["fpt"],
    "VPB": ["vpbank", "vp bank"],
    "MBB": ["mbbank", "mb bank"],
    "TCB": ["techcombank"],
    "ACB": ["ngân hàng á châu"],
    "HPG": ["hòa phát", "hoa phat"],
    "VNM": ["vinamilk"],
    "MWG": ["thế giới di động", "the gioi di dong"],
    "PNJ": ["phú nhuận", "phu nhuan"],
    "VIC": ["vingroup"],
    "VHM": ["vinhomes"],
    "VRE": ["vincom retail"],
    "FRT": ["fpt retail"],
    "SSI": ["chứng khoán ssi", "chung khoan ssi"],
    "VND": ["vndirect"],
}

POSITIVE = {
    "lợi nhuận tăng": 2.0,
    "tăng trưởng": 1.0,
    "vượt kế hoạch": 1.5,
    "kỷ lục": 1.0,
    "chia cổ tức": 0.8,
    "trúng thầu": 1.2,
    "ký hợp đồng": 0.8,
    "được chấp thuận": 0.8,
    "mở rộng": 0.5,
    "mua lại cổ phiếu": 1.0,
    "nâng hạng": 1.0,
    "giảm nợ": 0.8,
}

NEGATIVE = {
    "thua lỗ": -2.0,
    "báo lỗ": -2.0,
    "lợi nhuận giảm": -1.5,
    "bị phạt": -1.2,
    "điều tra": -1.8,
    "khởi tố": -2.2,
    "hủy niêm yết": -2.5,
    "nợ xấu": -1.2,
    "chậm trả": -1.5,
    "cảnh báo": -0.8,
    "ngoại trừ": -1.0,
    "pha loãng": -0.8,
    "giảm mạnh": -1.0,
}


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def article_score(text: str) -> float:
    s = text.lower()
    raw = sum(w for k, w in POSITIVE.items() if k in s) + sum(w for k, w in NEGATIVE.items() if k in s)
    # Squash to a stable [-1, 1] range.
    return max(-1.0, min(1.0, raw / 3.0))


def map_tickers(text: str) -> list[str]:
    low = text.lower()
    hits = set()
    for t in TICKERS:
        if re.search(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])", text.upper()):
            hits.add(t)
        for a in ALIASES.get(t, []):
            if a in low:
                hits.add(t)
    return sorted(hits)


def fetch_feed(name: str, url: str) -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0 HOSE-Quant-Research/2.0"}
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    rows = []
    for e in feed.entries:
        title = clean_html(getattr(e, "title", ""))
        summary = clean_html(getattr(e, "summary", ""))
        text = f"{title}. {summary}"
        tickers = map_tickers(text)
        if not tickers:
            continue
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        for t in tickers:
            rows.append(
                {
                    "ticker": t,
                    "source": name,
                    "title": title,
                    "url": getattr(e, "link", ""),
                    "published_raw": published,
                    "article_score": article_score(text),
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
    return rows


def main():
    rows = []
    errors = []
    for name, url in FEEDS.items():
        try:
            rows.extend(fetch_feed(name, url))
        except Exception as e:
            errors.append(f"{name}: {e}")

    raw = pd.DataFrame(rows)
    if raw.empty:
        pd.DataFrame(
            columns=["ticker","sentiment_score","article_count","positive_count","negative_count","last_title","last_url","updated_at_utc"]
        ).to_csv(OUT, index=False)
        print("No ticker-mapped RSS articles found.")
        if errors:
            print("Feed errors:", " | ".join(errors))
        return

    raw = raw.drop_duplicates(subset=["ticker", "url"], keep="last")
    grouped = []
    for ticker, g in raw.groupby("ticker"):
        g = g.copy()
        grouped.append(
            {
                "ticker": ticker,
                "sentiment_score": float(g["article_score"].mean()),
                "article_count": int(len(g)),
                "positive_count": int((g["article_score"] > 0).sum()),
                "negative_count": int((g["article_score"] < 0).sum()),
                "last_title": str(g.iloc[-1]["title"]),
                "last_url": str(g.iloc[-1]["url"]),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    out = pd.DataFrame(grouped).sort_values(["sentiment_score", "article_count"], ascending=[False, False])
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} ticker sentiment rows to {OUT}")
    if errors:
        print("Feed errors:", " | ".join(errors))


if __name__ == "__main__":
    main()
