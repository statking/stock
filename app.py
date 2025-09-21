# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from typing import Optional
import time

st.set_page_config(page_title="ETF 비교 대시보드 (yfinance)", layout="wide")
st.title("코스피 대비 수익률 비교 대시보드 (yfinance)")

# --- 티커 / 라벨 ---
TICKERS = ['KS11','KQ11','244580','091230','305540','266390','139280','117700','228790',
           '228800','138520','138540','138530','307520','161510','117460','139230','091220',
           '153130','183700','445290','401470','385510','261220','139320','379800','379810','305080',
           '464470','465580','458730','458760']

STOCKS = ['코스피','코스닥','바이오','반도체','2차전지','경기소비재','경기방어','건설','화장품',
          '여행레저','삼성그룹','현대그룹','LG그룹','지주회사','고배당주','화학','중공업','은행',
          '단기채','채권혼합','로봇','메타버스','신재생','원유선물','금은선물','S&P500','나스닥100','미국채10',
          '미국채30액티브','빅테크7','타미당','타미당7%']

# --- 사용자 입력 ---
c1, c2 = st.columns(2)
ref_date1 = c1.date_input("기준일 1", pd.to_datetime("2024-07-11")).strftime("%Y-%m-%d")
ref_date2 = c2.date_input("기준일 2", pd.to_datetime("2025-04-09")).strftime("%Y-%m-%d")

min_ref = min(pd.to_datetime(ref_date1), pd.to_datetime(ref_date2))
start = (min_ref - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

# --- Yahoo 티커 변환 ---
def to_yahoo_symbol(raw: str) -> list[str]:
    """
    입력 티커를 Yahoo Finance 심볼 후보 리스트로 변환.
    - KS11, KQ11 → ^KS11, ^KQ11 (지수)
    - 숫자형 → .KS 먼저, 실패시 .KQ 폴백
    - 그 외 → 그대로 사용
    """
    if raw == 'KS11':
        return ['^KS11']
    if raw == 'KQ11':
        return ['^KQ11']
    if raw.isdigit():
        return [f"{raw}.KS", f"{raw}.KQ"]
    return [raw]

# --- 데이터 로드 유틸 ---
def _choose_price_column(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ['Adj Close', 'Close', 'adjclose', 'close']:
        if col in df.columns:
            return df[col]
    return None

def yf_read_one(raw_ticker: str, start_date: str) -> pd.DataFrame:
    """
    yfinance로 단일 티커를 로드.
    후보 심볼을 순차 시도하며, 성공 시 표준화된 DF 반환(index=Datetime, columns=['Close']).
    실패 시 빈 DF 반환.
    """
    for sym in to_yahoo_symbol(raw_ticker):
        try:
            df = yf.download(sym, start=start_date, progress=False, auto_adjust=False)
            # 일부 환경서 멀티인덱스 컬럼으로 내려오면 단순화
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join([str(c) for c in col if c]) for col in df.columns]
            price = _choose_price_column(df)
            if price is not None and not price.dropna().empty:
                out = pd.DataFrame({'Close': price})
                out.index = pd.to_datetime(out.index)
                out.sort_index(inplace=True)
                return out
        except Exception:
            pass
    return pd.DataFrame()

def close_on_or_before(df: pd.DataFrame, date_str: str) -> Optional[float]:
    """주어진 날짜(휴장 포함)의 직전 거래일 종가 반환 (없으면 None)."""
    if df.empty or 'Close' not in df.columns:
        return None
    target = pd.to_datetime(date_str)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    mask = df.index <= target
    if not mask.any():
        return None
    val = df.loc[mask, 'Close'].iloc[-1]
    return float(val) if pd.notna(val) else None

# --- 캐시: 날짜/시작일/티커/라벨을 포함해 캐시키 분리 ---
@st.cache_data(show_spinner=True, ttl=3600)
def load_changes(start: str,
                 ref_date1: str,
                 ref_date2: str,
                 tickers: tuple[str, ...],
                 stocks: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for name, tic in zip(stocks, tickers):
        data = yf_read_one(tic, start)
        last_close = float(data['Close'].iloc[-1]) if (not data.empty and 'Close' in data.columns) else None
        p1 = close_on_or_before(data, ref_date1)
        p2 = close_on_or_before(data, ref_date2)

        chg1 = round((last_close / p1 - 1) * 100, 2) if (last_close is not None and p1 not in (None, 0)) else None
        chg2 = round((last_close / p2 - 1) * 100, 2) if (last_close is not None and p2 not in (None, 0)) else None
        rows.append([name, tic, chg1, chg2])

    col1 = f"Change since {ref_date1} (%)"
    col2 = f"Change since {ref_date2} (%)"
    return pd.DataFrame(rows, columns=['Stock','Ticker', col1, col2])

df = load_changes(start, ref_date1, ref_date2, tuple(TICKERS), tuple(STOCKS))

# --- 색상/차트 ---
def assign_colors(sorted_df: pd.DataFrame, col: str) -> list[str]:
    try:
        kospi_ratio = float(sorted_df.loc[sorted_df['Ticker'] == 'KS11', col].iloc[0])
    except Exception:
        kospi_ratio = None
    colors = []
    for t, r in zip(sorted_df['Ticker'], sorted_df[col]):
        if t == 'KS11':
            colors.append('black')            # 코스피는 항상 검정
        elif r is None or kospi_ratio is None:
            colors.append('gray')             # 비교불가 → 회색
        elif r > kospi_ratio:
            colors.append('red')              # 코스피보다 좋음
        else:
            colors.append('blue')             # 코스피보다 나쁨/같음
    return colors

def bar_fig(df_in: pd.DataFrame, col: str, title: str) -> go.Figure:
    if col not in df_in.columns:
        st.error(f"필요한 컬럼이 없습니다: '{col}'. 페이지를 다시 실행해 주세요.")
        return go.Figure()
    ordered = (
        df_in[['Stock','Ticker', col]]
        .sort_values(by=col, ascending=True, na_position='first')
        .reset_index(drop=True)
    )
    colors = assign_colors(ordered, col)
    fig = go.Figure([go.Bar(x=ordered['Stock'], y=ordered[col], marker_color=colors)])
    fig.update_layout(title=title, xaxis_title="Stock", yaxis_title="Change (%)",
                      showlegend=False, height=600, bargap=0.2)
    return fig

# --- 출력 ---
col1_name = f"Change since {ref_date1} (%)"
col2_name = f"Change since {ref_date2} (%)"

left, right = st.columns(2)
with left:
    st.subheader(f"기준일 1: {ref_date1}")
    st.plotly_chart(bar_fig(df, col1_name, f"Stock Changes since {ref_date1}"), use_container_width=True)
with right:
    st.subheader(f"기준일 2: {ref_date2}")
    st.plotly_chart(bar_fig(df, col2_name, f"Stock Changes since {ref_date2}"), use_container_width=True)

st.markdown("### 데이터 테이블")
st.dataframe(df, use_container_width=True)
