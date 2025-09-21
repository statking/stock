# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from typing import Optional
import time

st.set_page_config(page_title="ETF 비교 대시보드", layout="wide")
st.title("코스피 대비 수익률 비교 대시보드")

# --- 티커 / 라벨 ---
TICKERS = ['KS11','KQ11','244580','091230','305540','266390','139280','117700','228790',
           '228800','138520','138540','138530','307520','161510','117460','139230','091220',
           '153130','183700','445290','385510','261220','411060','379800','379810','305080',
           '464470','465580','458730','494670','449450']

STOCKS = ['코스피','코스닥','바이오','반도체','2차전지','경기소비재','경기방어','건설','화장품',
          '여행레저','삼성그룹','현대그룹','LG그룹','지주회사','고배당주','화학','중공업','은행',
          '단기채','채권혼합','로봇','신재생','원유선물','금현물','S&P500','나스닥100','미국채10',
          '미국채30액티브','빅테크7','타미당','조선','방산']

# --- 사용자 입력 ---
c1, c2 = st.columns(2)
ref_date1 = c1.date_input("기준일 1", pd.to_datetime("2024-07-11")).strftime("%Y-%m-%d")
ref_date2 = c2.date_input("기준일 2", pd.to_datetime("2025-04-09")).strftime("%Y-%m-%d")

# 컬럼명(중복 방지)
col1_name = f"Change since {ref_date1} (%)"
col2_base = f"Change since {ref_date2} (%)"
col2_name = col2_base if col2_base != col1_name else f"{col2_base} (2)"

min_ref = min(pd.to_datetime(ref_date1), pd.to_datetime(ref_date2))
start = (min_ref - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

# --- 유틸 ---
_YF_MAP = {
    # 지수(필요 시 확장)
    'KS11': '^KS11',  # KOSPI
    'KQ11': '^KQ11',  # KOSDAQ
}

def to_yf_symbol(ticker: str) -> str:
    """우리 코드 체계 → yfinance 심볼로 변환"""
    if ticker in _YF_MAP:
        return _YF_MAP[ticker]
    # 숫자 6자리면 KRX 종목으로 가정 → .KS 붙임 (대부분 ETF/주식이 KOSPI 상장)
    if ticker.isdigit() and len(ticker) == 6:
        return f"{ticker}.KS"
    # 그 외는 입력 그대로 시도
    return ticker

def safe_read(ticker: str, start: str, retry: int = 1, wait: float = 1.0) -> pd.DataFrame:
    sym = to_yf_symbol(ticker)
    last_exc: Optional[Exception] = None
    for i in range(retry + 1):
        try:
            df = yf.download(sym, start=start, progress=False, auto_adjust=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # yfinance 멀티컬럼 방지 및 표준화
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[-1] for c in df.columns]  # ('Adj Close','') -> 'Adj Close'
                # Close 없으면 Adj Close로 대체
                if 'Close' not in df.columns and 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                # 인덱스 보정
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                df = df.sort_index()
                return df
            else:
                last_exc = ValueError("Empty dataframe")
        except Exception as e:
            last_exc = e
        if i < retry:
            time.sleep(wait)
    # 실패 시 빈 DF
    return pd.DataFrame()

def close_on_or_before(df: pd.DataFrame, date_str: str) -> Optional[float]:
    if df.empty or 'Close' not in df.columns:
        return None
    target = pd.to_datetime(date_str)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()
    mask = df.index <= target
    if not mask.any():
        return None
    try:
        return float(df.loc[mask, 'Close'].iloc[-1])
    except Exception:
        return None

# --- 데이터 적재 (캐시) ---
@st.cache_data(show_spinner=True, ttl=3600)
def load_changes(start: str,
                 ref_date1: str,
                 ref_date2: str,
                 tickers: tuple[str, ...],
                 stocks: tuple[str, ...],
                 col1_name: str,
                 col2_name: str) -> pd.DataFrame:
    rows = []
    for name, tic in zip(stocks, tickers):
        data = safe_read(tic, start, retry=1, wait=1.0)
        last_close = float(data['Close'].iloc[-1]) if (not data.empty and 'Close' in data.columns) else None
        p1 = close_on_or_before(data, ref_date1)
        p2 = close_on_or_before(data, ref_date2)

        chg1 = round((last_close / p1 - 1) * 100, 2) if (last_close is not None and p1 not in (None, 0)) else None
        chg2 = round((last_close / p2 - 1) * 100, 2) if (last_close is not None and p2 not in (None, 0)) else None
        rows.append([name, tic, chg1, chg2])

    df = pd.DataFrame(rows, columns=['Stock','Ticker', col1_name, col2_name])
    return df

df = load_changes(start, ref_date1, ref_date2, tuple(TICKERS), tuple(STOCKS), col1_name, col2_name)

def assign_colors(sorted_df: pd.DataFrame, col: str) -> list[str]:
    kospi_ratio = None
    try:
        kospi_ratio = float(sorted_df.loc[sorted_df['Ticker'] == 'KS11', col].iloc[0])
    except Exception:
        pass
    colors = []
    for t, r in zip(sorted_df['Ticker'], sorted_df[col]):
        if t == 'KS11':
            colors.append('black')            # 코스피는 항상 검정
        elif r is None or pd.isna(r) or kospi_ratio is None:
            colors.append('gray')             # 비교불가 → 회색
        elif r > kospi_ratio:
            colors.append('red')              # 코스피보다 좋음
        else:
            colors.append('blue')             # 코스피보다 나쁨/같음
    return colors

def bar_fig(df_in: pd.DataFrame, col: str, title: str) -> go.Figure:
    # 방어: 컬럼 존재 확인
    if col not in df_in.columns:
        st.error(f"필요한 컬럼이 없습니다: '{col}'. 페이지를 다시 실행해 주세요.")
        return go.Figure()

    # 숫자 변환(정렬 오류 방지)
    s = pd.to_numeric(df_in[col], errors='coerce')

    ordered = (
        pd.DataFrame({
            'Stock': df_in['Stock'],
            'Ticker': df_in['Ticker'],
            col: s
        })
        .sort_values(by=col, ascending=True, na_position='first')
        .reset_index(drop=True)
    )
    colors = assign_colors(ordered, col)
    fig = go.Figure([go.Bar(x=ordered['Stock'], y=ordered[col], marker_color=colors)])
    fig.update_layout(title=title, xaxis_title="Stock", yaxis_title="Change (%)",
                      showlegend=False, height=600, bargap=0.2)
    return fig

# --- 출력 ---
left, right = st.columns(2)
with left:
    st.subheader(f"기준일 1: {ref_date1}")
    st.plotly_chart(bar_fig(df, col1_name, f"Stock Changes since {ref_date1}"), use_container_width=True)
with right:
    st.subheader(f"기준일 2: {ref_date2}")
    st.plotly_chart(bar_fig(df, col2_name, f"Stock Changes since {ref_date2}"), use_container_width=True)

st.markdown("### 데이터 테이블")
st.dataframe(df, use_container_width=True)
