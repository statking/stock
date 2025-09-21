# app.py
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from typing import Optional

st.set_page_config(page_title="ETF 비교 대시보드", layout="wide")

st.title("코스피 대비 수익률 비교 대시보드")

# --- 사용자 입력 ---
col1, col2 = st.columns(2)
ref_date1 = col1.date_input("기준일 1", pd.to_datetime("2024-07-11")).strftime("%Y-%m-%d")
ref_date2 = col2.date_input("기준일 2", pd.to_datetime("2025-04-09")).strftime("%Y-%m-%d")

min_ref = min(pd.to_datetime(ref_date1), pd.to_datetime(ref_date2))
start = (min_ref - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

tickers = ['KS11','KQ11','244580','091230','305540','266390','139280','117700','228790',
           '228800','138520','138540','138530','307520','161510','117460','139230','091220',
           '153130','183700','445290','401470','385510','261220','139320','379800','379810','305080',
           '464470','465580','458730','458760']

stocks = ['코스피','코스닥','바이오','반도체','2차전지','경기소비재','경기방어','건설','화장품',
          '여행레저','삼성그룹','현대그룹','LG그룹','지주회사','고배당주','화학','중공업','은행',
          '단기채','채권혼합','로봇','메타버스','신재생','원유선물','금은선물','S&P500','나스닥100','미국채10',
          '미국채30액티브','빅테크7','타미당','타미당7%']

def close_on_or_before(df: pd.DataFrame, date_str: str) -> Optional[float]:
    if df.empty or 'Close' not in df.columns:
        return None
    target = pd.to_datetime(date_str)
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    mask = df.index <= target
    if not mask.any():
        return None
    return float(df.loc[mask, 'Close'].iloc[-1])

@st.cache_data(show_spinner=True, ttl=3600)
def load_changes():
    rows = []
    for name, tic in zip(stocks, tickers):
        try:
            data = fdr.DataReader(tic, start)
        except Exception:
            data = pd.DataFrame()

        last_close = float(data['Close'].iloc[-1]) if (not data.empty and 'Close' in data.columns) else None
        p1 = close_on_or_before(data, ref_date1)
        p2 = close_on_or_before(data, ref_date2)

        chg1 = round((last_close / p1 - 1) * 100, 2) if (last_close is not None and p1 not in (None, 0)) else None
        chg2 = round((last_close / p2 - 1) * 100, 2) if (last_close is not None and p2 not in (None, 0)) else None
        rows.append([name, tic, chg1, chg2])
    return pd.DataFrame(rows, columns=['Stock','Ticker', f"Change since {ref_date1} (%)", f"Change since {ref_date2} (%)"])

df = load_changes()

def assign_colors(sorted_df: pd.DataFrame, col: str):
    try:
        kospi_ratio = float(sorted_df.loc[sorted_df['Ticker']=="KS11", col].iloc[0])
    except Exception:
        kospi_ratio = None
    colors = []
    for t, r in zip(sorted_df['Ticker'], sorted_df[col]):
        if t == "KS11":
            colors.append("black")
        elif r is None or kospi_ratio is None:
            colors.append("gray")
        elif r > kospi_ratio:
            colors.append("red")
        else:
            colors.append("blue")
    return colors

def bar_fig(df_in: pd.DataFrame, col: str, title: str):
    ordered = df_in[['Stock','Ticker', col]].sort_values(by=col, ascending=True, na_position='first').reset_index(drop=True)
    colors = assign_colors(ordered, col)
    fig = go.Figure([go.Bar(x=ordered['Stock'], y=ordered[col], marker_color=colors)])
    fig.update_layout(title=title, xaxis_title="Stock", yaxis_title="Change (%)", showlegend=False, height=600)
    return fig

c1, c2 = st.columns(2)
col1_name = f"Change since {ref_date1} (%)"
col2_name = f"Change since {ref_date2} (%)"
with c1:
    st.subheader(f"기준일 1: {ref_date1}")
    st.plotly_chart(bar_fig(df, col1_name, f"Stock Changes since {ref_date1}"), use_container_width=True)

with c2:
    st.subheader(f"기준일 2: {ref_date2}")
    st.plotly_chart(bar_fig(df, col2_name, f"Stock Changes since {ref_date2}"), use_container_width=True)

st.markdown("### 데이터 테이블")
st.dataframe(df)
