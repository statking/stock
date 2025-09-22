# app.py
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from typing import Optional
import time
from datetime import date

st.set_page_config(page_title="ETF 비교 대시보드", layout="wide")
st.title("코스피 대비 수익률 비교 대시보드")

# --- 티커 / 라벨 ---
TICKERS = ['KS11','KQ11','244580','091230','305540','266390','139280','117700','228790',
           '228800','138520','138540','138530','307520','161510','117460','139230','091220',
           '153130','183700','445290','385510','261220','411060','379800','379810','305080',
           '464470','465580','458730','494670','449450','491010']

STOCKS = ['코스피','코스닥','바이오','반도체','2차전지','경기소비재','경기방어','건설','화장품',
          '여행레저','삼성그룹','현대그룹','LG그룹','지주회사','고배당주','화학','중공업','은행',
          '단기채','채권혼합','로봇','신재생','원유선물','금현물','S&P500','나스닥100','미국채10',
          '미국채30','빅테크7','타미당','조선','방산','AI전력']

NAME2TIC = dict(zip(STOCKS, TICKERS))

# --- 유틸 ---
def safe_read(ticker: str, start: str, retry: int = 1, wait: float = 1.0) -> pd.DataFrame:
    for i in range(retry + 1):
        try:
            return fdr.DataReader(ticker, start)
        except Exception:
            if i < retry:
                time.sleep(wait)
            else:
                return pd.DataFrame()

def close_on_or_before(df: pd.DataFrame, date_str: str) -> Optional[float]:
    if df.empty or 'Close' not in df.columns:
        return None
    target = pd.to_datetime(date_str)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    mask = df.index <= target
    if not mask.any():
        return None
    return float(df.loc[mask, 'Close'].iloc[-1])

def close_on_or_after(df: pd.DataFrame, date_str: str) -> Optional[float]:
    if df.empty or 'Close' not in df.columns:
        return None
    target = pd.to_datetime(date_str)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    mask = df.index >= target
    if not mask.any():
        return None
    return float(df.loc[mask, 'Close'].iloc[0])

@st.cache_data(show_spinner=True, ttl=3600)
def load_interval_returns(start_anchor: str,
                          s1: str, e1: str,
                          s2: str, e2: str,
                          sel_tickers: tuple[str, ...],
                          sel_names: tuple[str, ...]) -> pd.DataFrame:
    """
    각 종목에 대해 구간1(s1~e1), 구간2(s2~e2) 수익률(%) 계산.
    DataReader는 start_anchor부터 읽어 효율을 확보.
    """
    rows = []
    for name, tic in zip(sel_names, sel_tickers):
        data = safe_read(tic, start_anchor, retry=1, wait=1.2)

        a1 = close_on_or_after(data, s1)
        b1 = close_on_or_before(data, e1)
        a2 = close_on_or_after(data, s2)
        b2 = close_on_or_before(data, e2)

        r1 = round((b1 / a1 - 1) * 100, 2) if (a1 not in (None, 0) and b1 not in (None, 0)) else None
        r2 = round((b2 / a2 - 1) * 100, 2) if (a2 not in (None, 0) and b2 not in (None, 0)) else None
        rows.append([name, tic, r1, r2])

    col1 = f"Return {s1}→{e1} (%)"
    col2 = f"Return {s2}→{e2} (%)"
    df = pd.DataFrame(rows, columns=['Stock', 'Ticker', col1, col2])
    return df

def assign_colors(sorted_df: pd.DataFrame, col: str) -> list[str]:
    # 코스피(‘KS11’) 기준 색상: 코스피 검정, 코스피보다 높으면 빨강, 낮으면 파랑, 비교불가 회색
    kospi_ratio = None
    try:
        kospi_ratio = float(sorted_df.loc[sorted_df['Ticker'] == 'KS11', col].iloc[0])
    except Exception:
        kospi_ratio = None

    colors = []
    for t, r in zip(sorted_df['Ticker'], sorted_df[col]):
        if t == 'KS11':
            colors.append('black')
        elif (r is None) or (pd.isna(r)) or (kospi_ratio is None):
            colors.append('gray')
        elif r > kospi_ratio:
            colors.append('red')
        else:
            colors.append('blue')
    return colors

def bar_fig(df_in: pd.DataFrame, col: str, title: str) -> go.Figure:
    if col not in df_in.columns:
        st.error(f"필요한 컬럼이 없습니다: '{col}'. 페이지를 다시 실행해 주세요.")
        return go.Figure()

    df_plot = df_in[['Stock','Ticker', col]].copy()
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

    ordered = (
        df_plot.sort_values(by=col, ascending=True, na_position='first')
               .reset_index(drop=True)
    )
    colors = assign_colors(ordered, col)
    fig = go.Figure([go.Bar(x=ordered['Stock'], y=ordered[col], marker_color=colors)])
    fig.update_layout(title=title, xaxis_title="Stock", yaxis_title="Return (%)",
                      showlegend=False, height=600, bargap=0.2)
    return fig

# ------------------------------------------
# 1) 첫 화면: 종목 선택 + 구간 날짜 입력(폼)
# ------------------------------------------
with st.form("config_form"):
    st.subheader("분석 설정")

    # 종목 선택
    st.markdown("**분석할 종목을 선택하세요.** (멀티선택)")
    c_sel1, c_sel2 = st.columns([3,1])
    default_all = c_sel2.checkbox("전체 선택/해제", value=True)
    if default_all:
        default_selection = STOCKS
    else:
        default_selection = []

    selected_names = c_sel1.multiselect(
        "종목 선택 (토글)",
        options=STOCKS,
        default=default_selection
    )

    if not selected_names:
        st.info("최소 1개 이상의 종목을 선택해 주세요.")
    st.divider()

    # 날짜 입력 - 구간1, 구간2
    today = date.today()
    st.markdown("**구간 1**")
    g1c1, g1c2 = st.columns(2)
    g1_start = g1c1.date_input("시작 날짜를 선택하세요. (구간 1 시작)", pd.to_datetime("2024-01-01"))
    g1_end   = g1c2.date_input("구간 1 끝 날짜", today)

    st.markdown("**구간 2**")
    g2c1, g2c2 = st.columns(2)
    g2_start = g2c1.date_input("시작 날짜를 선택하세요. (구간 2 시작)", pd.to_datetime("2024-07-01"))
    g2_end   = g2c2.date_input("구간 2 끝 날짜", today)

    submitted = st.form_submit_button("분석하기")

# 제출 전에는 종료
if not submitted:
    st.stop()

# --- 유효성 검사 ---
if not selected_names:
    st.error("분석할 종목을 한 개 이상 선택해 주세요.")
    st.stop()

# 날짜 문자열 변환
s1 = pd.to_datetime(g1_start).strftime("%Y-%m-%d")
e1 = pd.to_datetime(g1_end).strftime("%Y-%m-%d")
s2 = pd.to_datetime(g2_start).strftime("%Y-%m-%d")
e2 = pd.to_datetime(g2_end).strftime("%Y-%m-%d")

# 시작-끝 순서 체크
if pd.to_datetime(s1) > pd.to_datetime(e1):
    st.error("구간 1: 시작 날짜가 끝 날짜보다 늦습니다. 다시 선택해 주세요.")
    st.stop()
if pd.to_datetime(s2) > pd.to_datetime(e2):
    st.error("구간 2: 시작 날짜가 끝 날짜보다 늦습니다. 다시 선택해 주세요.")
    st.stop()

# 선택 종목에 해당하는 티커만
selected_tickers = tuple(NAME2TIC[n] for n in selected_names)

# 앵커(start_anchor): 두 구간의 가장 이른 시작 날짜보다 30일 앞에서 읽어 여유 확보
min_start = min(pd.to_datetime(s1), pd.to_datetime(s2))
start_anchor = (min_start - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

# 데이터 로드
df = load_interval_returns(
    start_anchor,
    s1, e1,
    s2, e2,
    tuple(selected_tickers),
    tuple(selected_names)
)

# 출력
col1_name = f"Return {s1}→{e1} (%)"
col2_name = f"Return {s2}→{e2} (%)"

left, right = st.columns(2)
with left:
    st.subheader(f"구간 1: {s1} → {e1}")
    st.plotly_chart(
        bar_fig(df, col1_name, f"Interval Return: {s1}→{e1}"),
        use_container_width=True,
        key=f"plot_{s1}_{e1}_left"
    )
with right:
    st.subheader(f"구간 2: {s2} → {e2}")
    st.plotly_chart(
        bar_fig(df, col2_name, f"Interval Return: {s2}→{e2}"),
        use_container_width=True,
        key=f"plot_{s2}_{e2}_right"
    )

st.markdown("### 데이터 테이블")
st.dataframe(df, use_container_width=True)
