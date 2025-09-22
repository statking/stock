# app.py
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from typing import Optional
import time
from datetime import date, datetime

st.set_page_config(page_title="ETF 비교 대시보드", layout="wide")
st.title("코스피 대비 수익률 비교 대시보드")

# --- 티커 / 라벨 (코스피는 내부 기준으로 항상 포함) ---
TICKERS = ['KS11','KQ11','244580','091230','305540','266390','139280','117700','228790',
           '228800','138520','138540','138530','307520','161510','117460','139230','091220',
           '153130','183700','445290','385510','261220','411060','379800','379810','305080',
           '464470','465580','458730','494670','449450','491010']

STOCKS = ['코스피','코스닥','바이오','반도체','2차전지','경기소비재','경기방어','건설','화장품',
          '여행레저','삼성그룹','현대그룹','LG그룹','지주회사','고배당주','화학','중공업','은행',
          '단기채','채권혼합','로봇','신재생','원유선물','금현물','S&P500','나스닥100','미국채10',
          '미국채30','빅테크7','타미당','조선','방산','AI전력']

NAME2TIC = dict(zip(STOCKS, TICKERS))

# UI에 표시할 선택 항목(코스피 제외)
VISIBLE_STOCKS = [s for s in STOCKS if s != '코스피']

# --------- 유틸 ---------
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
    """각 종목에 대해 구간1(s1~e1), 구간2(s2~e2) 수익률(%) 계산"""
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
    df = pd.DataFrame(rows, columns=['Stock','Ticker', col1, col2])

    # 안전장치: 티커 중복 방지 (코스피 이중표시 등)
    df = df.drop_duplicates(subset='Ticker', keep='first').reset_index(drop=True)
    return df

def assign_colors(sorted_df: pd.DataFrame, col: str) -> list[str]:
    # 코스피(KS11)를 기준으로 상대 성과 색상 결정
    kospi_ratio = None
    try:
        kospi_ratio = float(sorted_df.loc[sorted_df['Ticker'] == 'KS11', col].iloc[0])
    except Exception:
        kospi_ratio = None
    colors = []
    for t, r in zip(sorted_df['Ticker'], sorted_df[col]):
        if t == 'KS11':
            colors.append('black')        # 코스피는 검정
        elif (r is None) or (pd.isna(r)) or (kospi_ratio is None):
            colors.append('gray')         # 비교 불가
        elif r > kospi_ratio:
            colors.append('red')          # 코스피보다 좋음
        else:
            colors.append('blue')         # 코스피보다 나쁨/같음
    return colors

def bar_fig(df_in: pd.DataFrame, col: str, title: str) -> go.Figure:
    if col not in df_in.columns:
        st.error(f"필요한 컬럼이 없습니다: '{col}'.")
        return go.Figure()

    # 수치 변환 + 티커 중복 방지 (이중 막대 예방)
    df_plot = df_in[['Stock','Ticker', col]].copy()
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    df_plot = df_plot.drop_duplicates(subset='Ticker', keep='first')

    ordered = df_plot.sort_values(by=col, ascending=True, na_position='first').reset_index(drop=True)
    colors = assign_colors(ordered, col)

    fig = go.Figure([go.Bar(x=ordered['Stock'], y=ordered[col], marker_color=colors)])
    fig.update_layout(title=title, xaxis_title="Stock", yaxis_title="Return (%)",
                      showlegend=False, height=600, bargap=0.2)
    return fig

# -------------------------------
# 0) 토글 상태 초기화 (코스피 제외, 디폴트 = 전체 선택)
# -------------------------------
if "toggle_states" not in st.session_state:
    st.session_state.toggle_states = {name: True for name in VISIBLE_STOCKS}

def set_all_visible(value: bool):
    for name in VISIBLE_STOCKS:
        st.session_state.toggle_states[name] = value

# -------------------------------
# 1) 종목 선택 (코스피는 숨김, 내부적으로 항상 포함)
# -------------------------------
st.subheader("분석 대상 선택")

r1c1, r1c2, _ = st.columns([1,1,6])
with r1c1:
    if st.button("전체 선택", use_container_width=True):
        set_all_visible(True)
with r1c2:
    if st.button("전체 해제", use_container_width=True):
        set_all_visible(False)

N_COLS = 6
rows = (len(VISIBLE_STOCKS) + N_COLS - 1) // N_COLS
grid_index = 0
for _ in range(rows):
    cols = st.columns(N_COLS, gap="small")
    for c in cols:
        if grid_index >= len(VISIBLE_STOCKS):
            break
        name = VISIBLE_STOCKS[grid_index]
        st.session_state.toggle_states[name] = c.checkbox(
            label=name,
            value=st.session_state.toggle_states.get(name, True),  # 기본 전체 선택
            key=f"tg_{name}"
        )
        grid_index += 1

# 내부 로직용 선택 목록: 코스피는 항상 포함 (UI에는 미표시)
selected_user_names = [n for n, v in st.session_state.toggle_states.items() if v]
final_names   = ['코스피'] + selected_user_names
final_tickers = ['KS11']  + [NAME2TIC[n] for n in selected_user_names]

st.divider()

# -------------------------------
# 2) 구간 입력: 덩어리(컨테이너)로 묶고 위→아래 배치
# -------------------------------
st.subheader("기간 설정")
today = date.today()
this_year = today.year
# 디폴트: 구간1 시작 = 작년 1/1, 구간2 시작 = 올해 1/1, 종료는 모두 오늘
g1_default_start = date(this_year - 1, 1, 1)
g2_default_start = date(this_year, 1, 1)

col_g1, col_g2 = st.columns(2, vertical_alignment="top")

with col_g1:
    with st.container(border=True):
        st.markdown("**구간 1**")
        g1_start = st.date_input("시작 날짜를 선택하세요.", g1_default_start, key="g1_start")
        g1_end   = st.date_input("종료 날짜를 선택하세요.", today, key="g1_end")

with col_g2:
    with st.container(border=True):
        st.markdown("**구간 2**")
        g2_start = st.date_input("시작 날짜를 선택하세요.", g2_default_start, key="g2_start")
        g2_end   = st.date_input("종료 날짜를 선택하세요.", today, key="g2_end")

# 실행 버튼 중앙 정렬
bc1, bc2, bc3 = st.columns([4,2,4])
with bc2:
    run = st.button("분석하기", use_container_width=True)

if not run:
    st.stop()

# -------------------------------
# 3) 유효성 검사 및 데이터 로드
# -------------------------------
s1 = pd.to_datetime(g1_start).strftime("%Y-%m-%d")
e1 = pd.to_datetime(g1_end).strftime("%Y-%m-%d")
s2 = pd.to_datetime(g2_start).strftime("%Y-%m-%d")
e2 = pd.to_datetime(g2_end).strftime("%Y-%m-%d")

if pd.to_datetime(s1) > pd.to_datetime(e1):
    st.error("구간 1: 시작 날짜가 종료 날짜보다 늦습니다.")
    st.stop()
if pd.to_datetime(s2) > pd.to_datetime(e2):
    st.error("구간 2: 시작 날짜가 종료 날짜보다 늦습니다.")
    st.stop()

# 앵커: 두 구간의 가장 이른 시작일 - 30일
min_start = min(pd.to_datetime(s1), pd.to_datetime(s2))
start_anchor = (min_start - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

df = load_interval_returns(start_anchor, s1, e1, s2, e2,
                           tuple(final_tickers), tuple(final_names))

# -------------------------------
# 4) 출력
# -------------------------------
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
