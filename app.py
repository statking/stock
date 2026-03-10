# app.py
import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from typing import Optional, Tuple
import time
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="ETF 비교 대시보드", layout="wide")
st.title("코스피 대비 수익률 비교 대시보드")

# --- 티커 / 라벨 (코스피는 내부 기준으로 항상 포함) ---
TICKERS = [
    'KS11','KQ11','244580','091230','305540','266390','139280','117700','228790',
    '228800','138520','138540','138530','307520','161510','117460','139230','091220',
    '153130','183700','445290','385510','261220','411060','379800','379810','305080',
    '464470','465580','458730','494670','449450','491010', '138910',
    '005930','000660','005380'
]

STOCKS = [
    '코스피','코스닥','바이오','반도체','2차전지','경기소비재','경기방어','건설','화장품',
    '여행레저','삼성그룹','현대그룹','LG그룹','지주회사','고배당주','화학','중공업','은행',
    '단기채','채권혼합','로봇','신재생','원유선물','금현물','S&P500','나스닥100','미국채10',
    '미국채30','빅테크7','미국배당','조선','방산','AI전력', '구리선물',
    '삼성전자', 'SK하이닉스', '현대차'
]

NAME2TIC = dict(zip(STOCKS, TICKERS))

# -------------------------------
# 선택 UI에 노출할 종목(코스피 제외) + 정렬 규칙
# -------------------------------
def ui_sort_key(name: str):
    """가나다 -> 영어 -> 숫자 -> 기타 순 정렬 키"""
    ch = name[0]
    if '가' <= ch <= '힣':
        group = 0
    elif ch.isascii() and ch.isalpha():
        group = 1
    elif ch.isdigit():
        group = 2
    else:
        group = 3
    return (group, name.casefold())

VISIBLE_STOCKS = sorted([s for s in STOCKS if s != '코스피'], key=ui_sort_key)

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

def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out

def price_date_on_or_before(df: pd.DataFrame, date_str: str) -> Tuple[Optional[float], Optional[str]]:
    """date_str 이전(포함) 마지막 거래일의 종가와 실제 기준일 반환"""
    if df.empty or 'Close' not in df.columns:
        return None, None
    df = normalize_index(df)
    target = pd.to_datetime(date_str)
    mask = df.index <= target
    if not mask.any():
        return None, None
    used_date = df.index[mask][-1]
    price = float(df.loc[used_date, 'Close'])
    return price, used_date.strftime("%Y-%m-%d")

def price_date_on_or_after(df: pd.DataFrame, date_str: str) -> Tuple[Optional[float], Optional[str]]:
    """date_str 이후(포함) 첫 거래일의 종가와 실제 기준일 반환"""
    if df.empty or 'Close' not in df.columns:
        return None, None
    df = normalize_index(df)
    target = pd.to_datetime(date_str)
    mask = df.index >= target
    if not mask.any():
        return None, None
    used_date = df.index[mask][0]
    price = float(df.loc[used_date, 'Close'])
    return price, used_date.strftime("%Y-%m-%d")

@st.cache_data(show_spinner=True, ttl=3600)
def load_interval_returns(
    start_anchor: str,
    s1: str, e1: str,
    s2: str, e2: str,
    sel_tickers: Tuple[str, ...],
    sel_names: Tuple[str, ...]
) -> pd.DataFrame:
    """각 종목에 대해 구간1(s1~e1), 구간2(s2~e2) 수익률(%) 계산"""
    rows = []

    for name, tic in zip(sel_names, sel_tickers):
        data = safe_read(tic, start_anchor, retry=1, wait=1.2)

        a1, a1_date = price_date_on_or_after(data, s1)
        b1, b1_date = price_date_on_or_before(data, e1)
        a2, a2_date = price_date_on_or_after(data, s2)
        b2, b2_date = price_date_on_or_before(data, e2)

        r1 = round((b1 / a1 - 1) * 100, 2) if (a1 not in (None, 0) and b1 not in (None, 0)) else None
        r2 = round((b2 / a2 - 1) * 100, 2) if (a2 not in (None, 0) and b2 not in (None, 0)) else None

        rows.append([
            name, tic,
            a1_date, b1_date, r1,
            a2_date, b2_date, r2
        ])

    col1 = f"Return {s1}→{e1} (%)"
    col2 = f"Return {s2}→{e2} (%)"

    df = pd.DataFrame(
        rows,
        columns=[
            'Stock', 'Ticker',
            f'구간1 시작 기준일', f'구간1 종료 기준일', col1,
            f'구간2 시작 기준일', f'구간2 종료 기준일', col2
        ]
    )

    # 안전장치: 티커 중복 제거(코스피 이중표시 예방)
    df = df.drop_duplicates(subset='Ticker', keep='first').reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_reference_dates(start_anchor: str, s1: str, e1: str, s2: str, e2: str):
    """
    코스피(KS11)를 기준으로 실제 계산에 사용될 대표 기준일 확인
    안내 문구용
    """
    df_kospi = safe_read('KS11', start_anchor, retry=1, wait=1.0)
    if df_kospi.empty:
        return None

    _, s1_used = price_date_on_or_after(df_kospi, s1)
    _, e1_used = price_date_on_or_before(df_kospi, e1)
    _, s2_used = price_date_on_or_after(df_kospi, s2)
    _, e2_used = price_date_on_or_before(df_kospi, e2)

    return {
        "s1_used": s1_used,
        "e1_used": e1_used,
        "s2_used": s2_used,
        "e2_used": e2_used,
    }

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
            colors.append('black')   # 코스피는 검정
        elif (r is None) or (pd.isna(r)) or (kospi_ratio is None):
            colors.append('gray')    # 비교 불가
        elif r > kospi_ratio:
            colors.append('red')     # 코스피보다 좋음
        else:
            colors.append('blue')    # 코스피보다 나쁨/같음
    return colors

def bar_fig(df_in: pd.DataFrame, col: str, title: str) -> go.Figure:
    if col not in df_in.columns:
        st.error(f"필요한 컬럼이 없습니다: '{col}'.")
        return go.Figure()

    df_plot = df_in[['Stock', 'Ticker', col]].copy()
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    df_plot = df_plot.drop_duplicates(subset='Ticker', keep='first')

    ordered = df_plot.sort_values(by=col, ascending=True, na_position='first').reset_index(drop=True)
    colors = assign_colors(ordered, col)

    fig = go.Figure([go.Bar(x=ordered['Stock'], y=ordered[col], marker_color=colors)])
    fig.update_layout(
        title=title,
        xaxis_title="Stock",
        yaxis_title="Return (%)",
        showlegend=False,
        height=600,
        bargap=0.2
    )
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

r1c1, r1c2, _ = st.columns([1, 1, 6])
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
            value=st.session_state.toggle_states.get(name, True),
            key=f"tg_{name}"
        )
        grid_index += 1

selected_user_names = [n for n, v in st.session_state.toggle_states.items() if v]
final_names = ['코스피'] + selected_user_names
final_tickers = ['KS11'] + [NAME2TIC[n] for n in selected_user_names]

st.divider()

# -------------------------------
# 2) 구간 입력
# -------------------------------
st.subheader("기간 설정")

# 한국 시간 기준 오늘
today = datetime.now(ZoneInfo("Asia/Seoul")).date()
this_year = today.year

# 디폴트: 구간1 시작 = 작년 1/1, 구간2 시작 = 올해 1/1, 종료는 모두 오늘
g1_default_start = datetime(this_year - 1, 1, 1).date()
g2_default_start = datetime(this_year, 1, 1).date()

col_g1, col_g2 = st.columns(2, vertical_alignment="top")

with col_g1:
    with st.container(border=True):
        st.markdown("**구간 1**")
        g1_start = st.date_input("시작 날짜를 선택하세요.", g1_default_start, key="g1_start")
        g1_end = st.date_input("종료 날짜를 선택하세요.", today, key="g1_end")

with col_g2:
    with st.container(border=True):
        st.markdown("**구간 2**")
        g2_start = st.date_input("시작 날짜를 선택하세요.", g2_default_start, key="g2_start")
        g2_end = st.date_input("종료 날짜를 선택하세요.", today, key="g2_end")

bc1, bc2, bc3 = st.columns([4, 2, 4])
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

# 앵커: 두 구간의 가장 이른 시작일 - 40일
min_start = min(pd.to_datetime(s1), pd.to_datetime(s2))
start_anchor = (min_start - pd.Timedelta(days=40)).strftime("%Y-%m-%d")

# 대표 기준일 안내용 (차단 X, 안내만)
ref_dates = get_reference_dates(start_anchor, s1, e1, s2, e2)

st.caption(
    "입력한 날짜의 종가가 없으면 자동으로 가장 가까운 거래일 종가를 사용합니다. "
    "시작일은 해당 날짜 이후 첫 거래일, 종료일은 해당 날짜 이전 마지막 거래일 기준입니다."
)

if ref_dates is not None:
    info_msgs = []

    if ref_dates["s1_used"] is not None and ref_dates["s1_used"] != s1:
        info_msgs.append(f"구간 1 시작일 {s1} → 실제 적용일 {ref_dates['s1_used']}")
    if ref_dates["e1_used"] is not None and ref_dates["e1_used"] != e1:
        info_msgs.append(f"구간 1 종료일 {e1} → 실제 적용일 {ref_dates['e1_used']}")
    if ref_dates["s2_used"] is not None and ref_dates["s2_used"] != s2:
        info_msgs.append(f"구간 2 시작일 {s2} → 실제 적용일 {ref_dates['s2_used']}")
    if ref_dates["e2_used"] is not None and ref_dates["e2_used"] != e2:
        info_msgs.append(f"구간 2 종료일 {e2} → 실제 적용일 {ref_dates['e2_used']}")

    if info_msgs:
        st.info(" / ".join(info_msgs))

df = load_interval_returns(
    start_anchor, s1, e1, s2, e2,
    tuple(final_tickers), tuple(final_names)
)

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도하세요.")
    st.stop()

# 전부 결측이면 공급자 문제 가능성 안내
col1_name = f"Return {s1}→{e1} (%)"
col2_name = f"Return {s2}→{e2} (%)"

if df[col1_name].isna().all() and df[col2_name].isna().all():
    st.error("수익률을 계산할 수 있는 데이터가 없습니다. 데이터 제공 상태를 확인한 뒤 다시 시도하세요.")
    st.stop()

# -------------------------------
# 4) 출력
# -------------------------------
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
