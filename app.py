"""
AI Citation Analyzer v4.0 — 통합형 + 6기능 확장

추가 기능:
1. Google Sheets 인용율 추이 트래킹
2. 경쟁사 직접 비교 분석 (병렬 시뮬레이션 + 그룹 바 차트)
3. AI 질문 추천 엔진 (GPT/Gemini → 8개 추천 → 체크박스 선택)
4. PDF 리포트 자동 생성 (reportlab, 한글 폰트)
7. AI 응답 원문 표시 + 감성 분류
9. 콘텐츠 GEO 점수 편집기 (0-100점, 레이더 차트)
"""

import streamlit as st
import datetime
import re
import time
import json
import io
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import urlparse
import concurrent.futures as cf

import sys, os
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
sys.path.insert(0, "/mount/src/ai_geo_solution")

try:
    from core.cache import get_cache
    from core.logger import get_logger, CaptureError
    from core.citation import build_brand_variants
    from core.ai_client import run_all_simulations, CostTracker, SimResult, call_gpt, call_gemini
    from core.biz_analysis import run_strategy_analysis, BusinessInfo
except ImportError as _e:
    import streamlit as st
    st.set_page_config(page_title="Import Error", page_icon="❌")
    st.error(f"**core 모듈 import 실패**: {_e}")
    st.code(f"APP_ROOT: {_APP_ROOT}\nCWD: {os.getcwd()}\ncore/ 존재: {os.path.isdir(os.path.join(_APP_ROOT, 'core'))}")
    st.stop()

# Optional dependencies
try:
    import gspread
    _GSPREAD_OK = True
except ImportError:
    _GSPREAD_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm as rl_cm
    _REPORTLAB_OK = True
except ImportError:
    _REPORTLAB_OK = False

logger = get_logger("app")

st.set_page_config(
    page_title="AI Citation Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "dark_mode":       False,
    "cache_data":      {},
    "cost_tracker":    CostTracker(),
    "history":         [],
    "run_demo":        False,
    "q_count":         1,
    "questions_list":  [""],
    # 기능 3
    "recommended_qs":  [],
    "show_rec_qs":     False,
    # 기능 2
    "comp_urls":       ["", "", ""],
    "comp_brands":     ["", "", ""],
    "comp_results":    {},
    # 기능 4 / 7 / 9 persistence
    "last_all_results":  [],
    "last_strategy":     None,
    "last_questions":    [],
    "last_url":          "",
    "last_brand":        "",
    "last_sim_mentions": {},
    "geo_editor_result": None,
    # 기능 1
    "gsheets_url":       "",
    "sa_json_str":       "",
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_cache = get_cache()
if st.session_state["cache_data"]:
    try:
        _restored = type(_cache).from_serializable(st.session_state["cache_data"])
        _cache._store.update(_restored._store)
    except Exception:
        st.session_state["cache_data"] = {}

_dark = st.session_state["dark_mode"]

if _dark:
    _bg="#0F0F0F";_bg2="#1A1A1A";_card="#1E1E1E";_border="#333333";_text="#F0F0F0";_text_muted="#999999"
    _shadow="0 4px 24px rgba(0,0,0,.5)";_header_gr="linear-gradient(135deg,#1A1A1A,#2A2A2A,#3A3A3A)"
    _sidebar_gr="linear-gradient(180deg,#0F0F0F,#1A1A1A,#222222)";_metric_bg="#252525";_input_bg="rgba(255,255,255,.07)"
    _btn_gr="linear-gradient(135deg,#333,#555)";_tab_bg="#1E1E1E";_tab_sel="#333333";_tab_txt="#F0F0F0"
    _progress="linear-gradient(90deg,#555,#888)";_plot_bg="rgba(30,30,30,.9)";_plot_paper="#1E1E1E"
    _plot_font="#F0F0F0";_plot_grid="#333"
else:
    _bg="#F5F5F5";_bg2="#EEEEEE";_card="#FFFFFF";_border="#DDDDDD";_text="#111111";_text_muted="#666666"
    _shadow="0 4px 24px rgba(0,0,0,.08)";_header_gr="linear-gradient(135deg,#111,#333,#555)"
    _sidebar_gr="linear-gradient(180deg,#111,#222,#333)";_metric_bg="#FFFFFF";_input_bg="#FFFFFF"
    _btn_gr="linear-gradient(135deg,#111,#444)";_tab_bg="#FFFFFF";_tab_sel="#111111";_tab_txt="#FFFFFF"
    _progress="linear-gradient(90deg,#111,#555)";_plot_bg="rgba(245,245,245,.8)";_plot_paper="#FFFFFF"
    _plot_font="#111111";_plot_grid="#DDDDDD"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html,body{{background-color:{_bg}!important;color:{_text}!important}}
.stApp,.stApp>*{{background:{_bg}!important}}
.main .block-container{{background:{_bg}!important}}
*,*::before,*::after{{font-family:'Plus Jakarta Sans',sans-serif!important}}
.stApp p,.stApp span,.stApp div,.stApp label,.stApp small,.stApp strong,
.stMarkdown,.stMarkdown p,.stMarkdown li,
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,.stMarkdown h5,.stMarkdown h6,
[data-testid="stText"],[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{{
    color:{_text}!important;-webkit-text-fill-color:{_text}!important}}
div[data-testid="metric-container"]{{
    background:{_metric_bg}!important;border-radius:14px!important;
    padding:18px!important;border:1px solid {_border}!important;box-shadow:{_shadow}!important}}
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] span{{
    color:{_text_muted}!important;-webkit-text-fill-color:{_text_muted}!important;
    font-size:.8rem!important;font-weight:600!important}}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] *{{
    color:{_text}!important;-webkit-text-fill-color:{_text}!important;
    font-size:1.6rem!important;font-weight:800!important}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"],
div[data-testid="metric-container"] [data-testid="stMetricDelta"] *{{
    color:{_text_muted}!important;background:transparent!important}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg{{display:none!important}}
.stTextInput label,.stTextArea label,.stSelectbox label,.stSlider label,
.stRadio label,.stCheckbox label,
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p{{
    color:{_text}!important;-webkit-text-fill-color:{_text}!important;font-weight:600!important}}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{{
    border-radius:12px!important;border:1.5px solid {_border}!important;
    background:{_input_bg}!important;color:{_text}!important;
    -webkit-text-fill-color:{_text}!important;font-size:.9rem!important}}
.stTextInput>div>div>input::placeholder,.stTextArea>div>div>textarea::placeholder{{color:{_text_muted}!important}}
.stSelectbox>div>div,.stSelectbox [data-baseweb="select"]>div{{
    background:{_input_bg}!important;border:1.5px solid {_border}!important;
    border-radius:12px!important;color:{_text}!important}}
.stSelectbox [data-baseweb="select"] span{{color:{_text}!important}}
[data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"] li{{
    background:{_card}!important;color:{_text}!important}}
.stTabs [data-baseweb="tab-list"]{{
    background:{_tab_bg}!important;border-radius:14px!important;
    padding:6px!important;border:1px solid {_border}!important}}
.stTabs [data-baseweb="tab"]{{
    border-radius:10px!important;font-weight:600!important;font-size:.9rem!important;
    color:{_text_muted}!important;-webkit-text-fill-color:{_text_muted}!important;
    padding:10px 22px!important;background:transparent!important}}
.stTabs [aria-selected="true"]{{
    background:{_tab_sel}!important;color:{_tab_txt}!important;
    -webkit-text-fill-color:{_tab_txt}!important;box-shadow:0 4px 12px rgba(0,0,0,.3)!important}}
.stTabs [data-baseweb="tab"] p,.stTabs [data-baseweb="tab"] span{{
    color:inherit!important;-webkit-text-fill-color:inherit!important}}
.stButton>button{{
    background:{_btn_gr}!important;color:white!important;-webkit-text-fill-color:white!important;
    border:none!important;border-radius:12px!important;font-weight:700!important;
    padding:12px 28px!important;transition:all .2s!important}}
.stButton>button:hover{{transform:translateY(-2px)!important;opacity:.92!important}}
.stButton>button p{{color:white!important;-webkit-text-fill-color:white!important}}
.stProgress>div>div>div{{background:{_progress}!important;border-radius:8px!important}}
[data-testid="stSidebar"]{{background:{_sidebar_gr}!important}}
[data-testid="stSidebar"] *:not(.stButton>button){{color:white!important;-webkit-text-fill-color:white!important}}
[data-testid="stSidebar"] .stTextInput>div>div>input{{
    background:rgba(255,255,255,.12)!important;border:1px solid rgba(255,255,255,.25)!important;
    color:white!important;-webkit-text-fill-color:white!important}}
[data-testid="stSidebar"] .stSelectbox>div>div{{
    background:rgba(255,255,255,.1)!important;border:1px solid rgba(255,255,255,.2)!important}}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span{{
    color:white!important;-webkit-text-fill-color:white!important}}
details summary svg{{display:none!important}}
details summary::-webkit-details-marker{{display:none!important}}
details summary::marker{{display:none!important}}
[data-testid="stExpanderToggleIcon"]{{display:none!important;width:0!important}}
.streamlit-expander,details{{background:{_card}!important;border:1px solid {_border}!important;border-radius:12px!important}}
.streamlit-expander summary,details summary{{color:{_text}!important;-webkit-text-fill-color:{_text}!important;list-style:none!important}}
[data-testid="stExpander"]{{background:{_card}!important;border:1px solid {_border}!important;border-radius:12px!important}}
[data-testid="stExpander"] summary{{color:{_text}!important;-webkit-text-fill-color:{_text}!important}}
[data-testid="stExpander"] p,[data-testid="stExpander"] span{{color:{_text}!important;-webkit-text-fill-color:{_text}!important}}
[data-testid="stDataFrame"]{{background:{_card}!important}}
[data-testid="stDataFrame"] *{{color:{_text}!important;background:{_card}!important}}
.result-card{{background:{_card}!important;border-radius:16px;padding:22px 24px;border:1px solid {_border};box-shadow:{_shadow}}}
.result-card h4{{color:{_text}!important}}.result-card p{{color:{_text}!important}}
.strategy-item{{background:{_card};border-radius:12px;padding:14px 16px;margin:8px 0;border:1px solid {_border}}}
.strategy-item span{{font-size:.85rem;color:{_text}!important;word-break:keep-all;display:block;line-height:1.6}}
.blue-ocean-item{{background:{_bg2};border-radius:12px;padding:12px 16px;margin:8px 0;border:1px solid {_border};display:flex;align-items:center;gap:10px}}
.blue-ocean-item .label{{background:linear-gradient(135deg,#111,#444);color:white!important;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:700}}
.blue-ocean-item .kw{{font-size:.88rem;color:{_text}!important;font-weight:600}}
.geo-guide-item{{background:{_card};border-radius:14px;padding:18px 20px;margin:10px 0;border:1px solid {_border};box-shadow:0 2px 12px rgba(0,0,0,.06)}}
.geo-guide-item .geo-text{{margin:0;font-size:.88rem;color:{_text}!important;line-height:1.8;word-break:keep-all}}
.share-badge-high{{display:inline-block;background:linear-gradient(135deg,#10B981,#059669);color:white!important;padding:4px 12px;border-radius:20px;font-size:.85rem;font-weight:700}}
.share-badge-mid{{display:inline-block;background:linear-gradient(135deg,#F59E0B,#D97706);color:white!important;padding:4px 12px;border-radius:20px;font-size:.85rem;font-weight:700}}
.share-badge-low{{display:inline-block;background:linear-gradient(135deg,#EF4444,#DC2626);color:white!important;padding:4px 12px;border-radius:20px;font-size:.85rem;font-weight:700}}
.sidebar-logo{{text-align:center;padding:20px 0 24px;border-bottom:1px solid rgba(255,255,255,.15);margin-bottom:20px}}
.sidebar-logo .logo-icon{{font-size:2.5rem;display:block;margin-bottom:8px}}
.sidebar-logo h2{{color:white!important;font-size:1.1rem!important;font-weight:800!important;margin:0!important}}
.sidebar-logo p{{color:rgba(255,255,255,.6)!important;font-size:.75rem!important;margin:4px 0 0!important}}
.app-footer{{text-align:center;padding:20px;color:{_text_muted};font-size:.8rem;border-top:1px solid {_border}}}
[data-testid="stAlert"] p,[data-testid="stAlert"] span{{color:{_text}!important}}
.stCaptionContainer p{{color:{_text_muted}!important}}
</style>
""", unsafe_allow_html=True)


# ── Utilities ──────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    if not url.startswith("http"):
        url = "https://" + url
    return url.rstrip("/")

def extract_domain(url: str) -> str:
    try:
        p = urlparse(url if url.startswith("http") else "https://" + url)
        return p.netloc.replace("www.", "")
    except Exception:
        return url

def save_cache():
    with CaptureError("save_cache", log_level="debug"):
        st.session_state["cache_data"] = _cache.to_serializable()

def _aggregate_sim_mentions(all_results: list) -> dict:
    merged: dict = {}
    for r in all_results:
        comp_m = (
            r.competitor_mentions if hasattr(r, "competitor_mentions")
            else r.get("competitor_mentions", {}) if isinstance(r, dict) else {}
        )
        for brand, data in comp_m.items():
            if brand not in merged:
                merged[brand] = {"mentions": 0, "urls": []}
            cnt = data.get("mentions", 0) if isinstance(data, dict) else 1
            merged[brand]["mentions"] += cnt
            if isinstance(data, dict):
                merged[brand]["urls"] = list(set(merged[brand]["urls"] + data.get("urls", [])))
    return merged


# ── 기능 1: Google Sheets ──────────────────────────────────────────────────────

def _get_gsheets_client(sa_json_str: str):
    if not _GSPREAD_OK or not sa_json_str.strip():
        return None
    with CaptureError("gsheets_client", log_level="warning"):
        sa_info = json.loads(sa_json_str)
        return gspread.service_account_from_dict(sa_info)
    return None

def save_to_gsheets(gc, sheet_url: str, brand: str, domain: str,
                    questions: list, all_results: list) -> bool:
    if not gc or not sheet_url.strip():
        return False
    with CaptureError("gsheets_save", log_level="warning") as ctx:
        sh = gc.open_by_url(sheet_url)
        ws = sh.sheet1
        existing = ws.get_all_values()
        if not existing:
            ws.append_row(["timestamp", "brand", "domain", "question",
                           "gpt_rate", "gemini_rate", "avg_rate"])
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        rows = []
        for q, r in zip(questions, all_results):
            _g = lambda k, rr=r: (rr.get(k) if isinstance(rr, dict) else getattr(rr, k, None)) or 0
            rows.append([ts, brand, domain, q,
                         round(_g("gpt_rate"), 2),
                         round(_g("gemini_rate"), 2),
                         round(_g("avg_rate"), 2)])
        ws.append_rows(rows)
    return ctx.ok

def load_from_gsheets(gc, sheet_url: str) -> list:
    if not gc or not sheet_url.strip():
        return []
    with CaptureError("gsheets_load", log_level="warning"):
        sh = gc.open_by_url(sheet_url)
        ws = sh.sheet1
        return ws.get_all_records()
    return []

def render_gsheets_trend(records: list, brand: str):
    if not records:
        return
    df = pd.DataFrame(records)
    if "brand" in df.columns:
        df = df[df["brand"].astype(str) == brand]
    if df.empty:
        st.caption("저장된 데이터가 없습니다.")
        return
    for col in ["gpt_rate", "gemini_rate", "avg_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    fig = go.Figure()
    if "avg_rate" in df.columns and "timestamp" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["avg_rate"],
            mode="lines+markers", name="평균 인용율",
            line=dict(color="#555555", width=2),
            marker=dict(size=6),
        ))
    if "gpt_rate" in df.columns and "timestamp" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["gpt_rate"],
            mode="lines+markers", name="GPT",
            line=dict(color="#888888", dash="dot"),
        ))
    fig.update_layout(
        title=dict(text=f"{brand} 인용율 추이 (Google Sheets)", font=dict(size=14, color=_plot_font), x=0),
        plot_bgcolor=_plot_bg, paper_bgcolor=_plot_paper,
        font=dict(family="Plus Jakarta Sans", color=_plot_font),
        xaxis=dict(tickfont=dict(size=9, color=_plot_font), gridcolor=_plot_grid),
        yaxis=dict(title="인용율 (%)", ticksuffix="%", gridcolor=_plot_grid, tickfont=dict(color=_plot_font)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=_plot_font)),
        margin=dict(t=50, b=40, l=50, r=20), height=280,
    )
    _chart_counter["n"] += 1
    st.plotly_chart(fig, use_container_width=True, key=f"gs_trend_{_chart_counter['n']}")


# ── 기능 2: Competitor Comparison ─────────────────────────────────────────────

def run_competitor_comparison(client_gpt, client_gemini, urls: list, brands: list,
                               questions: list, n: int, model_gpt: str, tracker) -> dict:
    results: dict = {}

    def _run_one(url: str, brand: str):
        biz = {"brand_name": brand, "industry": "", "industry_category": "기타",
               "core_product": "", "target_audience": "잠재 고객"}
        return run_all_simulations(
            client_gpt, client_gemini, questions, url,
            model_gpt=model_gpt, n=n, biz_info=biz,
            tracker=tracker, use_cache=True,
        )

    pairs = [(u.strip(), b.strip()) for u, b in zip(urls, brands) if u.strip() and b.strip()]
    if not pairs:
        return results

    _ex = cf.ThreadPoolExecutor(max_workers=min(len(pairs), 3))
    try:
        futs = {_ex.submit(_run_one, u, b): (u, b) for u, b in pairs}
        done, pending = cf.wait(futs, timeout=360)
        for fut in done:
            _, brand = futs[fut]
            with CaptureError(f"comp_sim_{brand}", log_level="warning"):
                results[brand] = fut.result()
        for fut in pending:
            fut.cancel()
            _, brand = futs[fut]
            results[brand] = []
    finally:
        _ex.shutdown(wait=False)

    return results

def render_competitor_bar_chart(comp_results: dict, questions: list):
    if not comp_results or not questions:
        return
    short_q = [q[:18] + "..." if len(q) > 20 else q for q in questions]
    _colors = ["#111111", "#444444", "#777777", "#AAAAAA", "#CCCCCC"]
    fig = go.Figure()
    for i, (brand, res_list) in enumerate(comp_results.items()):
        avgs = []
        for r in (res_list or [])[:len(questions)]:
            v = (r.get("avg_rate") if isinstance(r, dict) else getattr(r, "avg_rate", None)) or 0
            avgs.append(v)
        while len(avgs) < len(questions):
            avgs.append(0)
        fig.add_trace(go.Bar(
            name=brand, x=short_q, y=avgs,
            marker=dict(color=_colors[i % len(_colors)]),
            text=[f"{v:.1f}%" for v in avgs],
            textposition="outside", textfont=dict(color=_plot_font),
        ))
    fig.update_layout(
        title=dict(text="경쟁사 AI 인용 점유율 비교", font=dict(size=16, color=_plot_font), x=0),
        barmode="group", bargap=0.2,
        plot_bgcolor=_plot_bg, paper_bgcolor=_plot_paper,
        font=dict(family="Plus Jakarta Sans", color=_plot_font),
        xaxis=dict(tickfont=dict(size=10, color=_plot_font), gridcolor=_plot_grid),
        yaxis=dict(title="인용 점유율 (%)", ticksuffix="%", gridcolor=_plot_grid,
                   tickfont=dict(color=_plot_font)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=_plot_font)),
        margin=dict(t=60, b=40, l=50, r=20), height=380,
    )
    _chart_counter["n"] += 1
    st.plotly_chart(fig, use_container_width=True, key=f"comp_chart_{_chart_counter['n']}")


# ── 기능 3: Question Recommendation ──────────────────────────────────────────

_REC_SYSTEM = "당신은 GEO 전문가입니다. 완성된 질문만 출력합니다."

def recommend_questions(url: str, brand_name: str, client_gpt, client_gemini, model_gpt: str) -> list:
    domain = extract_domain(url) if url else (brand_name or "서비스")
    prompt = f"""다음 사이트/브랜드의 AI 검색 점유율 분석에 효과적인 질문 8개를 추천하세요.

사이트 도메인: {domain}
브랜드명: {brand_name}

규칙:
1. 실제 사용자가 네이버·구글·ChatGPT 검색창에 입력하는 자연스러운 질문
2. 브랜드명·도메인 주소를 질문에 절대 포함하지 말 것
3. 구매 결정 5단계(인지·비교·신뢰·가격·전환) 중 다양하게 포함
4. 번호·기호 없이 질문 8개만, 한 줄에 하나, 물음표(?)로 끝낼 것"""

    result_str = ""
    with CaptureError("rec_qs", log_level="warning"):
        if client_gpt:
            result_str = call_gpt(client_gpt, prompt, system=_REC_SYSTEM,
                                  max_tokens=700, model=model_gpt, temperature=0.9)
        elif client_gemini:
            result_str = call_gemini(client_gemini, prompt, max_tokens=700, temperature=0.9)

    qs: list = []
    for ln in result_str.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        clean = re.sub(r"^\d+[.)]\s*", "", ln)
        clean = re.sub(r"^[-•*]\s*", "", clean).strip()
        if len(clean) > 10:
            if not clean.endswith("?"):
                clean += "?"
            qs.append(clean)
    return qs[:8]


# ── 기능 4: PDF Report ────────────────────────────────────────────────────────

def _get_pdf_font() -> str:
    if not _REPORTLAB_OK:
        return "Helvetica"
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        candidates = [
            ("C:/Windows/Fonts/malgun.ttf",  "MalgunGothic"),
            ("C:/Windows/Fonts/NanumGothic.ttf", "NanumGothic"),
            ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "NanumGothic"),
            (os.path.join(_APP_ROOT, "fonts", "NanumGothic.ttf"), "NanumGothic"),
        ]
        for path, name in candidates:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
                return name
    except Exception:
        pass
    return "Helvetica"

def generate_pdf(brand_name: str, target_url: str, questions: list,
                 all_results: list, strategy: dict) -> bytes:
    if not _REPORTLAB_OK:
        return b""
    buf = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*rl_cm, leftMargin=2*rl_cm,
                                topMargin=2*rl_cm, bottomMargin=2*rl_cm)
        fn = _get_pdf_font()
        styles = getSampleStyleSheet()
        s_title = ParagraphStyle("T", parent=styles["Title"], fontName=fn, fontSize=20, spaceAfter=6)
        s_sub   = ParagraphStyle("S", parent=styles["Normal"], fontName=fn, fontSize=9,
                                 textColor=rl_colors.grey, spaceAfter=4)
        s_h2    = ParagraphStyle("H2", parent=styles["Heading2"], fontName=fn, fontSize=13, spaceAfter=4)
        s_body  = ParagraphStyle("B", parent=styles["Normal"], fontName=fn, fontSize=9, leading=14)

        story = [
            Paragraph("AI Citation Analysis Report", s_title),
            Paragraph(f"Brand: {brand_name}  |  {target_url}", s_sub),
            Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", s_sub),
            Spacer(1, 0.6*rl_cm),
            Paragraph("Citation Rate by Question", s_h2),
        ]

        tdata = [["Question", "GPT %", "Gemini %", "Avg %"]]
        for q, r in zip(questions, all_results):
            _g = lambda k, rr=r: (rr.get(k) if isinstance(rr, dict) else getattr(rr, k, None)) or 0
            tdata.append([
                (q[:55] + "...") if len(q) > 58 else q,
                f"{_g('gpt_rate'):.1f}%",
                f"{_g('gemini_rate'):.1f}%",
                f"{_g('avg_rate'):.1f}%",
            ])
        tbl = Table(tdata, colWidths=[10*rl_cm, 2.3*rl_cm, 2.3*rl_cm, 2.3*rl_cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), rl_colors.HexColor("#333333")),
            ("TEXTCOLOR",     (0, 0), (-1,  0), rl_colors.white),
            ("FONTNAME",      (0, 0), (-1, -1), fn),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [rl_colors.white, rl_colors.HexColor("#F5F5F5")]),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5*rl_cm))

        if strategy:
            diags = strategy.get("diagnoses", [])
            if diags:
                story.append(Paragraph("Diagnosis", s_h2))
                for d in diags[:3]:
                    story.append(Paragraph(f"• {d}", s_body))
                story.append(Spacer(1, 0.3*rl_cm))

            kws = strategy.get("keywords", [])
            if kws:
                story.append(Paragraph("Blue Ocean Keywords", s_h2))
                for kw in kws[:5]:
                    kw_text = kw.get("keyword", str(kw)) if isinstance(kw, dict) else str(kw)
                    story.append(Paragraph(f"• {kw_text}", s_body))
                story.append(Spacer(1, 0.3*rl_cm))

            guides = strategy.get("geo_guides", [])
            if guides:
                story.append(Paragraph("GEO Optimization Guide", s_h2))
                for g in guides[:5]:
                    clean_g = re.sub(r"\*\*(.*?)\*\*", r"\1", g)
                    story.append(Paragraph(f"• {clean_g[:250]}", s_body))

        doc.build(story)
    except Exception as e:
        logger.warning(f"PDF build error: {e}")
        return b""
    return buf.getvalue()


# ── 기능 7: Sentiment & Highlight ─────────────────────────────────────────────

_POS_KW = ["추천", "좋은", "우수", "최고", "효과적", "편리", "만족", "탁월", "뛰어난", "강력",
           "최적", "선도", "인정", "신뢰", "안정"]
_NEG_KW = ["단점", "문제", "불편", "비싼", "어려운", "느린", "실패", "부족", "아쉬운", "낮은",
           "부정적", "위험", "취약", "불만", "단순"]

def classify_sentiment(text: str, brand_name: str) -> str:
    if not text:
        return "미언급"
    variants = build_brand_variants(brand_name) if brand_name else []
    mentioned = any(v and v.lower() in text.lower() for v in variants)
    p = sum(1 for kw in _POS_KW if kw in text)
    n = sum(1 for kw in _NEG_KW if kw in text)
    if not mentioned:
        return "미언급"
    if p > n:
        return "긍정"
    if n > p:
        return "부정"
    return "중립"

def highlight_brand(text: str, variants: list) -> str:
    if not text or not variants:
        return text
    for v in variants:
        if not v:
            continue
        try:
            text = re.sub(
                re.escape(v),
                f'<mark style="background:#F59E0B;color:#111;padding:0 3px;border-radius:3px;">{v}</mark>',
                text, flags=re.IGNORECASE,
            )
        except Exception:
            pass
    return text


# ── 기능 9: GEO Editor ────────────────────────────────────────────────────────

_GEO_DIMS = {
    "brand_clarity":     "브랜드 명확성",
    "faq_structure":     "FAQ 구조",
    "specificity":       "구체성·수치",
    "authority_signals": "권위 신호",
    "keyword_density":   "키워드 밀도",
}

_GEO_SYSTEM = "당신은 GEO(Generative Engine Optimization) 전문가입니다. JSON만 출력합니다."

def run_geo_editor_analysis(content_text: str, brand_name: str,
                             client_gpt, client_gemini, model_gpt: str) -> dict:
    prompt = f"""아래 콘텐츠를 GEO 관점에서 5개 항목 각 0-20점으로 평가해주세요.

브랜드명: {brand_name}

[콘텐츠]
{content_text[:3000]}

JSON 형식으로만 출력 (설명 없음):
{{
  "brand_clarity":     {{"score": 0-20, "feedback": "한 줄 피드백"}},
  "faq_structure":     {{"score": 0-20, "feedback": "한 줄 피드백"}},
  "specificity":       {{"score": 0-20, "feedback": "한 줄 피드백"}},
  "authority_signals": {{"score": 0-20, "feedback": "한 줄 피드백"}},
  "keyword_density":   {{"score": 0-20, "feedback": "한 줄 피드백"}},
  "improvement":       "전체 개선 제안 200자 이내",
  "rewritten_sample":  "개선된 첫 단락 샘플 (200자 이내)"
}}"""

    result_str = ""
    with CaptureError("geo_editor", log_level="warning"):
        if client_gpt:
            result_str = call_gpt(client_gpt, prompt, system=_GEO_SYSTEM,
                                  max_tokens=1000, model=model_gpt, temperature=0.2)
        elif client_gemini:
            result_str = call_gemini(client_gemini, prompt, max_tokens=1000, temperature=0.2)

    with CaptureError("geo_parse", log_level="warning"):
        m = re.search(r"\{.*\}", result_str, re.DOTALL)
        if m:
            return json.loads(m.group())

    return {d: {"score": 10, "feedback": "분석 실패"} for d in _GEO_DIMS} | {
        "improvement": "API 키를 확인하세요.",
        "rewritten_sample": "",
    }

def render_geo_radar(geo_result: dict):
    vals   = [geo_result.get(k, {}).get("score", 0) if isinstance(geo_result.get(k), dict)
              else int(geo_result.get(k, 0)) for k in _GEO_DIMS]
    labels = list(_GEO_DIMS.values())
    total  = sum(vals)

    col_r, col_s = st.columns([3, 2])
    with col_r:
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#555555", width=2),
            fillcolor="rgba(80,80,80,0.25)",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 20],
                                tickfont=dict(size=9, color=_plot_font),
                                gridcolor=_plot_grid),
                angularaxis=dict(tickfont=dict(size=11, color=_plot_font)),
                bgcolor=_plot_paper,
            ),
            plot_bgcolor=_plot_bg, paper_bgcolor=_plot_paper,
            font=dict(family="Plus Jakarta Sans", color=_plot_font),
            margin=dict(t=30, b=30, l=30, r=30), height=320,
        )
        _chart_counter["n"] += 1
        st.plotly_chart(fig, use_container_width=True, key=f"geo_radar_{_chart_counter['n']}")

    with col_s:
        score_color = "#10B981" if total >= 80 else "#F59E0B" if total >= 50 else "#EF4444"
        st.markdown(f"""
        <div style="text-align:center;padding:20px 10px;">
            <div style="font-size:3rem;font-weight:800;color:{score_color};">{total}</div>
            <div style="font-size:.85rem;color:{_text_muted};margin-bottom:16px;">/ 100점</div>
        </div>""", unsafe_allow_html=True)
        for dim_key, dim_label in _GEO_DIMS.items():
            dim_val = geo_result.get(dim_key, {})
            sc  = dim_val.get("score", 0) if isinstance(dim_val, dict) else 0
            fb  = dim_val.get("feedback", "") if isinstance(dim_val, dict) else ""
            bar_w = int(sc / 20 * 100)
            bar_c = "#10B981" if sc >= 15 else "#F59E0B" if sc >= 8 else "#EF4444"
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;font-size:.78rem;">
                    <span style="font-weight:600;color:{_text};">{dim_label}</span>
                    <span style="color:{bar_c};font-weight:700;">{sc}/20</span>
                </div>
                <div style="background:{_border};border-radius:4px;height:6px;margin:3px 0;">
                    <div style="background:{bar_c};width:{bar_w}%;height:6px;border-radius:4px;"></div>
                </div>
                <div style="font-size:.72rem;color:{_text_muted};">{fb}</div>
            </div>""", unsafe_allow_html=True)

    improvement = geo_result.get("improvement", "")
    rewritten   = geo_result.get("rewritten_sample", "")
    if improvement:
        st.markdown(f"""
        <div style="background:{_bg2};border-left:3px solid #F59E0B;border-radius:8px;padding:12px 16px;margin-top:10px;">
            <div style="font-size:.8rem;font-weight:700;color:{_text_muted};margin-bottom:4px;">개선 제안</div>
            <div style="font-size:.87rem;color:{_text};line-height:1.7;">{improvement}</div>
        </div>""", unsafe_allow_html=True)
    if rewritten:
        st.markdown(f"""
        <div style="background:{_card};border:1px solid {_border};border-radius:8px;padding:12px 16px;margin-top:10px;">
            <div style="font-size:.8rem;font-weight:700;color:{_text_muted};margin-bottom:4px;">개선 샘플</div>
            <div style="font-size:.87rem;color:{_text};line-height:1.7;">{rewritten}</div>
        </div>""", unsafe_allow_html=True)


# ── Demo data ──────────────────────────────────────────────────────────────────

_DEMO_SCENARIOS = {
    "naver.com": {
        "brand": "네이버", "industry": "포털/검색 서비스",
        "questions": ["국내 최고 포털 사이트 추천해줘?","한국어 지도 서비스 어디가 제일 좋아?","블로그 플랫폼 비교해줘?","뉴스 검색 사이트 추천?","국내 쇼핑 검색 비교해줘?"],
        "results": [
            {"gpt_rate":58,"gemini_rate":62,"avg_rate":60,"gpt_hits":29,"gemini_hits":31,"n":50},
            {"gpt_rate":44,"gemini_rate":51,"avg_rate":47.5,"gpt_hits":22,"gemini_hits":25,"n":50},
            {"gpt_rate":22,"gemini_rate":18,"avg_rate":20,"gpt_hits":11,"gemini_hits":9,"n":50},
            {"gpt_rate":39,"gemini_rate":43,"avg_rate":41,"gpt_hits":19,"gemini_hits":21,"n":50},
            {"gpt_rate":31,"gemini_rate":27,"avg_rate":29,"gpt_hits":15,"gemini_hits":13,"n":50},
        ],
    },
    "coupang.com": {
        "brand": "쿠팡", "industry": "이커머스/로켓배송",
        "questions": ["가장 빠른 배송 쇼핑몰 추천?","로켓배송 당일 수령 가능한 곳?","새벽배송 신선식품 어디가 좋아?","온라인 쇼핑몰 멤버십 혜택 비교?","무료배송 기준 낮은 쇼핑몰?"],
        "results": [
            {"gpt_rate":71,"gemini_rate":68,"avg_rate":69.5,"gpt_hits":35,"gemini_hits":34,"n":50},
            {"gpt_rate":65,"gemini_rate":59,"avg_rate":62,"gpt_hits":32,"gemini_hits":29,"n":50},
            {"gpt_rate":38,"gemini_rate":42,"avg_rate":40,"gpt_hits":19,"gemini_hits":21,"n":50},
            {"gpt_rate":52,"gemini_rate":48,"avg_rate":50,"gpt_hits":26,"gemini_hits":24,"n":50},
            {"gpt_rate":29,"gemini_rate":33,"avg_rate":31,"gpt_hits":14,"gemini_hits":16,"n":50},
        ],
    },
    "default": {
        "brand": "샘플브랜드", "industry": "디지털 서비스",
        "questions": ["이 분야 서비스 추천해줘?","경쟁 서비스 대비 장점 비교?","가격 비교 어디가 저렴해?","초보자도 쓰기 좋은 서비스?","고객 지원 잘 되는 곳?"],
        "results": [
            {"gpt_rate":7,"gemini_rate":5,"avg_rate":6,"gpt_hits":3,"gemini_hits":2,"n":50},
            {"gpt_rate":4,"gemini_rate":8,"avg_rate":6,"gpt_hits":2,"gemini_hits":4,"n":50},
            {"gpt_rate":12,"gemini_rate":9,"avg_rate":10.5,"gpt_hits":6,"gemini_hits":4,"n":50},
            {"gpt_rate":3,"gemini_rate":6,"avg_rate":4.5,"gpt_hits":1,"gemini_hits":3,"n":50},
            {"gpt_rate":15,"gemini_rate":11,"avg_rate":13,"gpt_hits":7,"gemini_hits":5,"n":50},
        ],
    },
}

_DEMO_STRATEGY = {
    "competitors": [
        {"rank":1,"domain":"wikipedia.org","brand_name":"Wikipedia","reason":"중립적 참조 정보","position":"업계1위"},
        {"rank":2,"domain":"namu.wiki","brand_name":"나무위키","reason":"한국어 위키","position":"신흥강자"},
        {"rank":3,"domain":"tistory.com","brand_name":"티스토리","reason":"SEO 블로그","position":"틈새전문"},
        {"rank":4,"domain":"brunch.co.kr","brand_name":"브런치","reason":"전문가 롱폼","position":"틈새전문"},
        {"rank":5,"domain":"medium.com","brand_name":"Medium","reason":"영문 고품질","position":"신흥강자"},
    ],
    "diagnoses": ["구조화 데이터 마크업 부재로 AI 맥락 파악 어려움","FAQ 섹션 없어 Q&A 기반 인용 기회 손실","핵심 키워드 밀도 경쟁사 대비 40% 낮음"],
    "keywords": ["AI 인용 최적화 전략 2025","GEO 적용 방법","챗봇 검색 브랜드 노출","LLM 친화적 콘텐츠","AI 답변 출처 선택 조건"],
    "geo_guides": [
        "1. FAQ 블록 추가\n홈페이지에 Q&A 형식 서비스 설명 섹션을 추가하세요.",
        "2. 구조화 데이터 적용\nJSON-LD로 Organization, FAQPage 스키마를 삽입하세요.",
        "3. 핵심 가치 제안 최상단 배치\n명확한 정의 문장으로 AI가 권위 출처로 인식하게 하세요.",
    ],
}

def get_demo(url: str) -> dict:
    domain = extract_domain(url).lower()
    for k, v in _DEMO_SCENARIOS.items():
        if k != "default" and k in domain:
            return v
    return _DEMO_SCENARIOS["default"]

_chart_counter = {"n": 0}


# ── Charts ─────────────────────────────────────────────────────────────────────

def render_bar_chart(results: list, questions: list, title: str = "AI 엔진별 인용 점유율"):
    if not results:
        return
    _chart_counter["n"] += 1
    short_q = [q[:20] + "..." if len(q) > 22 else q for q in questions]

    def _get(r, k):
        return r.get(k) if isinstance(r, dict) else getattr(r, k, None)

    gpt_rates    = [_get(r, "gpt_rate")    for r in results]
    gemini_rates = [_get(r, "gemini_rate") for r in results]

    fig = go.Figure()
    if any(v is not None for v in gpt_rates):
        fig.add_trace(go.Bar(
            name="GPT", x=short_q, y=[v or 0 for v in gpt_rates],
            marker=dict(color="#444444" if _dark else "#111111"),
            text=[f"{v:.1f}%" if v is not None else "" for v in gpt_rates],
            textposition="outside", textfont=dict(color=_plot_font),
        ))
    if any(v is not None for v in gemini_rates):
        fig.add_trace(go.Bar(
            name="Gemini", x=short_q, y=[v or 0 for v in gemini_rates],
            marker=dict(color="#888888"),
            text=[f"{v:.1f}%" if v is not None else "" for v in gemini_rates],
            textposition="outside", textfont=dict(color=_plot_font),
        ))

    all_vals = [v for v in gpt_rates + gemini_rates if v is not None]
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=_plot_font, family="Plus Jakarta Sans"), x=0),
        barmode="group", bargap=0.25,
        plot_bgcolor=_plot_bg, paper_bgcolor=_plot_paper,
        font=dict(family="Plus Jakarta Sans", color=_plot_font),
        xaxis=dict(tickfont=dict(size=11, color=_plot_font), gridcolor=_plot_grid),
        yaxis=dict(
            title="인용 점유율 (%)", ticksuffix="%",
            gridcolor=_plot_grid, tickfont=dict(color=_plot_font),
            range=[0, max(max(all_vals, default=0) + 15, 20)]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=_plot_font)),
        margin=dict(t=60, b=40, l=50, r=20), height=380,
    )
    st.plotly_chart(fig, use_container_width=True, key=f'chart_{_chart_counter["n"]}')


# ── render_strategy ────────────────────────────────────────────────────────────

def render_strategy(strategy: dict, target_url: str, sim_mentions: dict = None):
    domain = extract_domain(target_url)
    sim_mentions = sim_mentions or {}

    pos_colors = {
        "업계1위":"#10B981","업계 1위":"#10B981",
        "신흥강자":"#F59E0B","신흥 강자":"#F59E0B",
        "틈새전문":"#6366F1","틈새 전문":"#6366F1",
    }

    st.markdown("### 🏆 AI 인용 경쟁 현황 (TOP 5)")

    competitors = strategy.get("competitors", [])[:5]

    if not competitors and sim_mentions:
        sorted_m = sorted(sim_mentions.items(),
                          key=lambda x: x[1].get("mentions", 0), reverse=True)[:5]
        competitors = [
            {
                "rank": i + 1,
                "brand_name": brand,
                "domain": next(
                    (u.split("/")[2] for u in data.get("urls", []) if u.startswith("http")), ""
                ),
                "reason": f"AI 응답 {data.get('mentions', 0)}회 언급",
                "position": ["업계1위", "신흥강자", "신흥강자", "틈새전문", "틈새전문"][i],
            }
            for i, (brand, data) in enumerate(sorted_m) if brand
        ]
        if competitors:
            st.caption("시뮬레이션 AI 응답에서 집계된 언급 기준 (전략 분석 보완)")

    if competitors:
        for comp in competitors:
            r   = comp.get("rank", "?")
            d   = comp.get("domain", "")
            b   = comp.get("brand_name", d)
            reason = comp.get("reason", "")
            pos    = comp.get("position", comp.get("market_position", ""))
            is_t   = domain.lower() in d.lower() if d else False
            if _dark:
                bg = "linear-gradient(135deg,#2A2A2A,#333)" if is_t else "#1E1E1E"
                bd = "#888" if is_t else "#333"
            else:
                bg = "linear-gradient(135deg,#EEE,#E0E0E0)" if is_t else "#F8F8F8"
                bd = "#111" if is_t else "#DDD"
            lbl = " ← 내 사이트" if is_t else ""
            pc  = pos_colors.get(pos, "#888")
            pb  = (
                f'<span style="background:{pc};color:white;padding:2px 8px;border-radius:20px;'
                f'font-size:.72rem;font-weight:700;margin-left:8px;">{pos}</span>'
                if pos else ""
            )
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                padding:11px 16px;border-radius:10px;margin:5px 0;
                background:{bg};border:1.5px solid {bd};">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:28px;height:28px;border-radius:8px;
                        background:linear-gradient(135deg,#111,#444);
                        color:white;font-weight:700;font-size:.8rem;
                        display:flex;align-items:center;justify-content:center;">{r}</div>
                    <span style="font-weight:{'700' if is_t else '500'};color:{_text};font-size:.9rem;">
                        {b}{(' <span style="color:'+_text_muted+';font-size:.78rem;">('+d+')</span>') if d else ""}{lbl}{pb}
                    </span>
                </div>
                <span style="color:{_text_muted};font-size:.8rem;">{reason}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("시뮬레이션을 실행하면 AI 응답에서 경쟁사 데이터가 집계됩니다.")

    st.markdown(
        f"<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,{_border},transparent);margin:20px 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🔬 인용 실패 원인 진단")
    _icons = ["❌", "⚡", "🔧"]
    for i, d in enumerate(strategy.get("diagnoses", [])):
        col = ["#111" if not _dark else "#DDD", "#555", "#888"][i % 3]
        st.markdown(f"""
        <div class="strategy-item" style="border-left:4px solid {col};">
            <span>{_icons[i % len(_icons)]} {d}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f"<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,{_border},transparent);margin:20px 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🌊 블루오션 키워드")
    keywords = strategy.get("keywords", [])
    if keywords:
        st.caption("경쟁사 대비 AI 인용 가능성이 높은 틈새 키워드입니다. 콘텐츠 및 메타 태그에 활용하세요.")
        for kw in keywords:
            kw_text = kw.get("keyword", str(kw)) if isinstance(kw, dict) else str(kw)
            if not kw_text or kw_text in ("분석 중 오류",):
                continue
            st.markdown(f"""
            <div class="blue-ocean-item">
                <span class="label">🎯 추천</span>
                <span class="kw">{kw_text}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("키워드 분석을 완료하지 못했습니다. API 키를 확인하세요.")

    st.markdown(
        f"<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,{_border},transparent);margin:20px 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 GEO 최적화 가이드")
    geo_guides = strategy.get("geo_guides", [])
    if geo_guides:
        st.caption("아래 항목을 적용하면 AI가 내 사이트를 더 자주 인용합니다.")
        for i, g in enumerate(geo_guides):
            g_html = re.sub(r'\*\*(.*?)\*\*', rf'<strong style="color:{_text};">\1</strong>', g)
            g_html = g_html.replace('\n', '<br>')
            priority_color = ["#10B981","#F59E0B","#6366F1"][i % 3]
            st.markdown(f"""
            <div class="geo-guide-item" style="border-left:3px solid {priority_color};">
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="min-width:30px;height:30px;border-radius:8px;flex-shrink:0;
                        background:linear-gradient(135deg,#111,#444);color:white;font-weight:800;
                        font-size:.85rem;display:flex;align-items:center;justify-content:center;">{i+1}</div>
                    <div class="geo-text">{g_html}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("GEO 가이드를 생성하지 못했습니다.")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🔍</span>
        <h2>AI Citation Analyzer</h2>
        <p>v4.0 - 통합 분석</p>
    </div>""", unsafe_allow_html=True)

    if st.button("🌙 다크 모드" if not _dark else "☀️ 라이트 모드", key="btn_dark", use_container_width=True):
        st.session_state["dark_mode"] = not _dark
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    openai_key        = st.text_input("OpenAI API Key",  type="password", placeholder="sk-...")
    gemini_key        = st.text_input("Gemini API Key",  type="password", placeholder="AIza...")
    gpt_model         = st.selectbox("GPT 모델",  ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    gemini_model_name = st.selectbox("Gemini 모델", ["models/gemini-2.0-flash", "models/gemini-flash-latest"])

    st.markdown("---")

    sim_count    = st.slider("시뮬레이션 횟수", 10, 100, 30, 10, help="횟수가 낮을수록 빠르고 저렴합니다")
    market_scope = st.radio("경쟁사 범위", ["국내 (대한민국)", "글로벌"], horizontal=True)

    st.markdown("---")

    gpt_ok    = bool(openai_key and openai_key.startswith("sk-"))
    gemini_ok = bool(gemini_key and len(gemini_key) > 10)

    c1, c2 = st.columns(2)
    c1.markdown("🟢 **GPT**"    if gpt_ok    else "⚪ **GPT**")
    c2.markdown("🟢 **Gemini**" if gemini_ok else "🔴 **Gemini**")

    tracker: CostTracker = st.session_state["cost_tracker"]
    cost_summary = tracker.summary()
    if cost_summary["api_calls"] > 0:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);
            border-radius:8px;padding:10px 12px;margin-top:8px;font-size:.75rem;color:rgba(255,255,255,.85);">
            💰 누적 비용: ~${cost_summary['estimated_usd']:.4f}<br>
            🔢 API 호출: {cost_summary['api_calls']}회
        </div>""", unsafe_allow_html=True)

    cache_stats = _cache.stats()
    if cache_stats["entries"] > 0:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);
            border-radius:8px;padding:8px 12px;margin-top:6px;font-size:.73rem;color:rgba(255,255,255,.75);">
            ⚡ 캐시 히트율: {cache_stats['hit_rate']}% ({cache_stats['entries']}개 저장)
        </div>""", unsafe_allow_html=True)

    if st.button("🗑️ 캐시 초기화", key="btn_clear_cache", use_container_width=True):
        _cache._store.clear()
        st.session_state["cache_data"]   = {}
        st.session_state["cost_tracker"] = CostTracker()
        st.success("캐시 초기화 완료")

    st.markdown("---")
    if st.button("🎬 데모 실행", key="btn_demo_sidebar", use_container_width=True):
        st.session_state["run_demo"] = True
        st.rerun()

    # ── 기능 1: Google Sheets 연동 ──
    st.markdown("---")
    st.markdown("""<div style="font-size:.82rem;font-weight:700;color:rgba(255,255,255,.85);margin-bottom:6px;">
        📊 Google Sheets 연동</div>""", unsafe_allow_html=True)

    gs_sa = st.text_area(
        "Service Account JSON",
        value=st.session_state.get("sa_json_str", ""),
        placeholder='{"type":"service_account","project_id":"..."}',
        height=80, key="sa_json_input",
        help="GCP 서비스 계정 JSON 키를 붙여넣으세요",
    )
    st.session_state["sa_json_str"] = gs_sa

    gs_url = st.text_input(
        "Sheet URL",
        value=st.session_state.get("gsheets_url", ""),
        placeholder="https://docs.google.com/spreadsheets/d/...",
        key="gsheets_url_input",
    )
    st.session_state["gsheets_url"] = gs_url

    if gs_sa.strip() and gs_url.strip():
        if _GSPREAD_OK:
            st.markdown("""<div style="font-size:.72rem;color:#10B981;">🟢 연동 준비됨 (분석 후 자동 저장)</div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown("""<div style="font-size:.72rem;color:#F59E0B;">⚠️ gspread 패키지 미설치</div>""",
                        unsafe_allow_html=True)


# ── Clients ────────────────────────────────────────────────────────────────────

def get_clients():
    client_gpt = client_gemini = None
    if gpt_ok:
        with CaptureError("init_gpt", log_level="error"):
            import openai
            client_gpt = openai.OpenAI(api_key=openai_key)
    if gemini_ok:
        with CaptureError("init_gemini", log_level="error"):
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            client_gemini = genai.GenerativeModel(gemini_model_name)
    return client_gpt, client_gemini


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#111111,#1e1e1e,#2a2a2a);border-radius:20px;padding:36px 40px;margin-bottom:28px;box-shadow:0 4px 24px rgba(0,0,0,.4);">
    <div style="color:white!important;-webkit-text-fill-color:white!important;font-size:2rem!important;font-weight:800!important;margin:0!important;line-height:1.2!important;font-family:sans-serif;">
        🔍 AI 검색 점유율 분석 대시보드</div>
    <div style="color:rgba(255,255,255,.85)!important;-webkit-text-fill-color:rgba(255,255,255,.85)!important;font-size:1rem!important;margin:8px 0 0!important;font-family:sans-serif;">
        GPT &amp; Gemini AI 엔진에서 내 사이트 인용 점유율 측정 - 질문 직접 입력 · 비용 최적화 · 정밀 탐지</div>
</div>""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 엔진",  "GPT + Gemini", "2개 동시")
col2.metric("시뮬레이션", f"{sim_count}회", "적응형 조기종료")
col3.metric("API 연결",   f"{(1 if gpt_ok else 0)+(1 if gemini_ok else 0)}/2", "활성화")
col4.metric("누적 비용",  f"~${tracker.summary()['estimated_usd']:.4f}", "USD 추정")

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

tab_main, tab_hist = st.tabs(["인용 점유율 분석", "히스토리"])

client_gpt, client_gemini = get_clients()

# ── Tab: 분석 ──────────────────────────────────────────────────────────────────

with tab_main:

    st.markdown(f"""
    <div class="result-card" style="margin-bottom:20px;">
        <h4 style="margin:0 0 4px 0;">📝 분석 설정</h4>
        <p style="color:{_text_muted};font-size:.85rem;margin:0;">
            사이트 URL · 브랜드명을 입력하고, 분석할 질문을 최대 5개까지 직접 입력하세요.
        </p>
    </div>""", unsafe_allow_html=True)

    col_u, col_b = st.columns([2, 1])
    with col_u:
        url_input = st.text_input("🌐 사이트 URL *", placeholder="예) progress-roasup.co.kr", key="url_input")
    with col_b:
        brand_input = st.text_input("🏷️ 브랜드명 *", placeholder="예) 프로그레스미디어", key="brand_input",
                                    help="집계 데이터에 직접 반영되니 정확하게 입력해주세요!")
    if brand_input.strip():
        st.caption("브랜드명은 AI 응답 집계에 직접 반영됩니다. 정확한 브랜드명을 입력해주세요.")

    # ── 기능 2: 경쟁사 비교 분석 ──────────────────────────────────────────────
    with st.expander("🆚 경쟁사 비교 분석 (선택)", expanded=False):
        st.caption("비교할 경쟁사 URL과 브랜드명을 입력하면 동일 질문으로 병렬 시뮬레이션을 실행합니다.")
        comp_urls_new   = list(st.session_state["comp_urls"])
        comp_brands_new = list(st.session_state["comp_brands"])
        for ci in range(3):
            cc1, cc2 = st.columns([3, 2])
            with cc1:
                comp_urls_new[ci] = st.text_input(
                    f"경쟁사 {ci+1} URL", value=comp_urls_new[ci],
                    placeholder="예) competitor.co.kr", key=f"comp_url_{ci}",
                )
            with cc2:
                comp_brands_new[ci] = st.text_input(
                    f"브랜드명", value=comp_brands_new[ci],
                    placeholder="예) 경쟁사A", key=f"comp_brand_{ci}",
                )
        st.session_state["comp_urls"]   = comp_urls_new
        st.session_state["comp_brands"] = comp_brands_new

        q_count_now = st.session_state["q_count"]
        qs_now      = [st.session_state["questions_list"][i]
                       for i in range(q_count_now)
                       if i < len(st.session_state["questions_list"])
                       and st.session_state["questions_list"][i].strip()]

        comp_btn = st.button("🆚 경쟁사 비교 실행", key="btn_comp_run", use_container_width=True)
        if comp_btn:
            valid_pairs = [(u.strip(), b.strip())
                           for u, b in zip(comp_urls_new, comp_brands_new) if u.strip() and b.strip()]
            if not valid_pairs:
                st.warning("경쟁사 URL과 브랜드명을 최소 1개 입력하세요.")
            elif not qs_now:
                st.warning("비교에 사용할 질문을 먼저 입력하세요.")
            elif not gpt_ok and not gemini_ok:
                st.error("사이드바에서 API 키를 입력하세요.")
            else:
                with st.spinner("경쟁사 시뮬레이션 실행 중..."):
                    _comp_res = run_competitor_comparison(
                        client_gpt, client_gemini,
                        [p[0] for p in valid_pairs],
                        [p[1] for p in valid_pairs],
                        qs_now, sim_count, gpt_model, tracker,
                    )
                    st.session_state["comp_results"] = _comp_res
                    save_cache()
                    st.session_state["cost_tracker"] = tracker

        if st.session_state.get("comp_results"):
            st.markdown("#### 경쟁사 비교 결과")
            render_competitor_bar_chart(st.session_state["comp_results"], qs_now or ["Q1", "Q2", "Q3"])
            # Summary table
            _cr = st.session_state["comp_results"]
            for brand, res_list in _cr.items():
                if not res_list:
                    continue
                avgs = []
                for r in res_list:
                    v = (r.get("avg_rate") if isinstance(r, dict) else getattr(r, "avg_rate", None)) or 0
                    avgs.append(v)
                overall = sum(avgs) / len(avgs) if avgs else 0
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 14px;border-radius:8px;margin:4px 0;
                    background:{_card};border:1px solid {_border};">
                    <span style="font-weight:600;color:{_text};">{brand}</span>
                    <span style="color:{_text_muted};font-size:.85rem;">전체 평균 인용율: <strong style="color:{_text};">{overall:.1f}%</strong></span>
                </div>""", unsafe_allow_html=True)

    # ── 기능 3: AI 질문 추천 ──────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns([3, 2])
    with rec_col1:
        if st.button("💡 AI 질문 추천 받기", key="btn_rec_qs", use_container_width=True):
            if not gpt_ok and not gemini_ok:
                st.warning("API 키를 먼저 입력하세요.")
            else:
                _u = url_input.strip() or ""
                _b = brand_input.strip() or ""
                with st.spinner("AI가 질문을 추천하는 중..."):
                    rec_qs = recommend_questions(_u, _b, client_gpt, client_gemini, gpt_model)
                st.session_state["recommended_qs"] = rec_qs
                st.session_state["show_rec_qs"]    = True
                st.rerun()

    if st.session_state.get("show_rec_qs") and st.session_state.get("recommended_qs"):
        with st.expander("💡 추천 질문 선택", expanded=True):
            st.caption("체크한 질문이 아래 입력창에 추가됩니다.")
            selected: list = []
            for qi, rq in enumerate(st.session_state["recommended_qs"]):
                if st.checkbox(rq, key=f"rec_q_chk_{qi}"):
                    selected.append(rq)
            if st.button("선택한 질문 적용", key="btn_apply_rec"):
                current_count = st.session_state["q_count"]
                current_list  = st.session_state["questions_list"]
                for sq in selected:
                    if current_count >= 5:
                        break
                    if len(current_list) <= current_count:
                        current_list.append(sq)
                    else:
                        current_list[current_count] = sq
                    current_count += 1
                st.session_state["q_count"]        = min(5, current_count)
                st.session_state["questions_list"] = current_list
                st.session_state["show_rec_qs"]    = False
                st.rerun()
            if st.button("닫기", key="btn_close_rec"):
                st.session_state["show_rec_qs"] = False
                st.rerun()

    # ── Question inputs ────────────────────────────────────────────────────────
    q_count = st.session_state["q_count"]
    st.markdown(f"""<div style="font-size:.82rem;font-weight:700;color:{_text};margin-bottom:10px;">
        🎯 분석할 질문 <span style="color:{_text_muted};font-size:.76rem;font-weight:400;margin-left:6px;">{q_count}/5개</span>
    </div>""", unsafe_allow_html=True)

    questions_input = []
    for idx in range(q_count):
        col_q, col_del = st.columns([10, 1])
        with col_q:
            q = st.text_input(
                f"Q{idx+1}",
                value=st.session_state["questions_list"][idx] if idx < len(st.session_state["questions_list"]) else "",
                placeholder="예) 퍼포먼스 마케팅 대행사 ROAS 잘 나오는 곳 추천해줘?" if idx == 0 else f"질문 {idx+1}번",
                key=f"q_dyn_{idx}", label_visibility="visible"
            )
            if idx < len(st.session_state["questions_list"]):
                st.session_state["questions_list"][idx] = q
            if q.strip():
                questions_input.append(q.strip())
        with col_del:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if q_count > 1:
                if st.button("X", key=f"del_q_{idx}", help="이 질문 삭제"):
                    st.session_state["questions_list"].pop(idx)
                    st.session_state["q_count"] = max(1, q_count - 1)
                    st.rerun()

    if q_count < 5:
        if st.button(f"+ 질문 추가 ({q_count}/5)", key="btn_add_q"):
            st.session_state["questions_list"].append("")
            st.session_state["q_count"] = min(5, q_count + 1)
            st.rerun()
    else:
        st.caption("질문 5개 모두 입력됨 (최대)")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    col_run, col_clear = st.columns([4, 1])
    with col_run:
        run_btn = st.button("🚀 분석 시작", key="btn_run", use_container_width=True)
    with col_clear:
        if st.button("🗑️ 초기화", key="btn_reset", use_container_width=True):
            for k in ["url_input", "brand_input"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state["q_count"] = 1
            st.session_state["questions_list"] = [""]
            st.rerun()

    # ── Demo block ─────────────────────────────────────────────────────────────
    if st.session_state.get("run_demo"):
        st.session_state["run_demo"] = False
        demo_url = normalize_url(url_input.strip() if url_input.strip() else "naver.com")
        dd = get_demo(demo_url)
        st.info(f"🎬 데모 모드 - {dd['brand']} ({dd['industry']}) 샘플 데이터")
        st.markdown("---")
        render_bar_chart(dd["results"], dd["questions"], f"[데모] {dd['brand']} 인용 점유율")
        st.markdown("### 📋 질문별 상세 결과")
        for i, (q, r) in enumerate(zip(dd["questions"], dd["results"])):
            avg = r.get("avg_rate", 0)
            badge_cls = "share-badge-high" if avg >= 30 else "share-badge-mid" if avg >= 10 else "share-badge-low"
            with st.expander(f"Q{i+1}. {q} - 평균 {avg:.1f}%", expanded=(i == 0)):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("GPT",      f"{r.get('gpt_rate', 0)}%")
                c2.metric("Gemini",   f"{r.get('gemini_rate', 0)}%")
                c3.metric("평균",     f"{avg:.1f}%")
                c4.metric("GPT 히트", f"{r.get('gpt_hits', 0)}/{r.get('n', 50)}회")
                st.markdown(f'<span class="{badge_cls}">점유율 {avg:.1f}%</span>', unsafe_allow_html=True)
        st.markdown("---")
        render_strategy(_DEMO_STRATEGY, demo_url)

    # ── Main analysis run ──────────────────────────────────────────────────────
    elif run_btn:
        errors = []
        if not url_input.strip():   errors.append("사이트 URL")
        if not brand_input.strip(): errors.append("브랜드명")
        if not questions_input:     errors.append("질문 (최소 1개)")

        if errors:
            st.error(f"필수 입력 누락: {', '.join(errors)}")
        elif not gpt_ok and not gemini_ok:
            st.error("사이드바에서 API 키를 입력하세요.")
        else:
            target_url = normalize_url(url_input.strip())
            domain     = extract_domain(target_url)
            brand_name = brand_input.strip()

            biz_info = BusinessInfo(
                brand_name=brand_name,
                industry="",
                industry_category="기타",
                core_product="",
                target_audience="잠재 고객",
                confidence="high",
                crawl_tier=0,
            )

            st.markdown("---")
            prog = st.progress(0)
            stat = st.empty()

            stat.markdown(f"⚡ **{len(questions_input)}개 질문 × {sim_count}회 시뮬레이션 중...**")
            prog.progress(0.1)

            t0 = time.time()
            all_results: list = run_all_simulations(
                client_gpt, client_gemini,
                questions_input, target_url,
                model_gpt=gpt_model, n=sim_count,
                biz_info=biz_info.to_dict(),
                tracker=tracker,
                use_cache=True,
            )
            elapsed_sim = int(time.time() - t0)
            prog.progress(0.6)

            sim_mentions = _aggregate_sim_mentions(all_results)
            save_cache()
            st.session_state["cost_tracker"] = tracker
            stat.success(f"시뮬레이션 완료 ({elapsed_sim}초) | 추정비용 ~${tracker.summary()['estimated_usd']:.4f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("브랜드",    brand_name)
            c2.metric("질문 수",   f"{len(questions_input)}개")
            c3.metric("분석 엔진", f"{(1 if client_gpt else 0)+(1 if client_gemini else 0)}개")

            render_bar_chart(
                [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results],
                questions_input,
                f"'{brand_name}' AI 인용 점유율"
            )

            # ── 기능 7: 질문별 상세 + AI 응답 원문 ───────────────────────────
            st.markdown("### 📋 질문별 상세 결과")
            brand_variants = build_brand_variants(brand_name)
            _sent_colors = {
                "긍정": "#10B981", "부정": "#EF4444",
                "중립": "#F59E0B", "미언급": "#888888",
            }
            for i, (q, r) in enumerate(zip(questions_input, all_results)):
                _g = lambda k, rr=r: (rr.get(k) if isinstance(rr, dict) else getattr(rr, k, None))
                avg       = _g("avg_rate") or 0
                gpt_r     = _g("gpt_rate")
                gem_r     = _g("gemini_rate")
                gpt_hits  = _g("gpt_hits") or 0
                gem_hits  = _g("gemini_hits") or 0
                n_sim     = _g("n") or sim_count
                badge_cls = "share-badge-high" if avg >= 30 else "share-badge-mid" if avg >= 10 else "share-badge-low"

                with st.expander(f"Q{i+1}. {q[:55]} — 평균 {avg:.1f}%", expanded=(i == 0)):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("GPT",     f"{gpt_r:.1f}%" if gpt_r is not None else "-")
                    mc2.metric("Gemini",  f"{gem_r:.1f}%" if gem_r is not None else "-")
                    mc3.metric("평균",    f"{avg:.1f}%")
                    mc4.metric("GPT 히트", f"{gpt_hits}/{n_sim}회")
                    st.markdown(f'<span class="{badge_cls}">점유율 {avg:.1f}%</span>',
                                unsafe_allow_html=True)

                    gpt_samples    = (_g("gpt_samples")    or []) if isinstance(r, dict) else (getattr(r, "gpt_samples",    []) or [])
                    gemini_samples = (_g("gemini_samples") or []) if isinstance(r, dict) else (getattr(r, "gemini_samples", []) or [])

                    if gpt_samples or gemini_samples:
                        st.markdown(f"<div style='margin-top:10px;font-size:.82rem;font-weight:700;color:{_text};'>🤖 AI 응답 샘플</div>",
                                    unsafe_allow_html=True)

                    if gpt_samples:
                        with st.expander(f"GPT 응답 샘플 ({len(gpt_samples)}개)", expanded=False):
                            for j, sample in enumerate(gpt_samples[:3]):
                                sent  = classify_sentiment(sample, brand_name)
                                sc    = _sent_colors.get(sent, "#888")
                                hi    = highlight_brand(str(sample)[:600], brand_variants)
                                st.markdown(f"""
                                <div style="background:{_card};border:1px solid {_border};border-radius:8px;
                                    padding:10px 14px;margin:6px 0;">
                                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                                        <span style="font-size:.73rem;font-weight:700;color:{_text_muted};">GPT 응답 #{j+1}</span>
                                        <span style="background:{sc};color:white;padding:1px 8px;border-radius:12px;
                                            font-size:.7rem;font-weight:700;">{sent}</span>
                                    </div>
                                    <div style="font-size:.82rem;color:{_text};line-height:1.65;">{hi}</div>
                                </div>""", unsafe_allow_html=True)

                    if gemini_samples:
                        with st.expander(f"Gemini 응답 샘플 ({len(gemini_samples)}개)", expanded=False):
                            for j, sample in enumerate(gemini_samples[:3]):
                                sent  = classify_sentiment(sample, brand_name)
                                sc    = _sent_colors.get(sent, "#888")
                                hi    = highlight_brand(str(sample)[:600], brand_variants)
                                st.markdown(f"""
                                <div style="background:{_card};border:1px solid {_border};border-radius:8px;
                                    padding:10px 14px;margin:6px 0;">
                                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                                        <span style="font-size:.73rem;font-weight:700;color:{_text_muted};">Gemini 응답 #{j+1}</span>
                                        <span style="background:{sc};color:white;padding:1px 8px;border-radius:12px;
                                            font-size:.7rem;font-weight:700;">{sent}</span>
                                    </div>
                                    <div style="font-size:.82rem;color:{_text};line-height:1.65;">{hi}</div>
                                </div>""", unsafe_allow_html=True)

            prog.progress(0.7)
            stat.markdown("🧠 **전략 분석 생성 중...**")

            with CaptureError("strategy", log_level="warning") as s_ctx:
                strategy = run_strategy_analysis(
                    client_gpt, client_gemini,
                    biz_info=biz_info,
                    competitors=[],
                    sim_results=all_results,
                    questions=questions_input,
                    tracker=tracker,
                    target_url=target_url,
                    model_gpt=gpt_model,
                    market_scope=market_scope,
                    use_cache=True,
                )

            prog.progress(1.0)

            if s_ctx.ok and strategy:
                st.markdown("---")
                render_strategy(strategy, target_url, sim_mentions=sim_mentions)
                stat.success("전략 분석 완료!")
            else:
                if sim_mentions:
                    st.markdown("---")
                    render_strategy(
                        {"competitors": [], "diagnoses": [], "keywords": [], "geo_guides": []},
                        target_url, sim_mentions=sim_mentions,
                    )
                st.warning("전략 분석을 완료하지 못했습니다. API 키를 확인하세요.")
                strategy = {}

            # ── 기능 4: PDF 다운로드 ──────────────────────────────────────────
            st.markdown("---")
            if _REPORTLAB_OK:
                pdf_bytes = generate_pdf(
                    brand_name, target_url, questions_input,
                    [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results],
                    strategy or {},
                )
                if pdf_bytes:
                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        label="📄 PDF 리포트 다운로드",
                        data=pdf_bytes,
                        file_name=f"ai_citation_{brand_name}_{ts_str}.pdf",
                        mime="application/pdf",
                        key="btn_pdf_dl",
                        use_container_width=False,
                    )
                else:
                    st.caption("PDF 생성에 실패했습니다.")
            else:
                st.caption("PDF 기능을 사용하려면 `pip install reportlab` 을 실행하세요.")

            # ── 기능 1: GSheets 자동 저장 ─────────────────────────────────────
            _sa  = st.session_state.get("sa_json_str", "")
            _shu = st.session_state.get("gsheets_url", "")
            if _sa.strip() and _shu.strip() and _GSPREAD_OK:
                _gc = _get_gsheets_client(_sa)
                if _gc:
                    ok = save_to_gsheets(
                        _gc, _shu, brand_name, domain,
                        questions_input,
                        [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results],
                    )
                    if ok:
                        st.success("Google Sheets에 결과가 저장되었습니다.")
                    else:
                        st.warning("Google Sheets 저장 실패. JSON 키와 Sheet URL을 확인하세요.")

            # Persist for GEO editor
            st.session_state["last_all_results"]  = [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results]
            st.session_state["last_strategy"]     = strategy or {}
            st.session_state["last_questions"]    = questions_input
            st.session_state["last_url"]          = target_url
            st.session_state["last_brand"]        = brand_name
            st.session_state["last_sim_mentions"] = sim_mentions

            save_cache()
            st.session_state["cost_tracker"] = tracker

            st.session_state["history"].append({
                "ts":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "domain":    domain,
                "brand":     brand_name,
                "n_q":       len(questions_input),
                "questions": questions_input,
                "results":   [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results],
            })

    # ── 기능 1: GSheets 추이 차트 (persistent) ────────────────────────────────
    _sa_p  = st.session_state.get("sa_json_str", "")
    _shu_p = st.session_state.get("gsheets_url", "")
    _lb_p  = st.session_state.get("last_brand", "")
    if _sa_p.strip() and _shu_p.strip() and _lb_p and _GSPREAD_OK:
        with st.expander("📈 인용율 추이 (Google Sheets)", expanded=False):
            if st.button("추이 데이터 불러오기", key="btn_gs_load"):
                _gc_p = _get_gsheets_client(_sa_p)
                if _gc_p:
                    _recs = load_from_gsheets(_gc_p, _shu_p)
                    if _recs:
                        render_gsheets_trend(_recs, _lb_p)
                    else:
                        st.caption("저장된 데이터가 없습니다.")
                else:
                    st.warning("GSheets 연결 실패. 서비스 계정 JSON을 확인하세요.")

    # ── 기능 9: Content GEO 점수 편집기 (persistent) ──────────────────────────
    _lb9 = st.session_state.get("last_brand", "")
    with st.expander("✏️ 콘텐츠 GEO 점수 편집기", expanded=False):
        st.caption("홈페이지 또는 랜딩 페이지 본문을 붙여넣으면 AI가 GEO 점수(0-100)를 분석합니다.")
        geo_content = st.text_area(
            "분석할 콘텐츠 붙여넣기",
            height=180,
            placeholder="홈페이지 소개 텍스트, 서비스 설명 등을 붙여넣으세요...",
            key="geo_editor_input",
        )
        geo_brand_inp = st.text_input(
            "브랜드명 (GEO 분석용)",
            value=_lb9,
            placeholder="예) 프로그레스미디어",
            key="geo_brand_input",
        )
        if st.button("GEO 점수 분석", key="btn_geo_analyze", use_container_width=True):
            if not geo_content.strip():
                st.warning("분석할 콘텐츠를 입력하세요.")
            elif not gpt_ok and not gemini_ok:
                st.error("사이드바에서 API 키를 입력하세요.")
            else:
                with st.spinner("GEO 점수 분석 중..."):
                    geo_res = run_geo_editor_analysis(
                        geo_content, geo_brand_inp or _lb9 or "브랜드",
                        client_gpt, client_gemini, gpt_model,
                    )
                    st.session_state["geo_editor_result"] = geo_res

        if st.session_state.get("geo_editor_result"):
            st.markdown("#### GEO 점수 결과")
            render_geo_radar(st.session_state["geo_editor_result"])


# ── Tab: 히스토리 ──────────────────────────────────────────────────────────────

with tab_hist:
    history = st.session_state.get("history", [])
    if not history:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:3rem;margin-bottom:12px;">📭</div>
            <p style="color:{_text_muted};font-size:1rem;">아직 분석 기록이 없습니다.</p>
            <p style="color:{_text_muted};font-size:.85rem;">분석을 실행하면 여기에 기록됩니다.</p>
        </div>""", unsafe_allow_html=True)
    else:
        for h in reversed(history[-20:]):
            with st.expander(f"{h.get('ts','')} - {h.get('brand','')} | {h.get('n_q',0)}개 질문", expanded=False):
                if h.get("questions") and h.get("results"):
                    render_bar_chart(h["results"], h["questions"], f"{h.get('brand','')} 인용 점유율")
                    st.markdown("**📋 질문별 인용 현황**")
                    for i, (q, r) in enumerate(zip(h["questions"], h["results"])):
                        avg      = r.get("avg_rate") or 0
                        gpt_r    = r.get("gpt_rate")
                        gem_r    = r.get("gemini_rate")
                        gpt_hits = r.get("gpt_hits")
                        gem_hits = r.get("gemini_hits")
                        n_sim    = r.get("n", 50)
                        gpt_str  = f"{gpt_r:.1f}% ({gpt_hits}/{n_sim}회)" if gpt_r is not None else "-"
                        gem_str  = f"{gem_r:.1f}% ({gem_hits}/{n_sim}회)" if gem_r is not None else "-"
                        badge    = "share-badge-high" if avg >= 30 else "share-badge-mid" if avg >= 10 else "share-badge-low"
                        hc1,hc2,hc3,hc4 = st.columns([3,1,1,1])
                        hc1.caption(f"Q{i+1}. {q[:50]}")
                        hc2.caption(f"GPT: {gpt_str}")
                        hc3.caption(f"Gemini: {gem_str}")
                        hc4.markdown(f'<span class="{badge}">평균 {avg:.1f}%</span>', unsafe_allow_html=True)
                else:
                    st.json(h)


st.markdown(f"""
<div class="app-footer">
    AI Citation Analyzer v4.0 &nbsp;|&nbsp;
    <span style="color:{_text_muted};">Powered by GPT &amp; Gemini</span>
</div>""", unsafe_allow_html=True)
