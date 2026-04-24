# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 0: sys.path 설정 — 모든 import 최우선 실행
# Streamlit Cloud는 __file__ 기반이 실패하는 경우가 있으므로
# 3가지 전략을 모두 시도한다.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import sys
import os

def _setup_path():
    candidates = []

    # 전략 1: __file__ 절대경로 기반
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass

    # 전략 2: 현재 작업 디렉토리 (Streamlit Cloud = 프로젝트 루트)
    try:
        candidates.append(os.getcwd())
    except Exception:
        pass

    # 전략 3: /mount/src/ai-tool 하드코딩 (Streamlit Cloud 고정 경로)
    candidates.append("/mount/src/ai-tool")

    for p in candidates:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    # 실제로 core/ 가 있는 경로를 반환 (디버깅용)
    for p in candidates:
        if p and os.path.isdir(os.path.join(p, "core")):
            return p
    return candidates[0] if candidates else ""

_APP_ROOT = _setup_path()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: 일반 패키지 import
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import streamlit as st
import time
import re
import json
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import urlparse

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: core 모듈 import — 실패 시 UI에 진단 정보 표시
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from core.cache import get_cache          # TTLCache는 직접 import 하지 않음
    from core.logger import get_logger, CaptureError
    from core.citation import build_brand_variants, wilson_ci
    from core.ai_client import (
        call_gpt, call_gemini,
        run_all_simulations,
        CostTracker, SimResult,
    )
    from core.biz_analysis import run_strategy_analysis
    from core.schemas import BusinessInfo
except ImportError as _e:
    st.set_page_config(page_title="Import Error", page_icon="❌")
    st.error(f"**core 모듈 import 실패**")
    st.code(f"""
오류 유형 : {type(_e).__name__}
오류 내용 : {_e}

_APP_ROOT  : {_APP_ROOT}
os.getcwd(): {os.getcwd()}
sys.path   : {chr(10).join(sys.path[:6])}

core/ 존재 여부:
  __file__ 기준 : {os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '', 'core'))}
  CWD 기준       : {os.path.isdir(os.path.join(os.getcwd(), 'core'))}
  하드코딩 기준  : {os.path.isdir('/mount/src/ai-tool/core')}
""", language="text")
    st.info("위 정보를 개발자에게 전달해 주세요.")
    st.stop()

logger = get_logger("app")

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Citation Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────

def _init_session():
    defaults = {
        "dark_mode": False,
        "cache_data": {},
        "cost_tracker": CostTracker(),
        "analysis_results": None,   # 분석 결과 저장
        "questions_list": [""] * 5, # 질문 목록 (최대 10개)
        "n_questions": 5,           # 현재 질문 개수
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# 캐시 복원 — TTLCache 직접 import 없이 인스턴스에서 클래스 참조
_cache = get_cache()
if st.session_state["cache_data"]:
    try:
        _restored = type(_cache).from_serializable(st.session_state["cache_data"])
        _cache._store.update(_restored._store)
    except Exception:
        st.session_state["cache_data"] = {}

_dark = st.session_state["dark_mode"]

def save_cache():
    with CaptureError("save_cache", log_level="debug"):
        st.session_state["cache_data"] = _cache.to_serializable()

# ─────────────────────────────────────────────
# 테마 변수
# ─────────────────────────────────────────────

if _dark:
    _bg = "#0F0F0F"; _bg2 = "#1A1A1A"; _card = "#1E1E1E"
    _border = "#333"; _text = "#F0F0F0"; _text_muted = "#999"
    _primary = "#E0E0E0"; _shadow = "0 4px 24px rgba(0,0,0,.5)"
    _header_gr = "linear-gradient(135deg,#1A1A1A,#2A2A2A,#3A3A3A)"
    _sidebar_gr = "linear-gradient(180deg,#0F0F0F,#1A1A1A,#222)"
    _metric_bg = "#252525"; _input_bg = "rgba(255,255,255,.07)"
    _btn_gr = "linear-gradient(135deg,#333,#555)"
    _tab_bg = "#1E1E1E"; _tab_sel = "#333"
    _progress = "linear-gradient(90deg,#555,#888)"
    _accent = "#6366F1"
    _accent_light = "rgba(99,102,241,.15)"
else:
    _bg = "#F5F5F5"; _bg2 = "#EEE"; _card = "#FFF"
    _border = "#DDD"; _text = "#111"; _text_muted = "#666"
    _primary = "#111"; _shadow = "0 4px 24px rgba(0,0,0,.08)"
    _header_gr = "linear-gradient(135deg,#111,#333,#555)"
    _sidebar_gr = "linear-gradient(180deg,#111,#222,#333)"
    _metric_bg = "#FFF"; _input_bg = "#FFF"
    _btn_gr = "linear-gradient(135deg,#111,#444)"
    _tab_bg = "#FFF"; _tab_sel = "#111"
    _progress = "linear-gradient(90deg,#111,#555)"
    _accent = "#4F46E5"
    _accent_light = "rgba(79,70,229,.08)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {{
  --bg:{_bg}; --bg2:{_bg2}; --card:{_card}; --border:{_border};
  --text:{_text}; --text-muted:{_text_muted}; --primary:{_primary}; --shadow:{_shadow};
  --accent:{_accent};
}}

html, body {{ background-color:{_bg} !important; color:{_text} !important; }}
.stApp, .stApp > * {{ background:{_bg} !important; }}
.main .block-container {{ background:{_bg} !important; padding-top:1.5rem !important; }}

*, *::before, *::after {{ font-family:'Plus Jakarta Sans',sans-serif !important; }}

.stApp p, .stApp span, .stApp div, .stApp label, .stApp small, .stApp strong,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4,.stMarkdown h5,.stMarkdown h6,
[data-testid="stText"], [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  color:{_text} !important;
}}

/* 메트릭 */
div[data-testid="metric-container"] {{
  background:{_metric_bg} !important; border-radius:14px !important;
  padding:18px !important; border:1px solid {_border} !important;
  box-shadow:{_shadow} !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] span {{
  color:{_text_muted} !important; font-size:.8rem !important; font-weight:600 !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] * {{
  color:{_text} !important; font-size:1.6rem !important; font-weight:800 !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] * {{
  color:{_text_muted} !important; background:transparent !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg {{ display:none !important; }}

/* 입력창 */
.stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label,
.stRadio label, .stCheckbox label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{
  color:{_text} !important; font-weight:600 !important;
}}
.stTextInput>div>div>input, .stTextArea>div>div>textarea {{
  border-radius:12px !important; border:1.5px solid {_border} !important;
  background:{_input_bg} !important; color:{_text} !important; font-size:.9rem !important;
}}
.stTextInput>div>div>input::placeholder, .stTextArea>div>div>textarea::placeholder {{
  color:{_text_muted} !important;
}}

/* 셀렉트박스 */
.stSelectbox>div>div, .stSelectbox [data-baseweb="select"] > div {{
  background:{_input_bg} !important; border:1.5px solid {_border} !important;
  border-radius:12px !important; color:{_text} !important;
}}
.stSelectbox [data-baseweb="select"] span {{ color:{_text} !important; }}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] li {{
  background:{_card} !important; color:{_text} !important;
}}

/* 슬라이더 */
.stSlider [data-testid="stTickBar"] span, .stSlider p {{ color:{_text} !important; }}
.stRadio > div > label > div > p {{ color:{_text} !important; }}

/* 버튼 */
.stButton>button {{
  background:{_btn_gr} !important; color:white !important;
  border:none !important; border-radius:12px !important; font-weight:700 !important;
  padding:12px 28px !important; transition:all .2s !important;
}}
.stButton>button:hover {{ transform:translateY(-2px) !important; opacity:.92 !important; }}
.stButton>button p {{ color:white !important; }}

/* 프로그레스 */
.stProgress>div>div>div {{ background:{_progress} !important; border-radius:8px !important; }}

/* 사이드바 */
[data-testid="stSidebar"] {{ background:{_sidebar_gr} !important; }}
[data-testid="stSidebar"] *:not(.stButton>button) {{ color:white !important; }}
[data-testid="stSidebar"] .stTextInput>div>div>input {{
  background:rgba(255,255,255,.12) !important;
  border:1px solid rgba(255,255,255,.25) !important; color:white !important;
}}
[data-testid="stSidebar"] .stSelectbox>div>div {{
  background:rgba(255,255,255,.1) !important;
  border:1px solid rgba(255,255,255,.2) !important; color:white !important;
}}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {{ color:white !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
  color:white !important;
}}

/* expander */
.streamlit-expander {{
  background:{_card} !important; border:1px solid {_border} !important; border-radius:12px !important;
}}
.streamlit-expander summary {{ color:{_text} !important; }}
.streamlit-expander summary p {{ color:{_text} !important; }}

/* 데이터프레임 */
[data-testid="stDataFrame"] {{ background:{_card} !important; }}
[data-testid="stDataFrame"] * {{ color:{_text} !important; background:{_card} !important; }}

/* 커스텀 컴포넌트 */
.main-header {{
  background:{_header_gr}; border-radius:20px; padding:32px 40px;
  margin-bottom:24px; box-shadow:{_shadow};
}}
.main-header h1 {{ color:white !important; font-size:1.9rem !important; font-weight:800 !important; margin:0 !important; }}
.main-header p {{ color:rgba(255,255,255,.82) !important; font-size:.95rem !important; margin:8px 0 0 !important; }}

.section-card {{
  background:{_card}; border-radius:16px; padding:24px 28px;
  border:1px solid {_border}; box-shadow:{_shadow}; margin-bottom:20px;
}}
.section-title {{
  font-size:1rem; font-weight:800; color:{_text}; margin:0 0 16px;
  display:flex; align-items:center; gap:8px;
}}

.q-input-row {{
  display:flex; align-items:center; gap:10px; margin:6px 0;
  background:{_bg2}; border-radius:12px; padding:10px 14px;
  border:1px solid {_border};
}}
.q-num {{
  min-width:26px; height:26px; border-radius:8px;
  background:{_btn_gr}; color:white; font-weight:800;
  font-size:.8rem; display:flex; align-items:center; justify-content:center;
  flex-shrink:0;
}}

.result-question-card {{
  background:{_card}; border-radius:14px; padding:18px 22px;
  border:1px solid {_border}; margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,.06);
}}
.rate-badge-high {{
  display:inline-block; background:linear-gradient(135deg,#10B981,#059669);
  color:white !important; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700;
}}
.rate-badge-mid {{
  display:inline-block; background:linear-gradient(135deg,#F59E0B,#D97706);
  color:white !important; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700;
}}
.rate-badge-low {{
  display:inline-block; background:linear-gradient(135deg,#EF4444,#DC2626);
  color:white !important; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700;
}}

.strategy-item {{
  background:{_card}; border-radius:12px; padding:14px 16px;
  margin:8px 0; border:1px solid {_border};
}}
.strategy-item span {{
  font-size:.88rem; color:{_text} !important;
  word-break:keep-all; overflow-wrap:break-word;
  white-space:normal; display:block; line-height:1.6;
}}

.blue-ocean-item {{
  background:{_bg2}; border-radius:12px; padding:12px 16px;
  margin:8px 0; border:1px solid {_border};
  display:flex; align-items:center; gap:10px;
}}
.blue-ocean-item .label {{
  background:{_btn_gr}; color:white !important;
  padding:3px 10px; border-radius:20px; font-size:.75rem; font-weight:700;
}}
.blue-ocean-item .kw {{ font-size:.88rem; color:{_text} !important; font-weight:600; }}

.geo-guide-item {{
  background:{_card}; border-radius:14px; padding:18px 20px;
  margin:10px 0; border:1px solid {_border}; box-shadow:0 2px 12px rgba(0,0,0,.06);
}}
.geo-guide-item .geo-text {{
  margin:0; font-size:.88rem; color:{_text} !important; line-height:1.8; word-break:keep-all;
}}

.comp-row {{
  display:flex; align-items:center; justify-content:space-between;
  padding:11px 16px; border-radius:10px; margin:5px 0;
  border:1.5px solid {_border};
}}

.sidebar-logo {{
  text-align:center; padding:20px 0 24px;
  border-bottom:1px solid rgba(255,255,255,.15); margin-bottom:20px;
}}
.sidebar-logo .logo-icon {{ font-size:2.5rem; display:block; margin-bottom:8px; }}
.sidebar-logo h2 {{ color:white !important; font-size:1.1rem !important; font-weight:800 !important; margin:0 !important; }}
.sidebar-logo p {{ color:rgba(255,255,255,.6) !important; font-size:.75rem !important; margin:4px 0 0 !important; }}

.app-footer {{
  text-align:center; padding:24px; color:{_text_muted};
  font-size:.8rem; border-top:1px solid {_border}; margin-top:32px;
}}

.info-box {{
  background:{_accent_light}; border:1px solid {_accent};
  border-radius:12px; padding:14px 18px; margin:12px 0;
  font-size:.87rem; color:{_text} !important; line-height:1.7;
}}

.stToggle label p {{ color:{_text} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 차트 렌더링
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 브랜드 집계 렌더링
# ─────────────────────────────────────────────

def render_brand_ranking(results: list, brand_name: str = "", _card=_card, _bg2=_bg2,
                          _border=_border, _text=_text, _text_muted=_text_muted,
                          _btn_gr=_btn_gr, _accent=_accent, _accent_light=_accent_light):
    """
    모든 시뮬레이션 결과에서 competitor_mentions 를 합산하여
    AI 응답 내 브랜드 언급 순위를 표시.

    탐지 기준:
      - URL 포함 (url): avahair-avenue.com 형태
      - 텍스트 언급 (mention): "에이바헤어", "IKEA" 등 브랜드명 자체
    두 가지 모두 집계에 포함.
    """
    st.markdown("### 🏢 AI 응답 내 브랜드 언급 순위")
    st.caption("AI가 답변할 때 언급한 브랜드를 집계한 경쟁 현황입니다.")

    # 모든 결과에서 competitor_mentions 합산
    merged: dict = {}
    for r in results:
        cm = r.get("competitor_mentions") if isinstance(r, dict) else getattr(r, "competitor_mentions", {})
        if not cm:
            continue
        for brand, data in cm.items():
            if brand not in merged:
                merged[brand] = {"mentions": 0, "urls": []}
            merged[brand]["mentions"] += data.get("mentions", 0)
            for u in data.get("urls", []):
                if u not in merged[brand]["urls"]:
                    merged[brand]["urls"].append(u)

    if not merged:
        st.info("⚠️ 아직 브랜드 집계 데이터가 없습니다. 시뮬레이션이 완료되면 표시됩니다.")
        return

    # 정렬: mentions 내림차순
    sorted_brands = sorted(merged.items(), key=lambda x: x[1]["mentions"], reverse=True)

    # 내 브랜드는 강조, 나머지는 일반 표시
    my_brand_lower = brand_name.lower().replace(" ", "")

    st.markdown(f"""
<div style="background:{_card};border-radius:16px;padding:20px 24px;
border:1px solid {_border};margin-bottom:16px;">""", unsafe_allow_html=True)

    for rank, (b_name, b_data) in enumerate(sorted_brands[:10], 1):
        mentions   = b_data.get("mentions", 0)
        urls       = b_data.get("urls", [])
        b_lower    = b_name.lower().replace(" ", "")
        is_mine    = my_brand_lower and (my_brand_lower in b_lower or b_lower in my_brand_lower)

        bar_pct    = min(100, round(mentions / max(sorted_brands[0][1]["mentions"], 1) * 100))

        rank_color = "#10B981" if rank == 1 else ("#F59E0B" if rank <= 3 else "#6B7280")
        row_bg     = _accent_light if is_mine else "transparent"
        mine_badge = (' <span style="background:#10B981;color:white;padding:1px 7px;'
                      'border-radius:10px;font-size:.7rem;font-weight:700;">내 브랜드</span>'
                      if is_mine else "")
        url_html   = ""
        if urls:
            url_html = f'<span style="color:{_text_muted};font-size:.72rem;margin-left:6px;">🔗 {urls[0][:40]}</span>'

        st.markdown(f"""
<div style="padding:10px 14px;border-radius:10px;margin:5px 0;background:{row_bg};
border:1px solid {'#10B981' if is_mine else _border};">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <div style="min-width:26px;height:26px;border-radius:7px;background:{rank_color};
    color:white;font-weight:800;font-size:.8rem;display:flex;align-items:center;
    justify-content:center;flex-shrink:0;">{rank}</div>
    <span style="font-weight:{'700' if is_mine else '500'};color:{_text};font-size:.9rem;">
      {b_name}{mine_badge}
    </span>
    <span style="margin-left:auto;font-weight:700;color:{_text};font-size:.88rem;">{mentions}회</span>
  </div>
  <div style="height:6px;background:{_bg2};border-radius:3px;overflow:hidden;">
    <div style="height:100%;width:{bar_pct}%;background:{'linear-gradient(90deg,#10B981,#059669)' if is_mine else 'linear-gradient(90deg,#6366F1,#4F46E5)'};
    border-radius:3px;transition:width .4s;"></div>
  </div>
  {url_html}
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"총 {len(merged)}개 브랜드 탐지 | URL + 텍스트 언급 모두 집계")


def render_bar_chart(results: list, questions: list[str], brand_name: str = ""):
    if not results:
        return

    short_q = [q[:22] + "…" if len(q) > 24 else q for q in questions]

    def _get(r, k):
        if isinstance(r, dict): return r.get(k)
        return getattr(r, k, None)

    gpt_rates   = [_get(r, "gpt_rate")    for r in results]
    gemini_rates = [_get(r, "gemini_rate") for r in results]

    has_gpt    = any(v is not None for v in gpt_rates)
    has_gemini = any(v is not None for v in gemini_rates)

    fig = go.Figure()
    if has_gpt:
        fig.add_trace(go.Bar(
            name="GPT", x=short_q,
            y=[v or 0 for v in gpt_rates],
            marker=dict(color="#111111", line=dict(color="#000", width=1)),
            text=[f"{v:.1f}%" if v is not None else "" for v in gpt_rates],
            textposition="outside",
        ))
    if has_gemini:
        fig.add_trace(go.Bar(
            name="Gemini", x=short_q,
            y=[v or 0 for v in gemini_rates],
            marker=dict(color="#6366F1"),
            text=[f"{v:.1f}%" if v is not None else "" for v in gemini_rates],
            textposition="outside",
        ))

    all_vals = [v for v in gpt_rates + gemini_rates if v is not None]
    title = f"'{brand_name}' AI 인용 점유율" if brand_name else "AI 인용 점유율"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=_text, family="Plus Jakarta Sans"), x=0),
        barmode="group", bargap=0.25,
        plot_bgcolor=_bg2, paper_bgcolor=_card,
        font=dict(family="Plus Jakarta Sans", color=_text),
        xaxis=dict(tickfont=dict(size=10), gridcolor=_border, color=_text),
        yaxis=dict(title="인용 점유율 (%)", ticksuffix="%", gridcolor=_border,
                   range=[0, max(max(all_vals, default=0) + 20, 25)], color=_text),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=_text)),
        margin=dict(t=60, b=50, l=50, r=20), height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# 전략 결과 렌더링
# ─────────────────────────────────────────────

def render_strategy(strategy: dict, target_url: str):
    domain = extract_domain(target_url)

    st.markdown("### 🏆 AI 인용 경쟁 현황 (TOP 5)")
    pos_colors = {
        "업계1위": "#10B981", "업계 1위": "#10B981",
        "신흥강자": "#F59E0B", "신흥 강자": "#F59E0B",
        "틈새전문": "#6366F1", "틈새 전문": "#6366F1",
    }
    competitors = strategy.get("competitors", [])[:5]

    if not competitors:
        st.info("경쟁사 데이터를 불러오지 못했습니다.")
    else:
        for comp in competitors:
            r      = comp.get("rank", "?")
            d      = comp.get("domain", "")
            b      = comp.get("brand_name", d)
            reason = comp.get("reason", "")
            pos    = comp.get("position", "")
            is_t   = domain.lower() in d.lower()

            bg    = f"linear-gradient(135deg,{_bg2},{_border})" if is_t else _card
            bd    = _text if is_t else _border
            lbl   = " ← 내 사이트" if is_t else ""
            pc    = pos_colors.get(pos, "#888")
            pb    = (f'<span style="background:{pc};color:white;padding:2px 8px;border-radius:20px;'
                     f'font-size:.72rem;font-weight:700;margin-left:8px;">{pos}</span>' if pos else "")

            st.markdown(f"""
<div class="comp-row" style="background:{bg};border-color:{bd};">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="min-width:28px;height:28px;border-radius:8px;
         background:{'linear-gradient(135deg,#111,#444)' if is_t else '#CCC'};
         color:white;font-weight:700;font-size:.8rem;display:flex;align-items:center;justify-content:center;">{r}</div>
    <span style="font-weight:{'700' if is_t else '500'};color:{_text};font-size:.9rem;">
      {b} <span style="color:{_text_muted};font-size:.78rem;">({d})</span>{lbl}{pb}
    </span>
  </div>
  <span style="color:{_text_muted};font-size:.8rem;max-width:220px;text-align:right;">{reason}</span>
</div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,{_border},transparent);margin:20px 0;'>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔬 인용 실패 원인 진단")
        _diag_icons = ["❌", "⚡", "🔧"]
        for i, d in enumerate(strategy.get("diagnoses", [])):
            col = [_text, "#555", "#888"][i % 3]
            icon = _diag_icons[i % len(_diag_icons)]
            st.markdown(f"""
<div class="strategy-item" style="border-left:4px solid {col};">
  <span>{icon} {d}</span>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("### 🌊 블루오션 키워드")
        for kw in strategy.get("keywords", []):
            st.markdown(f"""
<div class="blue-ocean-item">
  <span class="label">NEW</span>
  <span class="kw">{kw}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("### 📋 GEO 최적화 가이드")
    for i, g in enumerate(strategy.get("geo_guides", [])):
        g_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', g)
        g_html = g_html.replace('\n', '<br>')
        st.markdown(f"""
<div class="geo-guide-item">
  <div style="display:flex;gap:12px;align-items:flex-start;">
    <div style="min-width:30px;height:30px;border-radius:8px;flex-shrink:0;
         background:linear-gradient(135deg,#111,#444);color:white;font-weight:800;
         font-size:.85rem;display:flex;align-items:center;justify-content:center;">{i+1}</div>
    <div class="geo-text">{g_html}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 질문별 상세 결과
# ─────────────────────────────────────────────

def render_question_details(questions: list[str], results: list):
    st.markdown("### 📋 질문별 상세 결과")
    for i, (q, r) in enumerate(zip(questions, results)):
        def _get(k):
            if isinstance(r, dict): return r.get(k)
            return getattr(r, k, None)

        gpt_v   = f"{_get('gpt_rate'):.1f}%"   if _get('gpt_rate')    is not None else "—"
        gem_v   = f"{_get('gemini_rate'):.1f}%" if _get('gemini_rate') is not None else "—"
        avg     = _get('avg_rate') or 0
        gpt_ci  = _get('gpt_ci')   or (None, None)
        gem_ci  = _get('gemini_ci') or (None, None)
        n       = _get('n') or 0

        # 배지 색상
        if avg >= 30:
            badge = f'<span class="rate-badge-high">🔥 {avg:.1f}%</span>'
        elif avg >= 10:
            badge = f'<span class="rate-badge-mid">⚡ {avg:.1f}%</span>'
        else:
            badge = f'<span class="rate-badge-low">📉 {avg:.1f}%</span>'

        # [FIX] expander label — 특수문자/이모지 아이콘 코드 제거
        # :arrow_down: 같은 Streamlit 아이콘 코드가 구버전에서 텍스트로 표시되는 문제
        avg_label = f"{avg:.1f}%" if avg else "0.0%"
        expander_label = f"Q{i+1}. {q[:55]}{'…' if len(q) > 55 else ''} — 평균 {avg_label}"
        with st.expander(expander_label, expanded=(i == 0)):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("GPT 점유율", gpt_v)
            col2.metric("Gemini 점유율", gem_v)
            col3.metric("평균 점유율", f"{avg:.1f}%")
            col4.metric("시뮬레이션", f"{n}회")

            # CI 표시
            ci_parts = []
            if gpt_ci[0] is not None:
                ci_parts.append(f"GPT 95% CI: [{gpt_ci[0]:.1f}%, {gpt_ci[1]:.1f}%]")
            if gem_ci[0] is not None:
                ci_parts.append(f"Gemini 95% CI: [{gem_ci[0]:.1f}%, {gem_ci[1]:.1f}%]")
            if ci_parts:
                st.caption("📊 신뢰구간 (Wilson Score) — " + "  |  ".join(ci_parts))

            # 샘플 응답
            gpt_samples = _get('gpt_samples') or []
            gem_samples = _get('gemini_samples') or []
            if gpt_samples or gem_samples:
                st.markdown("**💬 AI 응답 샘플**")
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    if gpt_samples:
                        st.caption("GPT 샘플")
                        st.info(gpt_samples[0][:300])
                with s_col2:
                    if gem_samples:
                        st.caption("Gemini 샘플")
                        st.info(gem_samples[0][:300])


# ─────────────────────────────────────────────
# API 클라이언트 초기화
# ─────────────────────────────────────────────

def get_clients(openai_key: str, gemini_key: str, gemini_model_name: str):
    client_gpt = None
    client_gemini = None

    if openai_key and openai_key.startswith("sk-"):
        with CaptureError("init_gpt", log_level="error"):
            import openai
            client_gpt = openai.OpenAI(api_key=openai_key)

    if gemini_key and len(gemini_key) > 10:
        with CaptureError("init_gemini", log_level="error"):
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            client_gemini = genai.GenerativeModel(gemini_model_name)

    return client_gpt, client_gemini


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
  <span class="logo-icon">🔍</span>
  <h2>AI Citation Analyzer</h2>
  <p>v3.0 — 사용자 질문 직접 입력</p>
</div>""", unsafe_allow_html=True)

    if st.button("🌙 다크 모드" if not _dark else "☀️ 라이트 모드",
                 key="btn_dark", use_container_width=True):
        st.session_state["dark_mode"] = not _dark
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    openai_key       = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    gemini_key       = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    gpt_model        = st.selectbox("GPT 모델", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    gemini_model_name = st.selectbox("Gemini 모델",
                                     ["models/gemini-2.0-flash", "models/gemini-flash-latest"])

    st.markdown("---")

    sim_count    = st.slider("시뮬레이션 횟수", 10, 100, 30, 10,
                             help="질문당 AI 호출 횟수. 많을수록 정밀하지만 비용이 증가합니다.")
    market_scope = st.radio("경쟁사 범위", ["국내 (대한민국)", "글로벌"], horizontal=True)

    st.markdown("---")

    # 연결 상태
    gpt_ok    = bool(openai_key and openai_key.startswith("sk-"))
    gemini_ok = bool(gemini_key and len(gemini_key) > 10)
    c1, c2 = st.columns(2)
    c1.markdown("🟢 **GPT**"    if gpt_ok    else "⚪ **GPT**")
    c2.markdown("🟢 **Gemini**" if gemini_ok else "🔴 **Gemini**")

    # 비용 추적
    tracker: CostTracker = st.session_state["cost_tracker"]
    cost_summary = tracker.summary()
    if cost_summary["api_calls"] > 0:
        st.markdown(f"""
<div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);
border-radius:8px;padding:10px 12px;margin-top:8px;font-size:.75rem;color:rgba(255,255,255,.85);">
💰 누적 비용: ~${cost_summary['estimated_usd']:.4f}<br>
🔢 API 호출: {cost_summary['api_calls']}회
</div>""", unsafe_allow_html=True)

    # 캐시
    cache_stats = _cache.stats()
    if cache_stats["entries"] > 0:
        st.markdown(f"""
<div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);
border-radius:8px;padding:8px 12px;margin-top:6px;font-size:.73rem;color:rgba(255,255,255,.75);">
⚡ 캐시 히트율: {cache_stats['hit_rate']}% ({cache_stats['entries']}개)
</div>""", unsafe_allow_html=True)

    if st.button("🗑️ 캐시 초기화", key="btn_clear_cache", use_container_width=True):
        _cache._store.clear()
        st.session_state["cache_data"] = {}
        st.session_state["cost_tracker"] = CostTracker()
        st.session_state["analysis_results"] = None
        st.success("초기화 완료")


# ─────────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🔍 AI 검색 점유율 분석 대시보드</h1>
  <p>분석할 질문을 직접 입력하고 GPT & Gemini에서의 인용 점유율을 측정하세요</p>
</div>""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 엔진", "GPT + Gemini", "2개 동시")
col2.metric("시뮬레이션", f"{sim_count}회", "질문당")
col3.metric("API 연결", f"{(1 if gpt_ok else 0)+(1 if gemini_ok else 0)}/2", "활성화")
col4.metric("누적 비용", f"~${tracker.summary()['estimated_usd']:.4f}", "USD 추정")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 1: 기본 설정 입력
# ─────────────────────────────────────────────

st.markdown("""
<div class="section-card">
  <div class="section-title">⚙️ STEP 1 &nbsp; 기본 정보 입력</div>
</div>""", unsafe_allow_html=True)

with st.container():
    col_u, col_b, col_i = st.columns([2, 1, 1])
    with col_u:
        url_input = st.text_input(
            "🌐 사이트 URL",
            placeholder="예) naver.com  또는  https://www.example.com",
            key="url_input",
        )
    with col_b:
        brand_input = st.text_input(
            "🏷️ 브랜드명",
            placeholder="예) 네이버, Coupang",
            key="brand_input",
        )
    with col_i:
        industry_input = st.text_input(
            "🏭 업종 (선택)",
            placeholder="예) 온라인 쇼핑몰, SaaS",
            key="industry_input",
        )

    st.markdown(f"""
<div class="info-box">
  💡 <strong>브랜드명</strong>은 AI 응답에서 해당 브랜드의 인용 여부를 탐지하는 데 사용됩니다.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;<strong>업종</strong>은 경쟁사 분석 및 GEO 가이드의 정확도를 높입니다 (미입력 시 URL로 추론).
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STEP 2: 질문 입력 (1~10개)
# ─────────────────────────────────────────────

st.markdown("""
<div class="section-card" style="margin-top:0;">
  <div class="section-title">✏️ STEP 2 &nbsp; 분석 질문 입력</div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
  💡 <strong>AI가 실제로 답변할 법한 질문</strong>을 입력하세요. 브랜드명을 포함하거나,
  해당 업종 사용자가 검색창에 입력하는 방식의 질문이 가장 효과적입니다.<br>
  &nbsp;&nbsp;&nbsp;&nbsp;예시: <em>"쿠팡과 네이버쇼핑의 배송 속도 비교는?"</em>
  &nbsp;/&nbsp; <em>"SaaS CRM 도입 비용 비교"</em>
</div>""", unsafe_allow_html=True)

# 질문 개수 컨트롤
n_q = st.session_state["n_questions"]
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
with ctrl_col1:
    if st.button("➕ 질문 추가", key="btn_add_q", use_container_width=True):
        if n_q < 10:
            n_q += 1
            qs = st.session_state["questions_list"]
            if len(qs) < n_q:
                qs.append("")
            st.session_state["n_questions"] = n_q
            st.session_state["questions_list"] = qs
            st.rerun()
with ctrl_col2:
    if st.button("➖ 질문 삭제", key="btn_rm_q", use_container_width=True):
        if n_q > 1:
            n_q -= 1
            st.session_state["n_questions"] = n_q
            st.session_state["questions_list"] = st.session_state["questions_list"][:n_q]
            st.rerun()
with ctrl_col3:
    st.caption(f"현재 {st.session_state['n_questions']}개 질문 (최대 10개)")

# 질문 입력 폼
questions_entered = []
qs_state = st.session_state["questions_list"]

for i in range(st.session_state["n_questions"]):
    q_val = qs_state[i] if i < len(qs_state) else ""
    entered = st.text_input(
        f"질문 {i+1}",
        value=q_val,
        placeholder=f"Q{i+1}: 분석할 질문을 입력하세요...",
        key=f"q_{i}",
        label_visibility="collapsed",
    )
    questions_entered.append(entered)
    # 상태 동기화
    if i < len(qs_state):
        qs_state[i] = entered
    else:
        qs_state.append(entered)

st.session_state["questions_list"] = qs_state


# ─────────────────────────────────────────────
# STEP 3: 실행 버튼
# ─────────────────────────────────────────────

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

col_run, col_demo, col_reset = st.columns([3, 1, 1])
with col_run:
    run_clicked = st.button(
        "🚀 분석 시작",
        key="btn_run",
        use_container_width=True,
        type="primary",
    )
with col_demo:
    demo_clicked = st.button("🎬 데모", key="btn_demo", use_container_width=True)
with col_reset:
    reset_clicked = st.button("🔄 초기화", key="btn_reset", use_container_width=True)

if reset_clicked:
    st.session_state["analysis_results"] = None
    st.session_state["questions_list"] = [""] * 5
    st.session_state["n_questions"] = 5
    st.rerun()


# ─────────────────────────────────────────────
# 데모 데이터
# ─────────────────────────────────────────────

_DEMO_DATA = {
    "url": "naver.com",
    "brand": "네이버",
    "industry": "포털/IT 플랫폼",
    "questions": [
        "네이버와 구글 검색 결과의 실제 차이는?",
        "네이버 쇼핑 vs 쿠팡 가격 비교 신뢰도는?",
        "네이버 블로그가 SEO에 미치는 영향은?",
        "네이버 클라우드 vs AWS 국내 기업 선택 기준은?",
        "네이버 페이 보안성과 카카오페이 비교는?",
    ],
    "results": [
        {"gpt_rate": 58.0, "gemini_rate": 62.0, "avg_rate": 60.0,
         "gpt_hits": 29, "gemini_hits": 31, "n": 50,
         "gpt_ci": (44.2, 71.0), "gemini_ci": (48.0, 74.5),
         "gpt_samples": ["네이버는 한국 시장에서 구글과 달리 지역화된 검색 알고리즘을 사용합니다..."],
         "gemini_samples": ["검색 결과의 경우 네이버는 국내 콘텐츠에 강점이 있으며..."]},
        {"gpt_rate": 44.0, "gemini_rate": 51.0, "avg_rate": 47.5,
         "gpt_hits": 22, "gemini_hits": 26, "n": 50,
         "gpt_ci": (31.5, 57.3), "gemini_ci": (38.0, 64.1),
         "gpt_samples": ["네이버 쇼핑은 국내 판매자 중심으로..."],
         "gemini_samples": []},
        {"gpt_rate": 22.0, "gemini_rate": 18.0, "avg_rate": 20.0,
         "gpt_hits": 11, "gemini_hits": 9, "n": 50,
         "gpt_ci": (12.5, 35.1), "gemini_ci": (9.6, 30.5),
         "gpt_samples": [], "gemini_samples": []},
        {"gpt_rate": 31.0, "gemini_rate": 27.0, "avg_rate": 29.0,
         "gpt_hits": 16, "gemini_hits": 14, "n": 50,
         "gpt_ci": (20.0, 44.5), "gemini_ci": (16.8, 40.1),
         "gpt_samples": [], "gemini_samples": []},
        {"gpt_rate": 39.0, "gemini_rate": 43.0, "avg_rate": 41.0,
         "gpt_hits": 20, "gemini_hits": 22, "n": 50,
         "gpt_ci": (27.1, 52.5), "gemini_ci": (30.2, 56.8),
         "gpt_samples": [], "gemini_samples": []},
    ],
    "strategy": {
        "competitors": [
            {"rank": 1, "domain": "google.com",   "brand_name": "Google",   "reason": "글로벌 검색 점유율 1위",   "position": "업계1위"},
            {"rank": 2, "domain": "daum.net",     "brand_name": "Daum",     "reason": "국내 2위 포털",            "position": "신흥강자"},
            {"rank": 3, "domain": "kakao.com",    "brand_name": "Kakao",    "reason": "메신저 기반 플랫폼",       "position": "신흥강자"},
            {"rank": 4, "domain": "coupang.com",  "brand_name": "Coupang",  "reason": "쇼핑 경쟁사",              "position": "틈새전문"},
            {"rank": 5, "domain": "youtube.com",  "brand_name": "YouTube",  "reason": "영상 검색 대체재",         "position": "틈새전문"},
        ],
        "diagnoses": [
            "구조화 데이터 마크업 부재로 AI가 정보 출처를 명확히 인식하기 어려움",
            "FAQ 섹션 부족으로 Q&A 기반 인용 기회 손실 중",
            "경쟁 키워드 대비 전문성 콘텐츠의 업데이트 빈도 낮음",
        ],
        "keywords": [
            "AI 검색 엔진 최적화 2025",
            "GEO 마케팅 전략 한국",
            "챗봇 답변 브랜드 노출 방법",
            "LLM 친화적 콘텐츠 작성법",
            "AI 인용 점유율 측정 도구",
        ],
        "geo_guides": [
            "**FAQ 블록 추가**: 홈페이지에 서비스 핵심 기능을 Q&A 형식으로 구조화하세요.\n"
            "AI는 명확한 질문-답변 패턴을 가진 페이지를 권위 있는 출처로 인식합니다.",

            "**구조화 데이터 적용**: JSON-LD로 Organization, FAQPage, Product 스키마를 삽입하세요.\n"
            "검색 엔진과 AI 모두 구조화된 마크업이 있는 페이지를 높이 평가합니다.",

            "**핵심 가치 제안 최상단 배치**: '~는 ~입니다'처럼 명확한 정의 문장으로 시작하세요.\n"
            "AI가 브랜드를 한 문장으로 설명할 수 있어야 인용 확률이 높아집니다.",
        ],
    },
}


# ─────────────────────────────────────────────
# 데모 실행
# ─────────────────────────────────────────────

if demo_clicked:
    # 데모 데이터로 입력 필드 채우기
    st.session_state["questions_list"] = _DEMO_DATA["questions"] + [""] * 5
    st.session_state["n_questions"] = len(_DEMO_DATA["questions"])
    st.session_state["analysis_results"] = {
        "url": normalize_url(_DEMO_DATA["url"]),
        "brand": _DEMO_DATA["brand"],
        "industry": _DEMO_DATA["industry"],
        "questions": _DEMO_DATA["questions"],
        "results": _DEMO_DATA["results"],
        "strategy": _DEMO_DATA["strategy"],
        "is_demo": True,
        "elapsed": 0,
    }
    st.rerun()


# ─────────────────────────────────────────────
# 실제 분석 실행
# ─────────────────────────────────────────────

if run_clicked:
    # ── 입력 검증 ──
    target_url  = url_input.strip()
    brand_name  = brand_input.strip()
    industry    = industry_input.strip()
    questions   = [q.strip() for q in questions_entered if q.strip()]

    errors = []
    if not target_url:
        errors.append("URL을 입력해 주세요.")
    if not brand_name:
        errors.append("브랜드명을 입력해 주세요.")
    if len(questions) < 1:
        errors.append("질문을 최소 1개 이상 입력해 주세요.")
    if not gpt_ok and not gemini_ok:
        errors.append("OpenAI 또는 Gemini API 키를 입력해 주세요.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    target_url = normalize_url(target_url)
    domain     = extract_domain(target_url)

    client_gpt, client_gemini = get_clients(openai_key, gemini_key, gemini_model_name)

    # BusinessInfo 생성 (질문 생성 생략 — 사용자 입력 사용)
    biz_info = BusinessInfo(
        brand_name=brand_name,
        industry=industry if industry else f"{domain} 서비스",
        industry_category="기타",
        core_product=industry if industry else "서비스",
        target_audience="잠재 고객",
        confidence="high" if industry else "medium",
        crawl_tier=0,
    )

    # ── 진행 상태 UI ──
    prog_container = st.container()
    prog_bar   = prog_container.progress(0)
    prog_text  = prog_container.empty()

    t_start    = time.time()
    n_engines  = (1 if client_gpt else 0) + (1 if client_gemini else 0)

    prog_text.markdown(f"🚀 **시뮬레이션 시작** — {len(questions)}개 질문 × {sim_count}회 × {n_engines}엔진")
    prog_bar.progress(0.05)

    # ── Step 1: 시뮬레이션 단독 실행 ──
    # 전략 분석과 동시 실행하면 Gemini Rate Limit 초과 → 전략 분석 전체 실패
    # 순차 실행으로 변경
    with CaptureError("sim_run", log_level="warning") as ctx:
        sim_results = run_all_simulations(
            client_gpt, client_gemini, questions, target_url,
            model_gpt=gpt_model, n=sim_count,
            biz_info=biz_info.to_dict(),
            tracker=tracker,
            use_cache=True,
        )
    if not ctx.ok:
        st.error(f"시뮬레이션 오류: {ctx.error}")
        st.stop()

    elapsed_sim = int(time.time() - t_start)
    prog_bar.progress(0.75)
    prog_text.markdown(f"✅ 시뮬레이션 완료 ({elapsed_sim}초) | 🔍 전략 분석 시작…")

    # ── Step 2: 전략 분석 (시뮬레이션 완료 후 순차 실행) ──
    strategy = {}
    if questions:
        with CaptureError("strategy_run", log_level="warning") as ctx:
            strategy = run_strategy_analysis(
                client_gpt, client_gemini,
                question=questions[0],
                target_url=target_url,
                model_gpt=gpt_model,
                biz_info=biz_info,
                market_scope=market_scope,
                use_cache=True,
            )
        if not ctx.ok:
            strategy = {}
            st.warning(f"⚠️ 전략 분석 실패 (재시도하세요): {ctx.error}")

    elapsed_total = int(time.time() - t_start)
    prog_bar.progress(1.0)
    prog_text.success(
        f"✅ 분석 완료! ({elapsed_total}초) | "
        f"추정 비용: ~${tracker.summary()['estimated_usd']:.4f}"
    )

    save_cache()
    st.session_state["cost_tracker"] = tracker
    st.session_state["analysis_results"] = {
        "url": target_url,
        "brand": brand_name,
        "industry": industry,
        "questions": questions,
        "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in sim_results],
        "strategy": strategy,
        "is_demo": False,
        "elapsed": elapsed_total,
    }
    st.rerun()


# ─────────────────────────────────────────────
# 결과 표시
# ─────────────────────────────────────────────

res = st.session_state.get("analysis_results")

if res:
    is_demo   = res.get("is_demo", False)
    questions = res["questions"]
    results   = res["results"]
    strategy  = res.get("strategy", {})
    brand     = res["brand"]
    target_url = res["url"]
    industry  = res.get("industry", "")

    st.markdown("---")

    # 결과 헤더
    demo_badge = '<span style="background:#F59E0B;color:white;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:700;margin-left:8px;">DEMO</span>' if is_demo else ""
    st.markdown(f"""
<div style="background:{_card};border-radius:16px;padding:20px 28px;border:1px solid {_border};
margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;">
  <div>
    <span style="font-size:1.1rem;font-weight:800;color:{_text};">
      📊 분석 결과 — {brand}{demo_badge}
    </span><br>
    <span style="font-size:.83rem;color:{_text_muted};">
      {extract_domain(target_url)}
      {(' · ' + industry) if industry else ''}
      · {len(questions)}개 질문
      {(' · ' + str(res.get('elapsed', 0)) + '초') if not is_demo else ' · 샘플 데이터'}
    </span>
  </div>
  <div style="font-size:.8rem;color:{_text_muted};">
    캐시 히트율 {_cache.stats()['hit_rate']}%
  </div>
</div>""", unsafe_allow_html=True)

    # ── 0. 0% 인용률 사용자 안내 ──
    all_rates = []
    for r in results:
        v = r.get('gemini_rate') if isinstance(r, dict) else getattr(r, 'gemini_rate', None)
        if v is not None: all_rates.append(v)
        v2 = r.get('gpt_rate') if isinstance(r, dict) else getattr(r, 'gpt_rate', None)
        if v2 is not None: all_rates.append(v2)

    avg_all = sum(all_rates) / len(all_rates) if all_rates else 0
    if avg_all == 0 and not is_demo:
        st.warning(
            "📌 **인용 점유율이 0%입니다.** 가능한 원인:\n"
            "① 질문에 **브랜드명**이 포함되지 않아 AI 응답에서 탐지가 안 됨\n"
            "② 해당 브랜드가 아직 AI 학습 데이터에 충분히 없음\n"
            "③ Gemini API만 사용 중이고 GPT 키가 없어 교차 검증이 안 됨\n\n"
            "**해결책**: 질문에 브랜드명을 자연스럽게 포함시켜 보세요.\n"
            "예) '네이버 쇼핑과 쿠팡 배송 속도 비교?' → 브랜드명 포함"
        )

    # ── 1. 인용 점유율 차트 ──
    render_bar_chart(results, questions, brand)

    # ── 2. 브랜드 언급 순위 (시뮬레이션에서 집계된 경쟁사 노출 현황) ──
    st.markdown("---")
    render_brand_ranking(results, brand)

    # ── 3. 질문별 상세 ──
    render_question_details(questions, results)

    # ── 3. 전략 분석 ──
    if strategy:
        st.markdown("---")
        render_strategy(strategy, target_url)
    else:
        st.info("전략 분석 데이터가 없습니다. 다시 분석을 실행해 주세요.")

    # ── 푸터 ──
    st.markdown(f"""
<div class="app-footer">
  AI Citation Analyzer v3.0 &nbsp;·&nbsp; GPT + Gemini 기반 인용 점유율 측정
</div>""", unsafe_allow_html=True)

else:
    # 빈 상태 안내
    st.markdown(f"""
<div style="text-align:center;padding:60px 20px;color:{_text_muted};">
  <div style="font-size:3rem;margin-bottom:16px;">🔍</div>
  <div style="font-size:1.1rem;font-weight:700;color:{_text};margin-bottom:8px;">
    분석 준비 완료
  </div>
  <div style="font-size:.9rem;line-height:1.7;">
    위에서 <strong>URL · 브랜드명 · 질문</strong>을 입력한 뒤<br>
    <strong>🚀 분석 시작</strong> 버튼을 누르세요.<br><br>
    먼저 <strong>🎬 데모</strong>로 결과 형식을 확인할 수 있습니다.
  </div>
</div>""", unsafe_allow_html=True)
