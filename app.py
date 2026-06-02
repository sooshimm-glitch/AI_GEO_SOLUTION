"""
AI Citation Analyzer v4.0 — 리디자인
색상: 오렌지/핑크 계열 (참고 이미지 기반)
차트: 질문별 탭 + 도넛 차트 (GPT/Gemini 각각)
"""

import streamlit as st
import datetime
import re
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import urlparse

import sys, os
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
_CLOUD_PATH = "/mount/src/ai_geo_solution"
if os.path.isdir(_CLOUD_PATH) and _CLOUD_PATH not in sys.path:
    sys.path.insert(0, _CLOUD_PATH)

try:
    from core.cache import get_cache
    from core.logger import get_logger, CaptureError
    from core.citation import build_brand_variants
    from core.ai_client import run_all_simulations, CostTracker, SimResult, fetch_competitor_brands
    from core.biz_analysis import run_strategy_analysis, BusinessInfo
except ImportError as _e:
    st.set_page_config(page_title="Import Error", page_icon="❌")
    st.error(f"**core 모듈 import 실패**: {_e}")
    st.stop()

logger = get_logger("app")

st.set_page_config(
    page_title="AI Citation Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state 초기화 ──────────────────────────────────────────────────────
for k, v in {
    "dark_mode": False,
    "cache_data": {},
    "cost_tracker": CostTracker(),
    "history": [],
    "run_demo": False,
    "q_count": 1,
    "questions_list": [""],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

_cache = get_cache()
if st.session_state["cache_data"]:
    try:
        _r = type(_cache).from_serializable(st.session_state["cache_data"])
        _cache._store.update(_r._store)
    except Exception:
        st.session_state["cache_data"] = {}

_dark = st.session_state["dark_mode"]

# ── 색상 테마 (오렌지/핑크 계열) ─────────────────────────────────────────────
if _dark:
    _bg       = "#0D0D0D"
    _bg2      = "#181818"
    _card     = "#1C1C1C"
    _border   = "#2E2E2E"
    _text     = "#F2F2F2"
    _text_muted = "#888888"
    _shadow   = "0 4px 20px rgba(0,0,0,.6)"
    _sidebar_bg = "#FFFFFF"
    _accent   = "#FF6B35"   # 오렌지
    _accent2  = "#FF4D8B"   # 핑크
    _accent_gr = "linear-gradient(135deg,#FF6B35,#FF4D8B)"
    _running_bg = "rgba(255,107,53,.15)"
    _running_border = "#FF6B35"
    _plot_bg  = "#1C1C1C"
    _plot_paper = "#1C1C1C"
    _plot_font = "#F2F2F2"
    _plot_grid = "#2E2E2E"
else:
    _bg       = "#FFF8F5"
    _bg2      = "#FFF0E8"
    _card     = "#FFFFFF"
    _border   = "#FFD9C7"
    _text     = "#1A1A1A"
    _text_muted = "#888888"
    _shadow   = "0 4px 20px rgba(255,107,53,.1)"
    _sidebar_bg = "#FFFFFF"
    _accent   = "#FF6B35"
    _accent2  = "#FF4D8B"
    _accent_gr = "linear-gradient(135deg,#FF6B35,#FF4D8B)"
    _running_bg = "rgba(255,107,53,.08)"
    _running_border = "#FF6B35"
    _plot_bg  = "#FFFFFF"
    _plot_paper = "#FFFFFF"
    _plot_font = "#1A1A1A"
    _plot_grid = "#FFE8DC"

# ── 전역 CSS ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html,body{{background:{_bg}!important;color:{_text}!important}}
.stApp,.stApp>*{{background:{_bg}!important}}
.main .block-container{{background:{_bg}!important;padding-top:1.5rem!important;max-width:1100px!important}}
*,*::before,*::after{{font-family:'Pretendard','Plus Jakarta Sans',sans-serif!important}}

/* 텍스트 전체 */
.stApp p,.stApp span,.stApp div,.stApp label,.stApp li,
.stMarkdown,.stMarkdown p,.stMarkdown li,
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p{{
    color:{_text}!important;-webkit-text-fill-color:{_text}!important}}

/* 메트릭 */
div[data-testid="metric-container"]{{
    background:{_card}!important;border-radius:16px!important;
    padding:16px 20px!important;border:1.5px solid {_border}!important;
    box-shadow:{_shadow}!important}}
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] *{{
    color:{_text_muted}!important;font-size:.78rem!important;font-weight:600!important}}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] *{{
    color:{_text}!important;font-size:1.5rem!important;font-weight:800!important}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg{{display:none!important}}

/* 인풋 */
.stTextInput label,.stTextArea label,.stSelectbox label,.stSlider label,
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p{{
    color:{_text}!important;font-weight:600!important;font-size:.82rem!important}}
.stTextInput>div>div>input,.stTextArea>div>div>textarea{{
    border-radius:12px!important;border:1.5px solid {_border}!important;
    background:{_bg2}!important;color:{_text}!important;font-size:.9rem!important}}
.stTextInput>div>div>input:focus,.stTextArea>div>div>textarea:focus{{
    border-color:{_accent}!important;box-shadow:0 0 0 3px rgba(255,107,53,.15)!important}}
.stSelectbox>div>div{{
    background:{_bg2}!important;border:1.5px solid {_border}!important;
    border-radius:12px!important}}
.stSelectbox [data-baseweb="select"] span{{color:{_text}!important}}
[data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"] li{{
    background:{_card}!important;color:{_text}!important}}

/* 버튼 — 기본은 오렌지 그라디언트 */
.stButton>button{{
    background:{_accent_gr}!important;color:white!important;
    border:none!important;border-radius:12px!important;
    font-weight:700!important;padding:10px 24px!important;
    transition:all .2s!important;box-shadow:0 4px 12px rgba(255,107,53,.3)!important}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(255,107,53,.4)!important}}
.stButton>button p{{color:white!important}}

/* 탭 */
.stTabs [data-baseweb="tab-list"]{{
    background:{_bg2}!important;border-radius:14px!important;
    padding:5px!important;border:1.5px solid {_border}!important;gap:4px!important}}
.stTabs [data-baseweb="tab"]{{
    border-radius:10px!important;font-weight:600!important;font-size:.88rem!important;
    color:{_text_muted}!important;padding:9px 20px!important;background:transparent!important}}
.stTabs [aria-selected="true"]{{
    background:{_accent_gr}!important;color:white!important;
    box-shadow:0 4px 12px rgba(255,107,53,.3)!important}}
.stTabs [data-baseweb="tab"] p,.stTabs [data-baseweb="tab"] span{{color:inherit!important}}
/* 탭 hover 툴팁 제거 */
.stTabs [data-baseweb="tab"]::after{{content:none!important}}
.stTabs [data-baseweb="tab"][title]{{pointer-events:auto}}
.stTabs [data-baseweb="tab"]:hover::before,.stTabs [data-baseweb="tab"]:hover::after{{display:none!important}}
button[data-baseweb="tab"]{{-webkit-user-select:none}}
button[data-baseweb="tab"][title]{{title:none}}

/* 프로그레스 */
.stProgress>div>div>div{{background:{_accent_gr}!important;border-radius:8px!important}}

/* 사이드바 */
[data-testid="stSidebar"]{{background:{_sidebar_bg}!important;border-right:1.5px solid {_border}!important}}
[data-testid="stSidebar"] *:not(.stButton>button){{color:{_text}!important;-webkit-text-fill-color:{_text}!important}}
[data-testid="stSidebar"] .stTextInput>div>div>input{{
    background:{_bg2}!important;border:1.5px solid {_border}!important;color:{_text}!important}}
[data-testid="stSidebar"] .stSelectbox>div>div{{
    background:{_bg2}!important;border:1.5px solid {_border}!important}}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span{{color:{_text}!important}}
[data-testid="stSidebar"] .stButton>button{{
    background:{_accent_gr}!important;
    box-shadow:0 4px 12px rgba(255,107,53,.3)!important;border:none!important}}

/* expander */
[data-testid="stExpander"]{{background:{_card}!important;border:1.5px solid {_border}!important;border-radius:14px!important}}
[data-testid="stExpander"] summary,[data-testid="stExpander"] p,[data-testid="stExpander"] span{{color:{_text}!important}}
details summary::-webkit-details-marker{{display:none!important}}

/* 커스텀 컴포넌트 */
.aca-card{{
    background:{_card};border:1.5px solid {_border};
    border-radius:18px;padding:20px 24px;box-shadow:{_shadow};margin-bottom:16px}}
.aca-running-banner{{
    background:{_running_bg};border:1.5px solid {_running_border};
    border-radius:14px;padding:16px 20px;margin-bottom:16px}}
.aca-badge-running{{
    display:inline-block;background:{_accent_gr};color:white;
    font-size:.65rem;font-weight:800;padding:2px 8px;border-radius:20px;
    letter-spacing:.05em;margin-left:8px;vertical-align:middle}}
.aca-status-dot{{
    display:inline-block;width:8px;height:8px;border-radius:50%;
    background:#22C55E;margin-right:6px;animation:pulse 1.5s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.aca-workflow-card{{
    background:{_card};border:1.5px solid {_border};border-radius:14px;
    padding:20px;text-align:center;position:relative}}
.aca-workflow-card.active{{border-color:{_accent}!important;box-shadow:0 0 0 3px rgba(255,107,53,.15)}}
.aca-number{{
    font-size:2.2rem;font-weight:800;color:{_text};line-height:1}}
.aca-number-unit{{
    font-size:.85rem;color:{_text_muted};margin-left:4px;font-weight:500}}
.aca-label{{font-size:.75rem;color:{_text_muted};font-weight:600;margin-top:6px}}
.aca-sub{{font-size:.7rem;color:{_text_muted};margin-top:2px}}
.aca-competitor-row{{
    display:flex;align-items:center;justify-content:space-between;
    padding:10px 16px;border-radius:10px;margin:5px 0;
    background:{_bg2};border:1.5px solid {_border}}}
.aca-competitor-row.mine{{
    background:rgba(255,107,53,.08);border-color:{_accent}!important}}
.aca-diag-item{{
    padding:14px 16px;border-radius:12px;margin:8px 0;
    background:{_card};border:1.5px solid {_border}}}
.aca-geo-item{{
    padding:16px 20px;border-radius:14px;margin:10px 0;
    background:{_card};border:1.5px solid {_border};box-shadow:{_shadow}}}
.aca-kw-item{{
    display:flex;align-items:center;gap:10px;
    padding:10px 14px;border-radius:10px;margin:6px 0;
    background:{_bg2};border:1.5px solid {_border}}}
.aca-q-badge{{
    display:inline-flex;align-items:center;justify-content:center;
    min-width:22px;height:22px;border-radius:6px;
    background:{_accent_gr};color:white;font-size:.7rem;font-weight:800;
    margin-right:8px;flex-shrink:0}}
[data-testid="stAlert"] p,[data-testid="stAlert"] span{{color:{_text}!important}}
.stCaptionContainer p{{color:{_text_muted}!important}}
</style>
""", unsafe_allow_html=True)


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
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


# ── 데모 데이터 ───────────────────────────────────────────────────────────────
_DEMO_SCENARIOS = {
    "naver.com": {
        "brand": "네이버", "industry": "포털/검색 서비스",
        "questions": ["국내 최고 포털 사이트 추천해줘?","한국어 지도 서비스 어디가 제일 좋아?","블로그 플랫폼 비교해줘?"],
        "results": [
            {"gpt_rate":58,"gemini_rate":62,"avg_rate":60,"gpt_hits":29,"gemini_hits":31,"n":50,"gpt_ci":[44,71],"gemini_ci":[48,74]},
            {"gpt_rate":44,"gemini_rate":51,"avg_rate":47.5,"gpt_hits":22,"gemini_hits":25,"n":50,"gpt_ci":[31,58],"gemini_ci":[37,65]},
            {"gpt_rate":22,"gemini_rate":18,"avg_rate":20,"gpt_hits":11,"gemini_hits":9,"n":50,"gpt_ci":[12,35],"gemini_ci":[9,31]},
        ],
    },
    "default": {
        "brand": "샘플브랜드", "industry": "디지털 서비스",
        "questions": ["이 분야 서비스 추천해줘?","경쟁 서비스 대비 장점 비교?","초보자도 쓰기 좋은 서비스?"],
        "results": [
            {"gpt_rate":7,"gemini_rate":5,"avg_rate":6,"gpt_hits":3,"gemini_hits":2,"n":50,"gpt_ci":[2,18],"gemini_ci":[1,15]},
            {"gpt_rate":4,"gemini_rate":8,"avg_rate":6,"gpt_hits":2,"gemini_hits":4,"n":50,"gpt_ci":[1,13],"gemini_ci":[2,19]},
            {"gpt_rate":12,"gemini_rate":9,"avg_rate":10.5,"gpt_hits":6,"gemini_hits":4,"n":50,"gpt_ci":[5,24],"gemini_ci":[3,21]},
        ],
    },
}
_DEMO_STRATEGY = {
    "competitors": [
        {"rank":1,"domain":"wikipedia.org","brand_name":"Wikipedia","reason":"중립적 참조","position":"업계1위"},
        {"rank":2,"domain":"namu.wiki","brand_name":"나무위키","reason":"한국어 위키","position":"신흥강자"},
        {"rank":3,"domain":"tistory.com","brand_name":"티스토리","reason":"SEO 블로그","position":"틈새전문"},
    ],
    "diagnoses": ["구조화 데이터 마크업 부재로 AI 맥락 파악 어려움","FAQ 섹션 없어 Q&A 기반 인용 기회 손실","핵심 키워드 밀도 경쟁사 대비 40% 낮음"],
    "keywords": ["AI 인용 최적화 전략 2025","GEO 적용 방법","챗봇 검색 브랜드 노출","LLM 친화적 콘텐츠","AI 답변 출처 선택 조건"],
    "geo_guides": [
        "1. **FAQ 블록 추가**\n홈페이지에 Q&A 형식 서비스 설명 섹션을 추가하세요. AI는 Q&A 구조를 우선 인용합니다.",
        "2. **구조화 데이터 적용**\nJSON-LD로 Organization, FAQPage 스키마를 삽입하세요. AI가 공식 출처로 인식합니다.",
        "3. **핵심 가치 제안 최상단 배치**\n명확한 정의 문장으로 AI가 권위 출처로 인식하게 하세요.",
    ],
}

def get_demo(url: str) -> dict:
    domain = extract_domain(url).lower()
    for k, v in _DEMO_SCENARIOS.items():
        if k != "default" and k in domain:
            return v
    return _DEMO_SCENARIOS["default"]


# ── 워크플로우 카드 (상단 요약) ───────────────────────────────────────────────
def render_workflow_cards(brand_name, n_questions, n_results, n_engines, elapsed=None):
    cols = st.columns(3)
    items = [
        {
            "icon": "📋",
            "number": n_questions,
            "unit": "개",
            "label": "분석 질문",
            "sub": f"총 {n_questions * 50}회 API",
            "active": n_questions > 0,
        },
        {
            "icon": "⏳",
            "number": 0 if not n_results else n_results,
            "unit": "완료",
            "label": "시뮬레이션",
            "sub": f"{elapsed}초 소요" if elapsed else "대기 중",
            "active": n_results > 0,
        },
        {
            "icon": "✅",
            "number": n_engines,
            "unit": "개",
            "label": "분석 엔진",
            "sub": "GPT · Gemini",
            "active": n_engines > 0,
        },
    ]
    for col, item in zip(cols, items):
        active_cls = "active" if item["active"] else ""
        col.markdown(f"""
        <div class="aca-workflow-card {active_cls}">
            <div style="font-size:1.4rem;margin-bottom:6px">{item['icon']}</div>
            <div class="aca-number">{item['number']}<span class="aca-number-unit">{item['unit']}</span></div>
            <div class="aca-label">{item['label']}</div>
            <div class="aca-sub">{item['sub']}</div>
        </div>""", unsafe_allow_html=True)


# ── 도넛 차트 (질문 탭 별) ────────────────────────────────────────────────────
_donut_counter = {"n": 0}

def render_donut_tabs(results: list, questions: list, brand_name: str):
    """질문별 탭 + GPT/Gemini 도넛 차트"""
    if not results:
        return

    def _get(r, k, default=None):
        return r.get(k, default) if isinstance(r, dict) else getattr(r, k, default)

    tab_labels = [f"Q{i+1}" for i in range(len(results))]
    tabs = st.tabs(tab_labels)

    for i, (tab, r) in enumerate(zip(tabs, results)):
        with tab:
            gpt_rate  = _get(r, "gpt_rate")
            gem_rate  = _get(r, "gemini_rate")
            avg_rate  = _get(r, "avg_rate")
            gpt_hits  = _get(r, "gpt_hits", 0)
            gem_hits  = _get(r, "gemini_hits", 0)
            n_sim     = _get(r, "n", 30)
            gpt_ci    = _get(r, "gpt_ci", (None, None))
            gem_ci    = _get(r, "gemini_ci", (None, None))

            # 질문 텍스트
            q_text = questions[i] if i < len(questions) else f"질문 {i+1}"
            st.markdown(f"""
            <div style="padding:12px 16px;border-radius:10px;margin-bottom:16px;
                background:{_bg2};border:1.5px solid {_border}">
                <span class="aca-q-badge">Q{i+1}</span>
                <span style="font-size:.9rem;font-weight:600;color:{_text}">{q_text}</span>
            </div>""", unsafe_allow_html=True)

            # 평균 점유율 배지
            if avg_rate is not None:
                badge_color = "#22C55E" if avg_rate >= 30 else "#F59E0B" if avg_rate >= 10 else "#EF4444"
                st.markdown(f"""
                <div style="text-align:center;margin-bottom:16px">
                    <span style="background:{badge_color};color:white;padding:6px 20px;
                        border-radius:30px;font-size:1.1rem;font-weight:800">
                        평균 점유율 {avg_rate:.1f}%
                    </span>
                </div>""", unsafe_allow_html=True)

            # 도넛 차트 2개 (GPT / Gemini)
            _donut_counter["n"] += 1

            def _make_donut(rate, hits, n, label, color, ci):
                val = rate if rate is not None else 0
                other = max(0, 100 - val)
                fig = go.Figure(go.Pie(
                    values=[val, other],
                    labels=[f"{label} 인용", "미인용"],
                    hole=0.72,
                    marker=dict(
                        colors=[color, "#E5E7EB" if not _dark else "#2A2A2A"],
                        line=dict(width=0),
                    ),
                    textinfo="none",
                    hovertemplate=f"<b>{label}</b><br>점유율: {val:.1f}%<br>히트: {hits}/{n}회<extra></extra>",
                    sort=False,
                ))
                ci_text = f"{ci[0]}~{ci[1]}%" if ci and ci[0] is not None else ""
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    annotations=[
                        dict(
                            text=f"<b style='font-size:22px'>{val:.0f}%</b>",
                            x=0.5, y=0.55, font_size=22,
                            font_color=_plot_font,
                            showarrow=False, align="center",
                        ),
                        dict(
                            text=f"{hits}/{n}회",
                            x=0.5, y=0.35, font_size=12,
                            font_color=_text_muted,
                            showarrow=False, align="center",
                        ),
                    ],
                )
                return fig, ci_text

            col_gpt, col_gem = st.columns(2)

            with col_gpt:
                st.markdown(f"""
                <div style="text-align:center;margin-bottom:4px">
                    <span style="font-size:.85rem;font-weight:700;color:{_accent}">⚡ GPT</span>
                    {"<span style='color:" + _text_muted + ";font-size:.75rem;margin-left:6px'>미연결</span>" if gpt_rate is None else ""}
                </div>""", unsafe_allow_html=True)
                if gpt_rate is not None:
                    fig, ci_text = _make_donut(gpt_rate, gpt_hits or 0, n_sim, "GPT", _accent, gpt_ci)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"donut_gpt_{_donut_counter['n']}_{i}")
                    if ci_text:
                        st.caption(f"95% 신뢰구간: {ci_text}")
                else:
                    st.markdown(f"<div style='height:200px;display:flex;align-items:center;justify-content:center;color:{_text_muted}'>-</div>", unsafe_allow_html=True)

            with col_gem:
                st.markdown(f"""
                <div style="text-align:center;margin-bottom:4px">
                    <span style="font-size:.85rem;font-weight:700;color:{_accent2}">✦ Gemini</span>
                    {"<span style='color:" + _text_muted + ";font-size:.75rem;margin-left:6px'>미연결</span>" if gem_rate is None else ""}
                </div>""", unsafe_allow_html=True)
                if gem_rate is not None:
                    fig, ci_text = _make_donut(gem_rate, gem_hits or 0, n_sim, "Gemini", _accent2, gem_ci)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"donut_gem_{_donut_counter['n']}_{i}")
                    if ci_text:
                        st.caption(f"95% 신뢰구간: {ci_text}")
                else:
                    st.markdown(f"<div style='height:200px;display:flex;align-items:center;justify-content:center;color:{_text_muted}'>-</div>", unsafe_allow_html=True)


# ── 전략 렌더링 ───────────────────────────────────────────────────────────────
def render_strategy(strategy: dict, target_url: str):
    target_domain = extract_domain(target_url)
    measured      = strategy.get("measured_competitors", {})
    competitors_ai = strategy.get("competitors", [])[:5]

    # 경쟁사
    st.markdown(f"""
    <div style="font-size:1rem;font-weight:800;color:{_text};margin:0 0 12px">
        🏆 AI 인용 경쟁 현황
        <span style="font-size:.72rem;font-weight:600;color:{_text_muted};margin-left:8px">
            {"AI 인용 분석 기반" if measured else "AI 추정 ⚠️"}
        </span>
    </div>""", unsafe_allow_html=True)

    if not measured and competitors_ai:
        st.caption("⚠️ 경쟁사 조회 실패 — AI 추정값을 표시합니다.")

    pos_colors = {
        "업계1위":"#22C55E","업계 1위":"#22C55E",
        "신흥강자":"#F59E0B","신흥 강자":"#F59E0B",
        "틈새전문":"#6366F1","틈새 전문":"#6366F1",
    }

    if measured:
        items = sorted(measured.items(), key=lambda x: x[1].get("mentions", 0), reverse=True)[:5]
        for rank, (brand, data) in enumerate(items, 1):
            score    = data.get("mentions", 0)   # fetch_competitor_brands의 mention_score
            dom      = data.get("domain", "")
            reason   = data.get("reason", "")
            is_mine  = target_domain.lower() in brand.lower() or target_domain.lower() in dom.lower()
            row_style = f"background:rgba(255,107,53,.08);border:1.5px solid {_accent}" if is_mine else f"background:{_bg2};border:1.5px solid {_border}"
            fw       = "700" if is_mine else "500"
            mine_badge = f'<span style="margin-left:6px;font-size:.68rem;font-weight:700;color:white;background:{_accent};padding:1px 7px;border-radius:20px">내 사이트</span>' if is_mine else ""
            dom_div  = f'<div style="font-size:.72rem;color:{_text_muted};margin-top:1px">{dom}</div>' if dom else ""
            reason_div = f'<div style="font-size:.72rem;color:{_text_muted};margin-top:2px">{reason}</div>' if reason else ""
            # score: 100점 만점 상대적 인용 가능성 점수
            score_bar_w = max(8, int(score * 0.6))  # 최대 60px
            score_label = f"{score}점" if score <= 100 else f"{score}회"
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:12px 16px;border-radius:12px;margin:6px 0;{row_style}">'
                f'<div style="display:flex;align-items:center;gap:12px">'
                f'<div style="width:28px;height:28px;border-radius:8px;display:flex;align-items:center;'
                f'justify-content:center;font-size:.8rem;font-weight:800;color:white;'
                f'background:{_accent_gr};flex-shrink:0">{rank}</div>'
                f'<div>'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<span style="font-weight:{fw};font-size:.92rem;color:{_text}">{brand}</span>'
                f'{mine_badge}</div>'
                f'{dom_div}{reason_div}'
                f'</div></div>'
                f'<div style="text-align:right;flex-shrink:0">'
                f'<div style="background:linear-gradient(90deg,{_accent},{_accent2});'
                f'height:6px;border-radius:4px;width:{score_bar_w}px;margin-bottom:4px;margin-left:auto"></div>'
                f'<span style="font-size:.72rem;font-weight:700;color:{_accent}">{score_label}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    else:
        for comp in competitors_ai:
            r   = comp.get("rank","?")
            d   = comp.get("domain","")
            b   = comp.get("brand_name", d)
            pos = comp.get("position", comp.get("market_position",""))
            reason = comp.get("reason","")
            is_mine = target_domain.lower() in d.lower()
            mine_cls = "mine" if is_mine else ""
            pc  = pos_colors.get(pos, "#888")
            st.markdown(f"""
            <div class="aca-competitor-row {mine_cls}">
                <div style="display:flex;align-items:center;gap:12px">
                    <div style="width:26px;height:26px;border-radius:8px;display:flex;align-items:center;justify-content:center;
                        font-size:.78rem;font-weight:800;color:white;background:{_accent_gr}">{r}</div>
                    <div>
                        <span style="font-weight:{'700' if is_mine else '500'};font-size:.9rem;color:{_text}">{b}</span>
                        {"<span style='margin-left:6px;font-size:.7rem;font-weight:700;color:white;background:" + _accent + ";padding:1px 7px;border-radius:20px'>내 사이트</span>" if is_mine else ""}
                        <div style="font-size:.75rem;color:{_text_muted}">{d}</div>
                    </div>
                </div>
                <div style="text-align:right">
                    {f'<span style="background:{pc};color:white;padding:2px 8px;border-radius:20px;font-size:.72rem;font-weight:700">{pos}</span>' if pos else ""}
                    {f'<div style="font-size:.75rem;color:{_text_muted};margin-top:2px">{reason}</div>' if reason else ""}
                </div>
            </div>""", unsafe_allow_html=True)

    if not measured and not competitors_ai:
        st.caption("경쟁사 정보를 가져오지 못했습니다.")

    st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)

    # 인용 실패 진단
    st.markdown(f"""
    <div style="font-size:1rem;font-weight:800;color:{_text};margin:16px 0 12px">
        🔬 인용 실패 원인 진단
    </div>""", unsafe_allow_html=True)
    diag_colors = [_accent, _accent2, "#6366F1"]
    diag_icons  = ["❌", "⚡", "🔧"]
    for i, d in enumerate(strategy.get("diagnoses", [])):
        st.markdown(f"""
        <div class="aca-diag-item" style="border-left:4px solid {diag_colors[i%3]}">
            <span style="font-size:.88rem;color:{_text};line-height:1.6">
                {diag_icons[i%3]} {d}
            </span>
        </div>""", unsafe_allow_html=True)

    # 블루오션 키워드
    st.markdown(f"""
    <div style="font-size:1rem;font-weight:800;color:{_text};margin:16px 0 12px">
        🌊 블루오션 키워드
    </div>""", unsafe_allow_html=True)
    for i, kw in enumerate(strategy.get("keywords", [])):
        st.markdown(f"""
        <div class="aca-kw-item">
            <span style="background:{_accent_gr};color:white;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:800">#{i+1}</span>
            <span style="font-size:.88rem;font-weight:600;color:{_text}">{kw}</span>
        </div>""", unsafe_allow_html=True)

    # GEO 가이드
    st.markdown(f"""
    <div style="font-size:1rem;font-weight:800;color:{_text};margin:16px 0 4px">
        📋 GEO 최적화 가이드
    </div>
    <div style="font-size:.8rem;color:{_text_muted};margin-bottom:12px">
        아래 항목을 적용하면 AI가 내 사이트를 더 자주 인용합니다.
    </div>""", unsafe_allow_html=True)
    guide_colors = [_accent, _accent2, "#6366F1"]
    geo_guides = strategy.get("geo_guides", [])
    numbered = [g.strip() for g in geo_guides if re.match(r'^\d+[.)]?\s', g.strip())]
    display_guides = (numbered if numbered else [g for g in geo_guides if g.strip()])[:3]
    for i, g in enumerate(display_guides):
        g_html = re.sub(r'\*\*(.*?)\*\*', rf'<strong style="color:{_text}">\1</strong>', g)
        g_html = g_html.replace('\n', '<br>')
        st.markdown(f"""
        <div class="aca-geo-item" style="border-left:3px solid {guide_colors[i%3]}">
            <div style="display:flex;gap:12px;align-items:flex-start">
                <div style="min-width:28px;height:28px;border-radius:8px;flex-shrink:0;
                    background:{_accent_gr};color:white;font-weight:800;font-size:.82rem;
                    display:flex;align-items:center;justify-content:center">{i+1}</div>
                <div style="font-size:.87rem;color:{_text};line-height:1.75">{g_html}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:20px 0 24px;border-bottom:1.5px solid {_border};margin-bottom:20px">
        <div style="font-size:2rem;margin-bottom:6px">🔍</div>
        <div style="color:{_text};font-size:1rem;font-weight:800">AI Citation Analyzer</div>
        <div style="background:{_accent_gr};color:white;font-size:.65rem;font-weight:700;
            padding:2px 10px;border-radius:20px;display:inline-block;margin-top:4px">v4.0</div>
    </div>""", unsafe_allow_html=True)

    if st.button("🌙 다크 모드" if not _dark else "☀️ 라이트 모드", key="btn_dark", use_container_width=True):
        st.session_state["dark_mode"] = not _dark
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    openai_key        = st.text_input("OpenAI API Key",  type="password", placeholder="sk-...")
    gemini_key        = st.text_input("Gemini API Key",  type="password", placeholder="AIza...")
    gpt_model         = st.selectbox("GPT 모델",   ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    _GEMINI_MODELS = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
    ]
    gemini_model_name = st.selectbox("Gemini 모델", _GEMINI_MODELS, index=0)

    st.markdown("---")

    sim_count    = st.slider("시뮬레이션 횟수", 10, 100, 30, 10)
    market_scope = st.radio("경쟁사 범위", ["국내 (대한민국)", "글로벌"], horizontal=True)

    st.markdown("---")

    gpt_ok    = bool(openai_key and openai_key.startswith("sk-"))
    gemini_ok = bool(gemini_key and len(gemini_key) > 10)

    c1, c2 = st.columns(2)
    c1.markdown(f"{'🟢' if gpt_ok else '⚪'} **GPT**")
    c2.markdown(f"{'🟢' if gemini_ok else '🔴'} **Gemini**")

    tracker: CostTracker = st.session_state["cost_tracker"]
    cost_summary = tracker.summary()
    if cost_summary["api_calls"] > 0:
        st.markdown(f"""
        <div style="background:rgba(255,107,53,.08);border:1px solid rgba(255,107,53,.25);
            border-radius:8px;padding:10px 12px;margin-top:8px;font-size:.75rem;color:{_text}">
            💰 누적 비용: ~${cost_summary['estimated_usd']:.4f}<br>
            🔢 API 호출: {cost_summary['api_calls']}회
        </div>""", unsafe_allow_html=True)

    cache_stats = _cache.stats()
    if cache_stats["entries"] > 0:
        st.markdown(f"""
        <div style="background:{_bg2};border:1px solid {_border};
            border-radius:8px;padding:8px 12px;margin-top:6px;font-size:.72rem;color:{_text_muted}">
            ⚡ 캐시 히트율: {cache_stats['hit_rate']}% ({cache_stats['entries']}개)
        </div>""", unsafe_allow_html=True)

    if st.button("🗑️ 캐시 초기화", key="btn_clear_cache", use_container_width=True):
        _cache._store.clear()
        st.session_state["cache_data"]   = {}
        st.session_state["cost_tracker"] = CostTracker()
        st.success("초기화 완료")

    st.markdown("---")
    if st.button("🎬 데모 실행", key="btn_demo_sidebar", use_container_width=True):
        st.session_state["run_demo"] = True
        st.rerun()


# ── 클라이언트 초기화 ──────────────────────────────────────────────────────────
def get_clients():
    client_gpt = client_gemini = None
    if gpt_ok:
        with CaptureError("init_gpt", log_level="error"):
            import openai
            client_gpt = openai.OpenAI(api_key=openai_key)
    if gemini_ok:
        with CaptureError("init_gemini", log_level="error"):
            # 신 SDK(google.genai): Client를 매번 생성하므로 (api_key, model_name) 튜플 전달
            client_gemini = (gemini_key, gemini_model_name)
    return client_gpt, client_gemini


# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,#FF6B35 0%,#FF4D8B 100%);
    border-radius:20px;padding:28px 36px;margin-bottom:24px;
    box-shadow:0 8px 32px rgba(255,107,53,.35)">
    <div style="color:white;font-size:1.8rem;font-weight:800;line-height:1.2">
        🔍 AI 검색 점유율 분석
    </div>
    <div style="color:rgba(255,255,255,.88);font-size:.9rem;margin-top:6px">
        GPT &amp; Gemini가 내 사이트를 얼마나 인용하는지 실측합니다
    </div>
</div>
""", unsafe_allow_html=True)

client_gpt, client_gemini = get_clients()

# ── Gemini 단독 테스트 ────────────────────────────────────────────────────────
with st.expander("🧪 Gemini 연결 테스트 (디버그용)", expanded=False):
    test_q = st.text_input("테스트 질문", value="네이버 같은 포털 사이트 추천해줘?", key="gem_test_q")
    if st.button("Gemini 직접 호출", key="btn_gem_test"):
        if not gemini_ok:
            st.error("Gemini API 키를 먼저 입력하세요.")
        else:
            with st.spinner("Gemini 호출 중..."):
                try:
                    import google.genai as genai
                    from google.genai import types as gtypes
                    _api_key, _model_name = client_gemini
                    st.write(f"모델명: `{_model_name}`")
                    _client = genai.Client(api_key=_api_key)
                    _test_resp = _client.models.generate_content(
                        model=_model_name,
                        contents=test_q,
                        config=gtypes.GenerateContentConfig(
                            max_output_tokens=300, temperature=0.7)
                    )
                    _text = (_test_resp.text or "").strip()
                    if _text:
                        st.success("✅ Gemini 응답 성공")
                        st.code(_text[:500])
                        # 브랜드 매칭 테스트
                        from core.citation import build_brand_variants, detect_citation
                        _url = url_input.strip() if 'url_input' in st.session_state else ""
                        _brand = brand_input.strip() if 'brand_input' in st.session_state else ""
                        if _url and _brand:
                            _variants = build_brand_variants(
                                ("https://"+_url if not _url.startswith("http") else _url),
                                {"brand_name": _brand}
                            )
                            st.write(f"brand_variants: `{_variants}`")
                            _result = detect_citation(_text, _variants)
                            from core.ai_client import _check_mention
                            _cm = _check_mention(_text, _variants)
                            st.write(f"detect_citation: cited=`{_result.cited}` conf=`{_result.confidence:.2f}`")
                            st.write(f"_check_mention: `{_cm}`")
                    else:
                        st.error("❌ Gemini 응답이 빈 문자열")
                        st.write("candidates:", _test_resp.candidates if hasattr(_test_resp, "candidates") else "없음")
                except Exception as e:
                    st.error(f"❌ 예외 발생: {type(e).__name__}: {e}")

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab_main, tab_hist = st.tabs(["📊 인용 점유율 분석", "🕘 히스토리"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: 분석
# ════════════════════════════════════════════════════════════════════════
with tab_main:

    # 입력 카드
    st.markdown(f'<div class="aca-card">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:.95rem;font-weight:800;color:{_text};margin-bottom:14px">
        ⚡ 분석 자동화
        <span class="aca-badge-running">RUNNING</span>
    </div>""", unsafe_allow_html=True)

    col_u, col_b = st.columns([2, 1])
    with col_u:
        url_input = st.text_input("🌐 사이트 URL *", placeholder="example.com", key="url_input")
    with col_b:
        brand_input = st.text_input("🏷️ 브랜드명 *", placeholder="", key="brand_input")

    q_count = st.session_state["q_count"]
    st.markdown(f"""
    <div style="font-size:.82rem;font-weight:700;color:{_text};margin:12px 0 8px">
        🎯 분석 질문
        <span style="color:{_text_muted};font-weight:400;font-size:.76rem;margin-left:6px">{q_count}/5개</span>
    </div>""", unsafe_allow_html=True)

    questions_input = []
    for idx in range(q_count):
        col_q, col_del = st.columns([10, 1])
        with col_q:
            q = st.text_input(
                f"Q{idx+1}",
                value=st.session_state["questions_list"][idx] if idx < len(st.session_state["questions_list"]) else "",
                placeholder="",
                key=f"q_dyn_{idx}", label_visibility="visible"
            )
            if idx < len(st.session_state["questions_list"]):
                st.session_state["questions_list"][idx] = q
            if q.strip():
                questions_input.append(q.strip())
        with col_del:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if q_count > 1:
                if st.button("✕", key=f"del_q_{idx}", help="삭제"):
                    st.session_state["questions_list"].pop(idx)
                    st.session_state["q_count"] = max(1, q_count - 1)
                    st.rerun()

    col_add, col_run, col_reset = st.columns([2, 4, 1])
    with col_add:
        if q_count < 5:
            if st.button(f"＋ 질문 추가", key="btn_add_q"):
                st.session_state["questions_list"].append("")
                st.session_state["q_count"] = min(5, q_count + 1)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # aca-card

    col_run2, col_reset2 = st.columns([5, 1])
    with col_run2:
        run_btn = st.button("🚀 분석 시작", key="btn_run", use_container_width=True)
    with col_reset2:
        if st.button("초기화", key="btn_reset", use_container_width=True):
            for k in ["url_input", "brand_input"]:
                if k in st.session_state: del st.session_state[k]
            st.session_state["q_count"] = 1
            st.session_state["questions_list"] = [""]
            st.rerun()

    # ── 데모 ──
    if st.session_state.get("run_demo"):
        st.session_state["run_demo"] = False
        demo_url = normalize_url(url_input.strip() if url_input.strip() else "naver.com")
        dd = get_demo(demo_url)
        st.info(f"🎬 데모 모드 — {dd['brand']} ({dd['industry']})")

        # 워크플로우 요약
        render_workflow_cards(dd["brand"], len(dd["questions"]), len(dd["results"]), 2)
        st.markdown(f'<div class="aca-card" style="margin-top:16px">', unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:.95rem;font-weight:800;color:{_text};margin-bottom:12px">
            📊 '{dd['brand']}' AI 인용 점유율</div>""", unsafe_allow_html=True)
        render_donut_tabs(dd["results"], dd["questions"], dd["brand"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        render_strategy(_DEMO_STRATEGY, demo_url)

    # ── 실제 분석 실행 ──
    elif run_btn:
        errors = []
        if not url_input.strip():   errors.append("사이트 URL")
        if not brand_input.strip(): errors.append("브랜드명")
        if not questions_input:     errors.append("질문 (최소 1개)")

        if errors:
            st.error(f"필수 입력 누락: {', '.join(errors)}")
        elif not gpt_ok and not gemini_ok:
            st.error("사이드바에서 API 키를 먼저 입력해주세요.")
        else:
            target_url = normalize_url(url_input.strip())
            domain     = extract_domain(target_url)
            brand_name = brand_input.strip()

            biz_info = BusinessInfo(
                brand_name=brand_name,
                industry="",           # run_strategy_analysis 내부에서 크롤 후 채워짐
                industry_category="기타",
                core_product=f"{brand_name} 서비스",
                target_audience="잠재 고객",
                confidence="high",
                crawl_tier=0,
            )

            # 실행 상태 배너
            st.markdown(f"""
            <div class="aca-running-banner">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                    <span style="font-size:.95rem;font-weight:700;color:{_accent}">⚡ 분석 자동화</span>
                    <span class="aca-badge-running">RUNNING</span>
                </div>
                <div style="font-size:.82rem;color:{_text_muted}">
                    AI가 콘텐츠 생성 → 집계 → 분석 중입니다.
                </div>
            </div>""", unsafe_allow_html=True)

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
                tracker=tracker, use_cache=True,
            )
            elapsed_sim = int(time.time() - t0)
            prog.progress(0.6)
            save_cache()
            st.session_state["cost_tracker"] = tracker
            stat.success(f"시뮬레이션 완료 ({elapsed_sim}초) | ~${tracker.summary()['estimated_usd']:.4f}")

            # ── Gemini 디버그 (0% 원인 파악용) ──
            with st.expander("🔍 Gemini 디버그 (0% 원인 파악)", expanded=True):
                for i, _r in enumerate(all_results):
                    _rd = _r.to_dict() if hasattr(_r, "to_dict") else _r
                    gem_rate = _rd.get("gemini_rate")
                    gem_hits = _rd.get("gemini_hits")
                    gem_samp = _rd.get("gemini_samples", [])
                    st.markdown(f"**Q{i+1}** gemini_rate={gem_rate} hits={gem_hits}")
                    if gem_samp:
                        for s in gem_samp[:2]:
                            st.code(s[:300], language=None)
                    else:
                        st.warning("gemini_samples 없음 → 응답이 빈 문자열이거나 hit이 없음")

            # 워크플로우 카드
            n_engines = (1 if client_gpt else 0) + (1 if client_gemini else 0)
            render_workflow_cards(brand_name, len(questions_input), len(all_results), n_engines, elapsed_sim)

            # 도넛 차트 탭
            st.markdown(f'<div class="aca-card">', unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:.95rem;font-weight:800;color:{_text};margin-bottom:12px">
                📊 '{brand_name}' AI 인용 점유율</div>""", unsafe_allow_html=True)
            render_donut_tabs(
                [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results],
                questions_input, brand_name
            )
            st.markdown("</div>", unsafe_allow_html=True)

            prog.progress(0.7)
            stat.markdown("🧠 **전략 분석 생성 중...**")

            # Option B: 전용 경쟁사 브랜드 조회 (1회 호출)
            with CaptureError("competitor_brands", log_level="warning"):
                competitor_brands = fetch_competitor_brands(
                    client_gpt, client_gemini,
                    questions=questions_input,
                    target_url=target_url,
                    brand_name=brand_name,
                    model_gpt=gpt_model,
                    market_scope=market_scope,
                    tracker=tracker,
                    use_cache=True,
                )

            with CaptureError("strategy", log_level="warning") as s_ctx:
                strategy = run_strategy_analysis(
                    client_gpt, client_gemini,
                    biz_info=biz_info, competitors=[],
                    sim_results=all_results, questions=questions_input,
                    tracker=tracker, target_url=target_url,
                    model_gpt=gpt_model, market_scope=market_scope,
                    use_cache=True,
                )

            # 전용 조회 결과를 경쟁 현황에 주입
            if strategy and competitor_brands:
                strategy["measured_competitors"] = competitor_brands

            prog.progress(1.0)

            if s_ctx.ok and strategy:
                st.markdown(f'<div class="aca-card">', unsafe_allow_html=True)
                render_strategy(strategy, target_url)
                st.markdown("</div>", unsafe_allow_html=True)
                stat.success("✅ 전략 분석 완료!")
            else:
                st.warning("전략 분석을 완료하지 못했습니다. API 키를 확인하세요.")

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


# ════════════════════════════════════════════════════════════════════════
# TAB 2: 히스토리
# ════════════════════════════════════════════════════════════════════════
with tab_hist:
    history = st.session_state.get("history", [])
    if not history:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px">
            <div style="font-size:3rem;margin-bottom:12px">📭</div>
            <p style="color:{_text_muted};font-size:1rem">아직 분석 기록이 없습니다.</p>
            <p style="color:{_text_muted};font-size:.85rem">분석을 실행하면 여기에 기록됩니다.</p>
        </div>""", unsafe_allow_html=True)
    else:
        for h in reversed(history[-20:]):
            with st.expander(
                f"{h.get('ts','')} — {h.get('brand','')} ({h.get('n_q',0)}개 질문)",
                expanded=False
            ):
                if h.get("questions") and h.get("results"):
                    render_donut_tabs(h["results"], h["questions"], h.get("brand",""))
                else:
                    st.json(h)


# ── 탭 툴팁 제거 (JS) ────────────────────────────────────────────────────────
st.markdown("""
<script>
(function removeTitles() {
    const clean = () => {
        document.querySelectorAll('[data-baseweb="tab"]').forEach(el => {
            el.removeAttribute('title');
            el.removeAttribute('data-original-title');
        });
    };
    clean();
    new MutationObserver(clean).observe(document.body, { subtree: true, childList: true, attributes: true });
})();
</script>
""", unsafe_allow_html=True)

# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px;color:{_text_muted};font-size:.78rem;
    border-top:1px solid {_border};margin-top:20px">
    AI Citation Analyzer v4.0 &nbsp;·&nbsp;
    <span style="background:{_accent_gr};-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;font-weight:700">
        Powered by GPT &amp; Gemini
    </span>
</div>""", unsafe_allow_html=True)
