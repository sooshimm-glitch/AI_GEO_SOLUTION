"""
AI Citation Analyzer v3.0 — 통합형
"""

import streamlit as st
import datetime
import re
import time
import pandas as pd
import plotly.graph_objects as go
from urllib.parse import urlparse

import sys, os
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)
sys.path.insert(0, "/mount/src/ai_geo_solution")

try:
    from core.cache import get_cache
    from core.logger import get_logger, CaptureError
    from core.citation import build_brand_variants
    from core.ai_client import run_all_simulations, CostTracker, SimResult
    from core.biz_analysis import run_strategy_analysis, BusinessInfo
except ImportError as _e:
    import streamlit as st
    st.set_page_config(page_title="Import Error", page_icon="❌")
    st.error(f"**core 모듈 import 실패**: {_e}")
    st.code(f"APP_ROOT: {_APP_ROOT}\nCWD: {os.getcwd()}\ncore/ 존재: {os.path.isdir(os.path.join(_APP_ROOT, 'core'))}")
    st.stop()

logger = get_logger("app")

st.set_page_config(
    page_title="AI Citation Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode"      not in st.session_state: st.session_state["dark_mode"]      = False
if "cache_data"     not in st.session_state: st.session_state["cache_data"]     = {}
if "cost_tracker"   not in st.session_state: st.session_state["cost_tracker"]   = CostTracker()
if "history"        not in st.session_state: st.session_state["history"]        = []
if "run_demo"       not in st.session_state: st.session_state["run_demo"]       = False
if "q_count"        not in st.session_state: st.session_state["q_count"]        = 1
if "questions_list" not in st.session_state: st.session_state["questions_list"] = [""]

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


def render_strategy(strategy: dict, target_url: str):
    domain = extract_domain(target_url)
    st.markdown("### 🏆 AI 인용 경쟁 현황 (TOP 5)")
    pos_colors = {
        "업계1위":"#10B981","업계 1위":"#10B981",
        "신흥강자":"#F59E0B","신흥 강자":"#F59E0B",
        "틈새전문":"#6366F1","틈새 전문":"#6366F1",
    }
    competitors = strategy.get("competitors", [])[:5]
    if competitors:
        for comp in competitors:
            r=comp.get("rank","?"); d=comp.get("domain",""); b=comp.get("brand_name",d)
            reason=comp.get("reason",""); pos=comp.get("position",comp.get("market_position",""))
            is_t=domain.lower() in d.lower()
            if _dark:
                bg="linear-gradient(135deg,#2A2A2A,#333)" if is_t else "#1E1E1E"
                bd="#888" if is_t else "#333"
            else:
                bg="linear-gradient(135deg,#EEE,#E0E0E0)" if is_t else "#F8F8F8"
                bd="#111" if is_t else "#DDD"
            lbl=" <- 내 사이트" if is_t else ""
            pc=pos_colors.get(pos,"#888")
            pb=(f'<span style="background:{pc};color:white;padding:2px 8px;border-radius:20px;font-size:.72rem;font-weight:700;margin-left:8px;">{pos}</span>' if pos else "")
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
                        {b} <span style="color:{_text_muted};font-size:.78rem;">({d})</span>{lbl}{pb}
                    </span>
                </div>
                <span style="color:{_text_muted};font-size:.8rem;">{reason}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.caption("경쟁사 정보를 가져오지 못했습니다.")

    st.markdown(f"<hr style='border:none;height:1px;background:linear-gradient(90deg,transparent,{_border},transparent);margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("### 🔬 인용 실패 원인 진단")
    _icons = ["❌", "⚡", "🔧"]
    for i, d in enumerate(strategy.get("diagnoses", [])):
        col = ["#111" if not _dark else "#DDD", "#555", "#888"][i % 3]
        st.markdown(f"""
        <div class="strategy-item" style="border-left:4px solid {col};">
            <span>{_icons[i % len(_icons)]} {d}</span>
        </div>""", unsafe_allow_html=True)

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


with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🔍</span>
        <h2>AI Citation Analyzer</h2>
        <p>v3.0 - 통합 분석</p>
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

            stat.markdown(f"⚡ **{len(questions_input)}개 질문 x {sim_count}회 시뮬레이션 중...**")
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
                render_strategy(strategy, target_url)
                stat.success("전략 분석 완료!")
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
                "results":   [r.to_dict() if hasattr(r,"to_dict") else r for r in all_results],
            })


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
    AI Citation Analyzer v3.0 &nbsp;|&nbsp;
    <span style="color:{_text_muted};">Powered by GPT &amp; Gemini</span>
</div>""", unsafe_allow_html=True)
