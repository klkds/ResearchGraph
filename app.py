"""ResearchGraph — Streamlit Application

A graph-based academic research discovery and idea generation system.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import DataLoader
from src.research_graph import ResearchGraph, ScoredPath
from src.idea_engine import IdeaEngine, _identify_concept_transfer
from src.graph_viz import (
    render_neighborhood,
    render_path,
    neighborhood_legend_html,
    path_legend_html,
)
from src.app_helpers import rubric_mapping_rows

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchGraph",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium-dashboard styling
# ---------------------------------------------------------------------------
# Goal: lift the default Streamlit look toward something that reads as a
# research-intelligence product — calm cool-gray canvas, modern type, soft
# shadows, refined controls. Applied globally; the Graph Explorer page
# layers its own workspace-card / sidebar treatment on top.
_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --rg-bg:        #f3f5f8;
  --rg-surface:   #ffffff;
  --rg-border:    #e3e8ef;
  --rg-text:      #1f2933;
  --rg-text-muted:#5b6b7c;
  --rg-accent:    #2f5fb3;
  --rg-accent-soft:#eaf1fb;
  --rg-shadow:    0 1px 2px rgba(20,40,80,0.04), 0 4px 18px rgba(20,40,80,0.06);
}

html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, .stSelectbox,
.stButton button, .stRadio, .stCaption, .stTitle, .stHeader,
[data-testid="stMarkdownContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif !important;
  letter-spacing: -0.005em;
}
.stApp { background: var(--rg-bg); color: var(--rg-text); }

/* Hide the default top toolbar / "Made with Streamlit" footer so the
   product chrome reads cleaner. */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none; }

/* Generous workspace padding (top bar feels intentional, not cramped). */
.main .block-container {
  padding-top: 1.4rem;
  padding-bottom: 2.5rem;
  padding-left: 2.2rem;
  padding-right: 2.2rem;
  max-width: 1480px;
}

/* App header — the strip that introduces ResearchGraph at the top. */
.rg-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 14px 20px;
  margin-bottom: 18px;
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  border-radius: 14px;
  box-shadow: var(--rg-shadow);
}
.rg-header-brand { display: flex; align-items: center; gap: 12px; }
.rg-header-mark {
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, #2f5fb3 0%, #4a7fbf 100%);
  display: flex; align-items: center; justify-content: center;
  color: #ffffff; font-weight: 700; font-size: 15px;
  box-shadow: 0 2px 8px rgba(47,95,179,0.25);
}
.rg-header-name {
  font-size: 17px; font-weight: 600; color: var(--rg-text);
  line-height: 1.15;
}
.rg-header-tagline {
  font-size: 12.5px; color: var(--rg-text-muted);
  margin-top: 1px;
}
.rg-header-meta {
  display: flex; gap: 16px; align-items: center;
  font-size: 12.5px; color: var(--rg-text-muted);
}
.rg-header-meta b { color: var(--rg-text); font-weight: 600; }
.rg-header-meta .rg-dot {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: #4a7fbf; margin-right: 6px; vertical-align: middle;
}
.rg-source-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.02em; text-transform: uppercase;
  border: 1px solid;
}
.rg-source-pill .rg-source-dot {
  width: 6px; height: 6px; border-radius: 50%; background: currentColor;
}
.rg-source-pill.is-bundled  { background: #f1f3f6; color: #5b6b7c; border-color: #dde2e8; }
.rg-source-pill.is-live     { background: #eaf6ec; color: #3d7c4c; border-color: #c8e3cd; }
.rg-source-pill.is-cached   { background: #eaf1fb; color: #2f5fb3; border-color: #d4e1f4; }
.rg-source-pill.is-fallback { background: #fff7e8; color: #b67e1c; border-color: #f1d9a9; }

/* Page-level section header (above each page's main content). */
.rg-page-title {
  font-size: 22px; font-weight: 600; color: var(--rg-text);
  letter-spacing: -0.01em; margin: 4px 0 2px 0;
}
.rg-page-subtitle {
  font-size: 13.5px; color: var(--rg-text-muted);
  margin: 0 0 14px 0; line-height: 1.5; max-width: 760px;
}

/* Sidebar polish (left navigation panel). */
[data-testid="stSidebar"] {
  background: var(--rg-surface) !important;
  border-right: 1px solid var(--rg-border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
  font-size: 15px; font-weight: 600; color: var(--rg-text);
}
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stSlider > label {
  font-size: 12px; font-weight: 600; color: var(--rg-text-muted);
  text-transform: uppercase; letter-spacing: 0.04em;
}

/* Streamlit buttons: compact, refined. */
.stButton > button {
  border-radius: 8px;
  border: 1px solid var(--rg-border);
  background: var(--rg-surface);
  color: var(--rg-text);
  font-weight: 500;
  font-size: 13px;
  padding: 6px 14px;
  transition: all 0.15s ease;
  box-shadow: 0 1px 1px rgba(20,40,80,0.03);
}
.stButton > button:hover {
  border-color: #c2cdd9;
  background: #fafbfc;
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(20,40,80,0.06);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
  background: var(--rg-accent);
  border-color: var(--rg-accent);
  color: #ffffff;
}
.stButton > button[kind="primary"]:hover {
  background: #244c93;
  border-color: #244c93;
}

/* Selectbox + text input refinement. */
[data-baseweb="select"] > div, .stTextInput input {
  border-radius: 8px !important;
  border-color: var(--rg-border) !important;
  background: var(--rg-surface) !important;
}

/* Workspace card — wraps the graph canvas (used on Graph Explorer). */
.rg-workspace {
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  border-radius: 20px;
  box-shadow: var(--rg-shadow);
  padding: 14px 14px 4px 14px;
  margin-top: 4px;
}

/* Section divider — softer than st.markdown('---'). */
hr { border-color: var(--rg-border) !important; }

/* ============================================================
   Home — hero / stat tiles / feature grid
   ============================================================ */
.rg-hero {
  position: relative;
  background:
    radial-gradient(900px 240px at 0% 0%, rgba(47,95,179,0.10), transparent 60%),
    radial-gradient(700px 200px at 100% 0%, rgba(122,150,210,0.10), transparent 70%),
    linear-gradient(180deg, #ffffff 0%, #f9fbfe 100%);
  border: 1px solid var(--rg-border);
  border-radius: 22px;
  padding: 36px 36px 30px 36px;
  margin: 4px 0 22px 0;
  box-shadow: var(--rg-shadow);
  overflow: hidden;
}
.rg-hero-eyebrow {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11.5px; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--rg-accent);
  background: var(--rg-accent-soft);
  border: 1px solid #d4e1f4;
  padding: 4px 10px;
  border-radius: 999px;
  margin-bottom: 14px;
}
.rg-hero-eyebrow .rg-hero-dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--rg-accent);
}
.rg-hero-title {
  font-size: 34px; font-weight: 700;
  color: var(--rg-text);
  letter-spacing: -0.018em;
  line-height: 1.15;
  margin: 0 0 10px 0;
  max-width: 820px;
}
.rg-hero-title em {
  font-style: normal;
  background: linear-gradient(90deg, #2f5fb3 0%, #6189cc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.rg-hero-tagline {
  font-size: 14.5px; color: var(--rg-text-muted);
  line-height: 1.55; max-width: 700px;
  margin: 0;
}

.rg-stat-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 22px;
}
.rg-stat-tile {
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: var(--rg-shadow-sm);
  transition: all 0.16s ease;
}
.rg-stat-tile:hover {
  border-color: var(--rg-border-strong);
  transform: translateY(-1px);
  box-shadow: var(--rg-shadow);
}
.rg-stat-label {
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--rg-text-faint);
}
.rg-stat-value {
  font-size: 24px; font-weight: 700;
  color: var(--rg-text);
  letter-spacing: -0.02em;
  margin-top: 4px;
}
.rg-stat-sub {
  font-size: 11.5px; color: var(--rg-text-muted); margin-top: 2px;
}

.rg-section-title {
  font-size: 12px; font-weight: 700;
  color: var(--rg-text-faint);
  letter-spacing: 0.07em; text-transform: uppercase;
  margin: 6px 0 12px 0;
}

.rg-feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
@media (max-width: 1080px) {
  .rg-feature-grid { grid-template-columns: repeat(2, 1fr); }
  .rg-stat-row    { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 720px) {
  .rg-feature-grid { grid-template-columns: 1fr; }
}
.rg-feature-card {
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  border-radius: 14px;
  padding: 18px 18px 16px 18px;
  box-shadow: var(--rg-shadow-sm);
  transition: all 0.16s ease;
}
.rg-feature-card:hover {
  border-color: var(--rg-border-strong);
  transform: translateY(-2px);
  box-shadow: var(--rg-shadow);
}
.rg-feature-icon {
  width: 36px; height: 36px;
  border-radius: 10px;
  background: var(--rg-accent-soft);
  color: var(--rg-accent);
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  margin-bottom: 12px;
  border: 1px solid #d4e1f4;
}
.rg-feature-name {
  font-size: 14.5px; font-weight: 600;
  color: var(--rg-text);
  margin-bottom: 4px;
}
.rg-feature-desc {
  font-size: 12.8px; color: var(--rg-text-muted);
  line-height: 1.5;
}
.rg-feature-card kbd {
  display: inline-block;
  font-family: 'Inter', sans-serif;
  font-size: 11px; font-weight: 600;
  background: #eef2f7; color: var(--rg-text-muted);
  border: 1px solid #dee4eb;
  border-radius: 5px;
  padding: 1px 6px;
  margin-top: 10px;
}

/* ============================================================
   User Guide — sectioned docs page
   ============================================================ */
.rg-guide {
  max-width: 1020px;
}
.rg-guide-section {
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  border-radius: 16px;
  padding: 22px 26px 22px 26px;
  margin-bottom: 18px;
  box-shadow: var(--rg-shadow-sm);
}
.rg-guide-section h2 {
  font-size: 18px; font-weight: 600;
  color: var(--rg-text);
  letter-spacing: -0.01em;
  margin: 0 0 6px 0;
  display: flex; align-items: center; gap: 10px;
}
.rg-guide-section h2 .rg-guide-icon {
  width: 28px; height: 28px;
  border-radius: 8px;
  background: var(--rg-accent-soft);
  color: var(--rg-accent);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 15px;
  border: 1px solid #d4e1f4;
}
.rg-guide-section h3 {
  font-size: 13.5px; font-weight: 600;
  color: var(--rg-text);
  margin: 16px 0 6px 0;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--rg-border);
}
.rg-guide-section p {
  font-size: 13.2px; color: var(--rg-text);
  line-height: 1.6;
  margin: 0 0 10px 0;
}
.rg-guide-lede {
  font-size: 13.5px; color: var(--rg-text-muted);
  line-height: 1.55;
  margin: 0 0 6px 0;
}
.rg-guide-list {
  list-style: none; padding: 0; margin: 6px 0;
}
.rg-guide-list li {
  font-size: 13px; color: var(--rg-text);
  line-height: 1.55;
  padding: 6px 0;
  border-bottom: 1px dashed #eef0f4;
  display: grid;
  grid-template-columns: 170px 1fr;
  gap: 14px;
}
.rg-guide-list li:last-child { border-bottom: 0; }
.rg-guide-list li b {
  color: var(--rg-text);
  font-weight: 600;
}
.rg-guide-list li .rg-guide-key {
  display: inline-flex; align-items: center; gap: 6px;
  color: var(--rg-text);
  font-weight: 600;
}
.rg-guide-list li code {
  background: #eef2f7;
  border: 1px solid #dee4eb;
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 11.5px;
  color: #2c3e50;
}
.rg-guide-icon-sm {
  display: inline-flex;
  width: 22px; height: 22px;
  border-radius: 6px;
  background: var(--rg-accent-soft);
  color: var(--rg-accent);
  align-items: center; justify-content: center;
  border: 1px solid #d4e1f4;
  flex-shrink: 0;
}
.rg-guide-icon-sm svg { width: 13px; height: 13px; }
.rg-guide-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.02em; text-transform: uppercase;
  background: var(--rg-accent-soft);
  color: var(--rg-accent);
  border: 1px solid #d4e1f4;
}
.rg-guide-pill.is-amber  { background: #fff7e8; color: #b67e1c; border-color: #f1d9a9; }
.rg-guide-pill.is-green  { background: #eaf6ec; color: #3d7c4c; border-color: #c8e3cd; }
.rg-guide-pill .rg-guide-pill-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor; opacity: 0.7;
}
.rg-edge-sample {
  display: inline-flex; align-items: center; gap: 6px;
}
.rg-edge-sample-line {
  display: inline-block; width: 28px; height: 2px;
  background: #7d8b96;
}
.rg-edge-sample-line.is-dashed {
  height: 0; background: transparent;
  border-top: 2px dashed #9ea7c9;
}
.rg-toc {
  display: flex; flex-wrap: wrap; gap: 8px;
  margin-bottom: 14px;
}
.rg-toc a {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 7px;
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  color: var(--rg-text-muted);
  font-size: 12px; font-weight: 500;
  text-decoration: none;
}
.rg-toc a:hover {
  background: var(--rg-accent-soft);
  border-color: #d4e1f4;
  color: var(--rg-accent);
}
"""


def _inject_global_css(css: str) -> None:
    """Inject a stylesheet into the *parent* document.

    Streamlit 1.56 sanitizes ``<style>`` tags out of both
    ``st.markdown(unsafe_allow_html=True)`` and ``st.html`` (DOMPurify
    strips them as a default-deny rule), so the rules end up rendered
    as visible body text. Workaround: use a zero-height
    ``components.html`` iframe whose JS appends a real ``<style>``
    element to the parent document's ``<head>`` — that never goes
    through the sanitizer, and the resulting stylesheet applies
    globally to the Streamlit app.
    """
    bridge = """
<script>
(function() {
  try {
    var topDoc = (window.top || window.parent).document;
    if (topDoc.getElementById('rg-global-styles')) return;  // idempotent
    var s = topDoc.createElement('style');
    s.id = 'rg-global-styles';
    s.textContent = __CSS__;
    topDoc.head.appendChild(s);
  } catch (e) {
    console.error('rg: cannot inject global CSS:', e);
  }
})();
</script>
""".replace("__CSS__", json.dumps(css))
    components.html(bridge, height=0)


_inject_global_css(_GLOBAL_CSS)

# ---------------------------------------------------------------------------
# Cached initialization
# ---------------------------------------------------------------------------

PRELOADED_TOPICS = {
    "Sample Dataset (bundled)": None,
    "Transformers": "transformers attention mechanism",
    "Diffusion Models": "diffusion models image generation",
    "Graph Neural Networks": "graph neural networks",
}


@st.cache_resource(show_spinner="Building research graph...")
def load_graph(
    topic_key: str, sim_threshold: float
) -> tuple[ResearchGraph, dict, dict]:
    """Build the research graph for the chosen topic + threshold.

    Returns a tuple ``(graph, stats, provenance)`` where ``provenance``
    carries the data-source label so the UI can be honest about
    whether it's showing live, cached, or fallback data:

        provenance = {
            "source": one of "bundled" | "live" | "cached"
                      | "fallback:api-error" | "fallback:empty"
                      | "error",
            "error":  str (empty unless source starts with "fallback"
                      or equals "error"),
        }
    """
    loader = DataLoader()
    papers: list = []
    try:
        if topic_key == "Sample Dataset (bundled)" or PRELOADED_TOPICS[topic_key] is None:
            papers = loader.load_local_dataset()
        else:
            query = PRELOADED_TOPICS[topic_key]
            papers = loader.fetch_from_semantic_scholar(query, limit=150)
    except Exception as exc:
        # Last-resort fallback so the app can still render the overview page
        loader.last_source = "error"
        loader.last_error = f"{type(exc).__name__}: {exc}"
        try:
            papers = loader.load_local_dataset(_keep_source=True)
        except Exception:
            papers = []
    rg = ResearchGraph()
    if papers:
        rg.build_graph(papers, similarity_threshold=sim_threshold)
    return rg, rg.stats(), {
        "source": loader.last_source,
        "error":  loader.last_error,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ResearchGraph")
st.sidebar.markdown(
    "Explore academic papers as a **knowledge graph**. "
    "Find meaningful connections, trace research trajectories, "
    "and discover structural insights invisible to keyword search."
)
st.sidebar.markdown("---")

topic = st.sidebar.selectbox("Dataset", list(PRELOADED_TOPICS.keys()))
sim_threshold = st.sidebar.slider(
    "Similarity threshold", 0.05, 0.50, 0.15, 0.05,
    help="Controls when a similarity edge is created.  Lower = denser graph.",
)

rg, graph_stats, data_provenance = load_graph(topic, sim_threshold)
engine = IdeaEngine()

# Graph stats — compact
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{graph_stats['nodes']}** papers &nbsp;&middot;&nbsp; "
    f"**{graph_stats['edges']}** edges &nbsp;&middot;&nbsp; "
    f"**{graph_stats['connected_components']}** component(s)"
)
st.sidebar.caption(
    f"{graph_stats['citation_edges']} citation &nbsp;/&nbsp; "
    f"{graph_stats['similarity_edges']} similarity &nbsp;/&nbsp; "
    f"density {graph_stats['density']}"
)

# If the user picked a live-fetch dataset but we ended up serving the
# bundled set (network blocked, rate-limited, empty result, …), tell
# them so the paper count isn't silently misleading.
if data_provenance["source"].startswith("fallback") or data_provenance["source"] == "error":
    _live_label = topic if topic != "Sample Dataset (bundled)" else "this query"
    _msg_intro = {
        "fallback:api-error": f"Live Semantic Scholar fetch for **{_live_label}** failed",
        "fallback:empty":     f"Semantic Scholar returned no results for **{_live_label}**",
        "error":              f"Could not load **{_live_label}**",
    }.get(data_provenance["source"], f"Live fetch for **{_live_label}** failed")
    _detail = (
        f" — {data_provenance['error']}." if data_provenance["error"] else "."
    )
    st.sidebar.warning(
        f"{_msg_intro}{_detail} Showing the bundled 70-paper set instead."
    )

st.sidebar.markdown("---")
_PAGE_NAMES = [
    "Home / Project Overview",
    "Search Papers",
    "Graph Explorer",
    "Research Trajectory",
    "Insights & Rankings",
    "Idea Generator",
    "User Guide",
]
# Seed the page selection from ``?page=`` if present (used by the in-canvas
# focus-swap to land back on Graph Explorer after the parent-window reload).
# Then clear the param so a later sidebar navigation isn't shadowed by it.
_qp_page = st.query_params.get("page")
if _qp_page in _PAGE_NAMES:
    st.session_state["nav_page"] = _qp_page
    del st.query_params["page"]
page = st.sidebar.radio(
    "Navigate",
    _PAGE_NAMES,
    key="nav_page",
)

# ---------------------------------------------------------------------------
# Scroll to top on page change.
# Streamlit reruns the script when the radio changes but does not reset the
# parent window's scroll position, so navigating from a long page (e.g.
# Insights) to a short one leaves you halfway down. We compare the active
# page against the last-rendered one in session_state; on a real change we
# inject a tiny same-origin script via components.html that scrolls the
# top window back to (0, 0). The iframe sandbox blocks reading the parent
# URL but allows touching its DOM, so window.top.scrollTo is permitted.
# ---------------------------------------------------------------------------
_prev_page = st.session_state.get("rg_last_page")
st.session_state["rg_last_page"] = page
if _prev_page is not None and _prev_page != page:
    # The "rg_nav_seq" counter is bumped on every real navigation so the
    # injected component HTML is unique per nav event — Streamlit caches
    # identical components.html bodies, so without the counter the second
    # nav could be a no-op.
    st.session_state["rg_nav_seq"] = st.session_state.get("rg_nav_seq", 0) + 1
    components.html(
        f"""
<script>
/* nav#{st.session_state['rg_nav_seq']} {_prev_page} -> {page} */
(function() {{
  function topWin() {{
    try {{ return window.top || window.parent; }}
    catch (e) {{ return window.parent; }}
  }}
  function scrollToTopOnce() {{
    try {{
      var w = topWin();
      if (w && typeof w.scrollTo === 'function') w.scrollTo(0, 0);
      if (w && w.document) {{
        if (w.document.scrollingElement) w.document.scrollingElement.scrollTop = 0;
        if (w.document.documentElement) w.document.documentElement.scrollTop = 0;
        if (w.document.body)            w.document.body.scrollTop = 0;
      }}
    }} catch (e) {{ /* sandboxed — give up silently */ }}
  }}
  // Run immediately, then again on the next two animation frames in case
  // Streamlit is still streaming new DOM into the parent (the parent's
  // intrinsic scrollHeight grows as widgets render, which would otherwise
  // let the browser restore the prior scroll position).
  scrollToTopOnce();
  requestAnimationFrame(function() {{
    scrollToTopOnce();
    requestAnimationFrame(scrollToTopOnce);
  }});
}})();
</script>
""",
        height=0,
    )

# ---------------------------------------------------------------------------
# Top app header — same on every page
# ---------------------------------------------------------------------------
_stats = rg.stats() if rg.papers else {"nodes": 0, "edges": 0, "density": 0}
_SOURCE_PILL = {
    "bundled":            ("is-bundled",  "Bundled"),
    "live":               ("is-live",     "Live · Semantic Scholar"),
    "cached":             ("is-cached",   "Cached · Semantic Scholar"),
    "fallback:api-error": ("is-fallback", "Bundled (live fetch failed)"),
    "fallback:empty":     ("is-fallback", "Bundled (no live results)"),
    "error":              ("is-fallback", "Bundled (load error)"),
    "none":               ("is-bundled",  "—"),
}
_pill_cls, _pill_label = _SOURCE_PILL.get(
    data_provenance["source"], ("is-bundled", data_provenance["source"])
)
st.markdown(
    f"""
<div class="rg-header">
  <div class="rg-header-brand">
    <div class="rg-header-mark">RG</div>
    <div>
      <div class="rg-header-name">ResearchGraph</div>
      <div class="rg-header-tagline">
        Academic paper discovery as a knowledge graph
      </div>
    </div>
  </div>
  <div class="rg-header-meta">
    <span class="rg-source-pill {_pill_cls}" title="Data source for the active dataset">
      <span class="rg-source-dot"></span>{_pill_label}
    </span>
    <span><span class="rg-dot"></span><b>{_stats.get('nodes', 0)}</b> papers</span>
    <span><b>{_stats.get('edges', 0)}</b> edges</span>
    <span>density <b>{_stats.get('density', 0)}</b></span>
    <span style="color:#a3afbe">·</span>
    <span>{page}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared data
# ---------------------------------------------------------------------------
paper_titles = {p.title: p.paper_id for p in rg.papers.values()}
sorted_titles = sorted(paper_titles.keys())


def _require_papers(page_label: str) -> bool:
    """Render an empty-state message and return False when no papers loaded.

    Pages that depend on at least one paper should call this and bail out
    early if it returns False.  The Home / Project Overview page does NOT
    need this because it always renders, even with an empty graph.
    """
    if sorted_titles:
        return True
    st.warning(
        f"No papers are available in the current graph for **{page_label}**. "
        f"Try the bundled sample dataset from the sidebar, or lower the "
        f"similarity threshold to surface more edges."
    )
    return False


def render_graph_legend(kind: str = "neighborhood") -> None:
    """Render the external Streamlit legend above a graph visualization.

    The pyvis canvas is intentionally legend-free; this helper writes
    the legend HTML directly into the Streamlit page so it never
    overlaps with the graph nodes.

    Parameters
    ----------
    kind : {"neighborhood", "path"}
        Selects which legend variant to render.
    """
    if kind == "path":
        html = path_legend_html()
    else:
        html = neighborhood_legend_html()
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def _paper_card(paper, show_abstract: bool = True) -> None:
    """Render a compact paper card inside an existing container."""
    authors_str = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
    if len(paper.authors) > 3:
        authors_str += f" + {len(paper.authors) - 3} more"
    st.markdown(f"**{paper.title}**")
    st.caption(
        f"{paper.year or '?'} &nbsp;&middot;&nbsp; "
        f"{paper.venue or 'N/A'} &nbsp;&middot;&nbsp; "
        f"{paper.citation_count:,} citations &nbsp;&middot;&nbsp; "
        f"{authors_str}"
    )
    if show_abstract and paper.abstract:
        abstract = paper.abstract
        if len(abstract) > 250:
            abstract = abstract[:250] + "..."
        st.markdown(f"<small>{abstract}</small>", unsafe_allow_html=True)


def _path_chain(papers: list) -> str:
    """Build a visual arrow chain: [Paper A] -> [Paper B] -> [Paper C]"""
    parts = []
    for p in papers:
        year = f", {p.year}" if p.year else ""
        parts.append(f"**{p.title}** ({p.citation_count:,} cites{year})")
    return " &rarr; ".join(parts)


def _quality_badge(label: str) -> str:
    """Return a colored quality badge string."""
    colors = {"Strong": "#2e7d32", "Moderate": "#e65100", "Weak": "#c62828"}
    color = colors.get(label, "#616161")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def _trajectory_narrative(sp: ScoredPath) -> str:
    """Build a one-sentence narrative describing *what* this trajectory reveals."""
    first, last = sp.papers[0], sp.papers[-1]
    first_topics = ", ".join(first.topic_words(3)) or "its foundational concepts"
    last_topics = ", ".join(last.topic_words(3)) or "its advanced concepts"

    if len(sp.papers) > 2:
        mid = sp.papers[len(sp.papers) // 2]
        mid_topics = ", ".join(mid.topic_words(2)) or "bridging ideas"
        return (
            f"This trajectory reveals how research on **{first_topics}** "
            f"evolves into **{last_topics}** through **{mid_topics}** — "
            f"a {sp.length - 1}-hop intellectual journey across "
            f"{'the same year' if first.year == last.year else f'{abs((last.year or 0) - (first.year or 0))} years of'} "
            f"research evolution."
        )
    return (
        f"This trajectory traces a direct connection from **{first_topics}** "
        f"to **{last_topics}**, revealing how these ideas are structurally linked "
        f"in the knowledge graph."
    )


def _render_path_result(sp: ScoredPath, idx: int, expanded: bool = False) -> None:
    """Render a single scored path as a polished card."""
    badge = _quality_badge(sp.label)
    header = (
        f"Trajectory {idx} &nbsp;&nbsp; {badge} &nbsp;&nbsp; "
        f"score {sp.score:.2f} &nbsp;&middot;&nbsp; "
        f"{sp.length - 1} hops"
    )

    with st.expander(
        f"Trajectory {idx}:  {sp.papers[0].title}  ...  {sp.papers[-1].title}  "
        f"[{sp.label}]",
        expanded=expanded,
    ):
        st.markdown(header, unsafe_allow_html=True)

        # Narrative summary — the key insight line
        st.markdown("")
        st.markdown(_trajectory_narrative(sp))

        # Visual chain
        st.markdown("---")
        st.markdown(_path_chain(sp.papers), unsafe_allow_html=True)

        # Score breakdown
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg. Similarity", f"{sp.avg_similarity:.2f}")
        c2.metric("Citation Strength", f"{sp.citation_strength:.0%}")
        c3.metric("Path Length", f"{sp.length} papers")

        # Step-by-step concept transfer
        st.markdown("---")
        st.markdown("#### How Concepts Evolve Along This Trajectory")
        for i in range(len(sp.papers) - 1):
            a, b = sp.papers[i], sp.papers[i + 1]
            transfer = _identify_concept_transfer(a, b)
            st.markdown(f"**{a.title}** &rarr; **{b.title}**", unsafe_allow_html=True)
            st.markdown(f"> {transfer}")
            st.markdown("")

        # Why this path is meaningful
        st.markdown("---")
        st.markdown("#### Why This Trajectory Matters")
        first, last = sp.papers[0], sp.papers[-1]
        first_topics = ", ".join(first.topic_words(3)) or "its concepts"
        last_topics = ", ".join(last.topic_words(3)) or "its concepts"

        reasons: list[str] = []
        if first.year and last.year and first.year != last.year:
            span = abs(last.year - first.year)
            earlier = first if first.year < last.year else last
            later = last if first.year < last.year else first
            reasons.append(
                f"This trajectory spans **{span} years** of intellectual evolution, "
                f"from *{earlier.title}* ({earlier.year}) to *{later.title}* "
                f"({later.year}), tracing how foundational ideas were transformed "
                f"and extended over time."
            )
        if sp.citation_strength > 0.5:
            reasons.append(
                f"**{sp.citation_strength:.0%}** of the edges are direct citation links, "
                f"confirming that this trajectory follows a real chain of intellectual "
                f"influence — each paper explicitly builds on its predecessor."
            )
        if sp.avg_similarity > 0.3:
            reasons.append(
                f"The average text similarity of **{sp.avg_similarity:.2f}** indicates "
                f"strong topical coherence — the concepts flow naturally from one paper "
                f"to the next without abrupt topic shifts."
            )
        if len(sp.papers) > 2:
            mid = sp.papers[len(sp.papers) // 2]
            mid_topics = ", ".join(mid.topic_words(2)) or "its ideas"
            reasons.append(
                f"The pivotal paper **{mid.title}** acts as the conceptual bridge: "
                f"it translates {first_topics} into {last_topics} through "
                f"its work on {mid_topics}."
            )
        if not reasons:
            reasons.append(
                f"This trajectory reveals a structural connection between research on "
                f"{first_topics} and {last_topics} — a link that exists in the graph's "
                f"topology but would be invisible to keyword search alone."
            )
        for r in reasons:
            st.markdown(f"- {r}")

        # Path graph visualization
        st.markdown("---")
        st.markdown("#### Trajectory in the Knowledge Graph")
        st.caption(
            "Red nodes and edges trace the trajectory. Light gray nodes "
            "show the surrounding research context. Labels are shortened "
            "for readability — hover over any node to see full metadata."
        )
        render_graph_legend(kind="path")
        path_ids = [p.paper_id for p in sp.papers]
        path_html = render_path(rg, path_ids, context_radius=1)
        components.html(path_html, height=600, scrolling=False)

        # Integrated idea generation
        st.markdown("---")
        if st.button(f"Generate Research Idea", key=f"idea_{idx}_{id(sp)}"):
            with st.spinner("Generating..."):
                idea = engine.generate_research_idea(sp.papers)
            st.markdown(idea)


# ===================================================================
# PAGES
# ===================================================================

# ---------------------------------------------------------------------------
# HOME / PROJECT OVERVIEW
# ---------------------------------------------------------------------------
if page == "Home / Project Overview":
    # ----- Hero ----------------------------------------------------------
    st.markdown(
        """
<div class="rg-hero">
  <div class="rg-hero-eyebrow">
    <span class="rg-hero-dot"></span>
    Research intelligence
  </div>
  <h1 class="rg-hero-title">
    Read the literature as a <em>knowledge graph</em>,
    not a search-result list.
  </h1>
  <p class="rg-hero-tagline">
    ResearchGraph turns academic papers into a force-directed network of
    citations, semantic similarity, and shared authorship — surfacing
    hubs, bridges, clusters, and trajectories that flat search results
    can never reveal.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ----- Stat tiles ----------------------------------------------------
    st.markdown(
        f"""
<div class="rg-stat-row">
  <div class="rg-stat-tile">
    <div class="rg-stat-label">Papers</div>
    <div class="rg-stat-value">{graph_stats['nodes']:,}</div>
    <div class="rg-stat-sub">Active dataset · {topic}</div>
  </div>
  <div class="rg-stat-tile">
    <div class="rg-stat-label">Edges</div>
    <div class="rg-stat-value">{graph_stats['edges']:,}</div>
    <div class="rg-stat-sub">
      {graph_stats['citation_edges']} citation · {graph_stats['similarity_edges']} similarity
    </div>
  </div>
  <div class="rg-stat-tile">
    <div class="rg-stat-label">Components</div>
    <div class="rg-stat-value">{graph_stats['connected_components']}</div>
    <div class="rg-stat-sub">Connected sub-graphs</div>
  </div>
  <div class="rg-stat-tile">
    <div class="rg-stat-label">Density</div>
    <div class="rg-stat-value">{graph_stats['density']}</div>
    <div class="rg-stat-sub">Edge-to-node ratio</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ----- Feature cards -------------------------------------------------
    st.markdown(
        '<div class="rg-section-title">What you can do</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="rg-feature-grid">

  <div class="rg-feature-card">
    <div class="rg-feature-icon">⌕</div>
    <div class="rg-feature-name">Search Papers</div>
    <div class="rg-feature-desc">
      Find papers by keyword across title, abstract, and topic tags.
      Each result also shows its structural role in the graph.
    </div>
  </div>

  <div class="rg-feature-card">
    <div class="rg-feature-icon">◉</div>
    <div class="rg-feature-name">Graph Explorer</div>
    <div class="rg-feature-desc">
      Anchor any paper at the center, drag the network around, and
      hover to spotlight its 1-hop neighbors. The right sidebar shows
      live metadata for whatever node you're inspecting.
    </div>
  </div>

  <div class="rg-feature-card">
    <div class="rg-feature-icon">↝</div>
    <div class="rg-feature-name">Research Trajectory</div>
    <div class="rg-feature-desc">
      Connect two papers (or two topics) and walk the scored, multi-hop
      intellectual path between them — citations grounding the route.
    </div>
  </div>

  <div class="rg-feature-card">
    <div class="rg-feature-icon">▦</div>
    <div class="rg-feature-name">Insights & Rankings</div>
    <div class="rg-feature-desc">
      Identify hubs (degree), bridges (betweenness), clusters, and
      surprising similarity-only links a citation count can't see.
    </div>
  </div>

  <div class="rg-feature-card">
    <div class="rg-feature-icon">✦</div>
    <div class="rg-feature-name">Idea Generator</div>
    <div class="rg-feature-desc">
      Turn graph structure — bridges, gaps, surprising pairs — into
      concrete, citation-grounded research directions.
    </div>
  </div>

  <div class="rg-feature-card">
    <div class="rg-feature-icon">?</div>
    <div class="rg-feature-name">User Guide</div>
    <div class="rg-feature-desc">
      A full walkthrough of every page, control, badge, icon, and
      legend mark used across the app. New here? Start there.
    </div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown(
        '<div class="rg-section-title">Get started</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
1. Pick a dataset and similarity threshold from the **left sidebar**.
2. Open **Graph Explorer** to anchor a paper and inspect its neighborhood.
3. Use **Research Trajectory** to walk between two papers or topics,
   then **Insights & Rankings** to find hubs and bridges.
4. New to the controls? See the **User Guide** at the bottom of the nav.
""",
    )


# ---------------------------------------------------------------------------
# SEARCH PAPERS
# ---------------------------------------------------------------------------
elif page == "Search Papers":
    st.markdown("## Search Papers")
    st.markdown(
        "Find papers in the knowledge graph by keyword. Unlike a database search, "
        "results here include each paper's **structural role** in the graph — "
        "how many research threads it connects, and through what types of links."
    )
    st.markdown("")

    if not _require_papers("Search Papers"):
        st.stop()

    query = st.text_input(
        "Search titles and abstracts",
        placeholder='Try "transformer", "diffusion", "graph", "attention"',
    )

    if query:
        results = rg.search_papers(query)
        if results:
            st.markdown(
                f"**{len(results)}** paper(s) match \"{query}\" — "
                f"sorted by citation count to surface the most established work first."
            )
        else:
            st.markdown(f"**0** results for \"{query}\".")
        st.markdown("")

        for paper in sorted(results, key=lambda p: p.citation_count, reverse=True):
            nbrs = rg.get_neighbors(paper.paper_id)
            cite_n = sum(1 for n in nbrs if n["edge_type"] in ("citation", "both"))
            sim_n = sum(1 for n in nbrs if n["edge_type"] in ("similarity", "both"))

            with st.expander(f"{paper.title} ({paper.year})"):
                col_main, col_stats = st.columns([3, 1])
                with col_main:
                    _paper_card(paper)
                    if paper.url:
                        st.markdown(f"[Open paper]({paper.url})")
                with col_stats:
                    st.metric("Graph Connections", len(nbrs))
                    st.caption(f"{cite_n} citation / {sim_n} similarity")
                    st.metric("Citations", f"{paper.citation_count:,}")
                if nbrs:
                    paper_topics = ", ".join(paper.topic_words(3)) or "its area"
                    st.caption(
                        f"This paper connects to **{cite_n}** paper(s) through citations "
                        f"and **{sim_n}** through text similarity, "
                        f"positioning it within the {paper_topics} research cluster."
                    )

# ---------------------------------------------------------------------------
# GRAPH EXPLORER
# ---------------------------------------------------------------------------
elif page == "Graph Explorer":
    st.markdown(
        '<div class="rg-page-title">Graph Explorer</div>'
        '<p class="rg-page-subtitle">'
        'Pick a paper to anchor the graph. '
        '<b>Hover</b> any node to spotlight its connections, '
        '<b>click</b> to load it into the side panel, then use '
        '<b>Set as focus</b> to recenter.'
        '</p>',
        unsafe_allow_html=True,
    )

    if not _require_papers("Graph Explorer"):
        st.stop()

    # ---- Session state: focus + history ----
    default_id = paper_titles[sorted_titles[0]]
    if (st.session_state.get("explorer_focus_id") not in rg.papers):
        st.session_state["explorer_focus_id"] = default_id
    st.session_state.setdefault("explorer_history", [])

    # ---- Handle focus_swap query param fired by the in-canvas overlay ----
    swap_target = st.query_params.get("focus_swap")
    if swap_target and swap_target in rg.papers:
        if swap_target != st.session_state["explorer_focus_id"]:
            st.session_state["explorer_history"].append(
                st.session_state["explorer_focus_id"]
            )
            st.session_state["explorer_focus_id"] = swap_target
        # Always clear so the swap fires only once per click.
        del st.query_params["focus_swap"]

    # ---- Top control bar ----
    col_sel, col_rad, col_back = st.columns([3, 1, 1])
    with col_sel:
        focus_title = rg.papers[st.session_state["explorer_focus_id"]].title
        try:
            idx_default = sorted_titles.index(focus_title)
        except ValueError:
            idx_default = 0
        new_title = st.selectbox(
            "Focus paper",
            sorted_titles,
            index=idx_default,
            key="explorer_focus_select",
        )
        new_id = paper_titles[new_title]
        if new_id != st.session_state["explorer_focus_id"]:
            st.session_state["explorer_history"].append(
                st.session_state["explorer_focus_id"]
            )
            st.session_state["explorer_focus_id"] = new_id
    with col_rad:
        radius = st.selectbox("Radius (hops)", [1, 2, 3], index=0)
    with col_back:
        st.markdown('<div style="height:1.85rem"></div>', unsafe_allow_html=True)
        back_disabled = not st.session_state["explorer_history"]
        if st.button("← Back", disabled=back_disabled, use_container_width=True):
            prev = st.session_state["explorer_history"].pop()
            st.session_state["explorer_focus_id"] = prev
            st.rerun()

    focus_id = st.session_state["explorer_focus_id"]
    focus_paper = rg.papers[focus_id]

    render_graph_legend(kind="neighborhood")

    sub = rg.extract_subgraph(focus_id, radius=radius)
    st.markdown(
        f"#### Neighborhood of *{focus_paper.title}* — "
        f"{sub.number_of_nodes()} papers, {sub.number_of_edges()} edges"
    )
    max_nodes_for_radius = 60 if radius == 1 else 150
    if radius >= 2 and sub.number_of_nodes() > max_nodes_for_radius:
        st.caption(
            f"Showing top {max_nodes_for_radius} most-relevant nodes."
        )
    html = render_neighborhood(
        rg, focus_id, radius=radius, max_nodes=max_nodes_for_radius,
    )
    # Wrap the iframe in the .rg-workspace card so the canvas reads as a
    # designed workspace rather than a raw embed.
    st.markdown('<div class="rg-workspace">', unsafe_allow_html=True)
    components.html(html, height=720, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if not rg.get_neighbors(focus_id):
        st.warning(
            "This paper has no connections in the current graph. "
            "Try lowering the similarity threshold to reveal latent "
            "relationships."
        )

# ---------------------------------------------------------------------------
# RESEARCH TRAJECTORY
# ---------------------------------------------------------------------------
elif page == "Research Trajectory":
    st.markdown("## Research Trajectory")
    st.markdown(
        "Discover how ideas evolve across the knowledge graph. Each trajectory "
        "is a **scored, multi-hop path** ranked by semantic coherence and citation "
        "grounding — revealing the intellectual stepping stones between two "
        "research points that no keyword search could uncover."
    )

    if not _require_papers("Research Trajectory"):
        st.stop()

    tab_paper, tab_topic = st.tabs(["Between Two Papers", "Between Two Topics"])

    # ---- Tab 1: paper-to-paper ----
    with tab_paper:
        st.markdown(
            "Select two papers to trace how ideas flow between them through "
            "the knowledge graph. The trajectories reveal intermediate papers "
            "that carry concepts from one research context to another."
        )
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            source_title = st.selectbox("From", sorted_titles, key="traj_src")
        with col2:
            target_title = st.selectbox(
                "To", sorted_titles,
                index=min(1, len(sorted_titles) - 1), key="traj_tgt",
            )

        col_k, col_btn = st.columns([1, 2])
        with col_k:
            top_k = st.selectbox("Max paths", [1, 2, 3, 5], index=2)
        with col_btn:
            st.markdown("")  # spacing
            find_btn = st.button("Find Trajectories", type="primary", key="find_paper")

        if find_btn:
            src_id = paper_titles[source_title]
            tgt_id = paper_titles[target_title]

            if src_id == tgt_id:
                st.warning("Select two different papers.")
            else:
                with st.spinner("Searching and scoring paths..."):
                    paths = rg.find_meaningful_paths(src_id, tgt_id, top_k=top_k)

                if paths:
                    st.markdown("---")
                    st.markdown(
                        f"#### {len(paths)} trajectory(s) discovered — "
                        f"ranked by semantic coherence and citation grounding"
                    )
                    for i, sp in enumerate(paths, 1):
                        _render_path_result(sp, i, expanded=(i == 1))
                else:
                    st.error(
                        "No connecting trajectory found. These papers may occupy "
                        "disconnected regions of the knowledge graph. Try lowering "
                        "the similarity threshold to create additional edges."
                    )

    # ---- Tab 2: topic-to-topic (learning path) ----
    with tab_topic:
        st.markdown(
            "Enter two research topics to discover how one field connects to "
            "another through the knowledge graph. The trajectories reveal the "
            "conceptual stepping stones bridging distinct research areas."
        )
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            topic_a = st.text_input(
                "From topic",
                placeholder='e.g. "transformer", "graph neural"',
                key="lp_a",
            )
        with col2:
            topic_b = st.text_input(
                "To topic",
                placeholder='e.g. "diffusion", "image generation"',
                key="lp_b",
            )

        find_topic_btn = st.button("Find Trajectories", type="primary", key="find_topic")

        if find_topic_btn:
            if not topic_a or not topic_b:
                st.warning("Enter both topics.")
            elif topic_a.strip().lower() == topic_b.strip().lower():
                st.warning("Topics should be different.")
            else:
                with st.spinner("Searching cross-topic paths..."):
                    paths = rg.learning_path(topic_a, topic_b)

                if paths:
                    st.markdown("---")
                    st.markdown(
                        f'#### {len(paths)} trajectory(s) bridging '
                        f'"{topic_a}" to "{topic_b}" — '
                        f'tracing how concepts evolve across fields'
                    )
                    for i, sp in enumerate(paths, 1):
                        _render_path_result(sp, i, expanded=(i == 1))
                else:
                    matches_a = rg.search_papers(topic_a)
                    matches_b = rg.search_papers(topic_b)
                    if not matches_a:
                        st.error(f'No papers match "{topic_a}". Try a different keyword.')
                    elif not matches_b:
                        st.error(f'No papers match "{topic_b}". Try a different keyword.')
                    else:
                        st.error(
                            "No connecting path found. Try lowering the "
                            "similarity threshold to create more edges."
                        )

# ---------------------------------------------------------------------------
# INSIGHTS & RANKINGS
# ---------------------------------------------------------------------------
elif page == "Insights & Rankings":
    st.markdown("## Structural Insights")
    st.markdown(
        "These insights are derived from **graph topology**, not citation counts. "
        "They reveal structural roles — hubs, bridges, clusters, and hidden "
        "connections — that traditional bibliometric analysis cannot detect."
    )

    if not _require_papers("Insights & Rankings"):
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Hub Papers", "Bridge Papers", "Bridge Spotlight",
        "Cluster Insights", "Surprising Connections",
    ])

    with tab1:
        st.markdown("#### Hub Papers — The Most Influential Connectors")
        st.caption(
            "These papers have the highest **degree centrality**: the most "
            "combined citation and similarity edges. Each hub is a focal point "
            "where multiple research threads converge — removing one would "
            "leave many papers disconnected from each other."
        )
        st.markdown("")
        hub_papers = rg.rank_by_degree(10)
        if hub_papers:
            top_hub = hub_papers[0][0]
            top_hub_topics = ", ".join(top_hub.topic_words(3)) or "its core ideas"
            st.info(
                f"The graph's most connected paper is **{top_hub.title}** "
                f"({top_hub.year}), with **{hub_papers[0][1]} edges**. "
                f"Its work on {top_hub_topics} serves as a reference point "
                f"for **{hub_papers[0][1]}** other papers in this research space."
            )
        for rank, (paper, deg) in enumerate(hub_papers, 1):
            paper_topics = ", ".join(paper.topic_words(3)) or "its area"
            with st.expander(
                f"{rank}. {paper.title} ({paper.year}) — {deg} connections"
            ):
                _paper_card(paper, show_abstract=True)
                nbrs = rg.get_neighbors(paper.paper_id)
                cite_n = sum(1 for n in nbrs if n["edge_type"] in ("citation", "both"))
                sim_n = sum(1 for n in nbrs if n["edge_type"] in ("similarity", "both"))
                st.markdown(
                    f"**Structural role:** This paper connects to **{cite_n} "
                    f"citation neighbor(s)** and **{sim_n} similarity "
                    f"neighbor(s)**, making it a convergence point for research "
                    f"on {paper_topics}."
                )

    with tab2:
        st.markdown("#### Bridge Papers — Connectors Between Research Communities")
        st.caption(
            "These papers have the highest **betweenness centrality**: they "
            "sit on the shortest paths between many other papers. Each bridge "
            "translates concepts across community boundaries — without it, "
            "entire research areas would be structurally isolated."
        )
        st.markdown("")
        bridge_papers = rg.rank_by_betweenness(10)
        for rank, (paper, score) in enumerate(bridge_papers, 1):
            role = rg.describe_bridge_role(paper.paper_id)
            with st.expander(
                f"{rank}. {paper.title} ({paper.year}) — betweenness {score:.4f}"
            ):
                _paper_card(paper, show_abstract=True)
                if role:
                    st.markdown("---")
                    st.markdown("**Structural Role Analysis**")
                    st.markdown(role["role_description"])
                    if role["fragmentation_risk"] > 0:
                        st.warning(
                            f"Removing this paper would split the graph into "
                            f"**{role['fragmentation_risk']} additional component(s)** — "
                            f"it is a critical structural bottleneck."
                        )

    with tab3:
        st.markdown("#### Bridge Paper Spotlight — The Graph's Keystone")
        st.caption(
            "A deep dive into the single most important bridge paper — "
            "the one whose removal would most disrupt the flow of ideas "
            "between research communities."
        )
        st.markdown("")

        spotlight = rg.bridge_paper_spotlight()
        if spotlight:
            bp = spotlight["paper"]
            bp_topics = ", ".join(bp.topic_words(3)) or "its concepts"
            col_info, col_stats = st.columns([3, 1])
            with col_info:
                _paper_card(bp)
            with col_stats:
                st.metric("Betweenness", f"{spotlight['betweenness']:.4f}")
                st.metric("Connections", spotlight["num_connections"])

            # Structural role narrative
            role = rg.describe_bridge_role(bp.paper_id)
            if role:
                st.markdown("---")
                st.markdown("**Why This Paper Is the Graph's Keystone**")
                st.markdown(role["role_description"])

            if spotlight["bridged_pairs"]:
                st.markdown("---")
                st.markdown(
                    "**Communities this paper bridges** — without **"
                    f"{bp.title}**, these paper pairs would lose their "
                    f"only connection through the knowledge graph:"
                )
                for p1, p2 in spotlight["bridged_pairs"]:
                    p1_topics = ", ".join(p1.topic_words(2)) or "its area"
                    p2_topics = ", ".join(p2.topic_words(2)) or "its area"
                    with st.expander(
                        f"{p1.title}  <->  {p2.title}"
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            _paper_card(p1, show_abstract=False)
                        with c2:
                            _paper_card(p2, show_abstract=False)
                        st.markdown(
                            f"*{bp.title}* bridges {p1_topics} "
                            f"(from *{p1.title}*) to {p2_topics} "
                            f"(from *{p2.title}*) — a connection that exists "
                            f"only because of this keystone paper."
                        )

                # Graph visualization of the bridge
                st.markdown("---")
                st.markdown("**Bridge Neighborhood in the Graph**")
                st.caption(
                    "Hover any node for full metadata. The bridge paper is "
                    "highlighted in blue."
                )
                render_graph_legend(kind="neighborhood")
                html = render_neighborhood(rg, bp.paper_id, radius=1)
                components.html(html, height=600, scrolling=False)
        else:
            st.info("Graph is too small or dense for a meaningful bridge spotlight.")

    with tab4:
        st.markdown("#### Cluster Insights — Detecting Research Subfields")
        st.caption(
            "The graph's community structure reveals natural groupings of "
            "papers — each cluster represents a **coherent research subfield** "
            "where papers are more densely connected to each other than to the "
            "rest of the graph."
        )
        st.markdown("")

        clusters = rg.detect_clusters(min_size=2)
        if clusters:
            st.info(
                f"The knowledge graph contains **{len(clusters)} distinct "
                f"research cluster(s)**, identified through community detection "
                f"on the combined citation + similarity network."
            )
            for idx, cl in enumerate(clusters, 1):
                themes = ", ".join(cl["theme_words"][:4]) if cl["theme_words"] else "mixed topics"
                yr = cl["year_range"]
                year_str = f"{yr[0]}–{yr[1]}" if yr[0] and yr[1] else "unknown span"
                with st.expander(
                    f"Cluster {idx}: {themes} "
                    f"({len(cl['papers'])} papers, {year_str})"
                ):
                    st.markdown(
                        f"**This cluster represents a subfield focused on "
                        f"{themes}.** It contains **{len(cl['papers'])} papers** "
                        f"spanning **{year_str}** with a combined "
                        f"**{cl['total_citations']:,} citations**."
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Papers", len(cl["papers"]))
                    c2.metric("Internal Density", f"{cl['internal_density']:.3f}")
                    c3.metric("Total Citations", f"{cl['total_citations']:,}")

                    if cl["internal_density"] > 0.5:
                        st.success(
                            "High internal density indicates a **tightly-knit "
                            "research community** where papers heavily reference "
                            "and resemble each other."
                        )
                    elif cl["internal_density"] > 0.2:
                        st.info(
                            "Moderate density suggests a **loosely connected subfield** "
                            "with shared themes but diverse methodologies."
                        )

                    st.markdown("**Papers in this cluster:**")
                    for p in cl["papers"]:
                        st.markdown(
                            f"- **{p.title}** ({p.year or '?'}) — "
                            f"{p.citation_count:,} citations"
                        )

                    # Cluster summary from IdeaEngine
                    summary = engine.summarize_cluster(cl["papers"])
                    st.markdown("---")
                    st.markdown(summary)
        else:
            st.info(
                "No distinct clusters detected. The graph may be too small "
                "or too uniformly connected to exhibit clear community structure."
            )

    with tab5:
        st.markdown("#### Surprising Connections — Unexpected Bridges Between Fields")
        st.caption(
            "These paper pairs have **high text similarity but no citation link** — "
            "a connection that only the graph's similarity layer reveals. Each one "
            "represents a potential case of **independent convergence**, "
            "**missed citations**, or **under-explored cross-pollination** between fields."
        )
        st.markdown("")

        surprises = rg.find_surprising_connections(10)
        if surprises:
            st.info(
                f"Found **{len(surprises)} unexpected connection(s)** — "
                f"paper pairs that are textually similar but never cite each other. "
                f"These hidden links are invisible to traditional citation analysis."
            )
            for rank, (p1, p2, sim) in enumerate(surprises, 1):
                p1_topics = ", ".join(p1.topic_words(2)) or "its area"
                p2_topics = ", ".join(p2.topic_words(2)) or "its area"
                with st.expander(
                    f"{rank}. {p1.title}  <->  {p2.title}  "
                    f"(similarity {sim:.3f})"
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        _paper_card(p1, show_abstract=True)
                    with c2:
                        _paper_card(p2, show_abstract=True)

                    st.markdown("---")
                    st.markdown("**Why This Connection Is Surprising**")
                    description = rg.describe_surprising_connection(p1, p2, sim)
                    st.markdown(description)

                    shared = set(p1.topic_words(8)) & set(p2.topic_words(8))
                    unique_1 = set(p1.topic_words(5)) - shared
                    unique_2 = set(p2.topic_words(5)) - shared
                    if shared:
                        st.markdown(
                            f"**Converging themes:** {', '.join(shared)} &nbsp;|&nbsp; "
                            f"**Unique to first:** {', '.join(unique_1) or 'none'} &nbsp;|&nbsp; "
                            f"**Unique to second:** {', '.join(unique_2) or 'none'}"
                        )
        else:
            st.info(
                "No surprising connections at this threshold. Try lowering the "
                "similarity threshold to discover hidden links between papers."
            )

# ---------------------------------------------------------------------------
# IDEA GENERATOR
# ---------------------------------------------------------------------------
elif page == "Idea Generator":
    st.markdown("## Idea Generator")
    st.markdown(
        "Generate research-direction suggestions seeded by a paper or a topic. "
        "Each idea is grounded in a real graph relationship — the engine never "
        "invents pairings the graph does not already support."
    )

    if not _require_papers("Idea Generator"):
        st.stop()

    mode_paper, mode_topic, mode_bridge = st.tabs(
        ["From a paper", "From a topic", "Bridge two topics"]
    )

    with mode_paper:
        st.markdown(
            "Pick a paper. The engine looks at its highest-scoring related "
            "papers (combining citation, similarity, shared authors, and "
            "shared topic words) and proposes cross-pollination ideas."
        )
        anchor_title = st.selectbox(
            "Anchor paper", sorted_titles, key="idea_anchor",
        )
        n_ideas_p = st.slider("Number of ideas", 1, 5, 3, key="idea_n_paper")
        if st.button("Generate ideas", key="idea_btn_paper"):
            anchor_id = paper_titles[anchor_title]
            anchor = rg.get_paper(anchor_id)
            if anchor is None:
                st.error("Paper not found.")
            else:
                related_records = rg.get_related_papers(anchor_id, top_k=n_ideas_p * 2)
                related_papers = [r["paper"] for r in related_records[:n_ideas_p * 2]]
                ideas = engine.generate_ideas_from_paper(
                    anchor, related_papers, n_ideas=n_ideas_p,
                )
                st.markdown("### Suggested directions")
                for i, idea in enumerate(ideas, 1):
                    st.markdown(f"**{i}.** {idea}")

                if related_records:
                    st.markdown("---")
                    st.markdown("### Why these papers seeded the ideas")
                    for r in related_records[:n_ideas_p]:
                        with st.expander(
                            f"{r['paper'].title} (score {r['score']})"
                        ):
                            st.caption(
                                f"{r['paper'].year or '?'} · "
                                f"{r['paper'].venue or 'N/A'} · "
                                f"{r['paper'].citation_count:,} citations"
                            )
                            st.markdown("**Reasons related:**")
                            for reason in r["reasons"]:
                                st.markdown(f"- {reason}")

    with mode_topic:
        st.markdown(
            "Type a topic keyword. The engine surfaces the most-cited "
            "papers in that area and suggests follow-up directions."
        )
        topic_q = st.text_input(
            "Topic keyword",
            placeholder='e.g. "diffusion", "transformer"',
            key="idea_topic_q",
        )
        n_ideas_t = st.slider("Number of ideas", 1, 5, 3, key="idea_n_topic")
        if st.button("Generate ideas", key="idea_btn_topic"):
            if not topic_q.strip():
                st.warning("Enter a topic keyword.")
            else:
                matches = rg.search_papers(topic_q)
                if not matches:
                    st.error(f'No papers match "{topic_q}".')
                else:
                    ideas = engine.generate_ideas_from_topic(
                        topic_q, matches, n_ideas=n_ideas_t,
                    )
                    st.markdown(f"### Ideas seeded by {len(matches)} matching paper(s)")
                    for i, idea in enumerate(ideas, 1):
                        st.markdown(f"**{i}.** {idea}")

    with mode_bridge:
        st.markdown(
            "Enter two topics. The engine finds papers that connect them "
            "in the graph and proposes a bridge research study."
        )
        col1, col2 = st.columns(2)
        with col1:
            t_a = st.text_input("Topic A", key="idea_bridge_a")
        with col2:
            t_b = st.text_input("Topic B", key="idea_bridge_b")
        if st.button("Suggest bridge research", key="idea_btn_bridge"):
            if not t_a.strip() or not t_b.strip():
                st.warning("Enter both topics.")
            elif t_a.strip().lower() == t_b.strip().lower():
                st.warning("Topics must be different.")
            else:
                st.markdown(engine.suggest_bridge_research(t_a, t_b))
                bridges = rg.find_bridge_papers(t_a, t_b, top_k=5)
                if bridges:
                    st.markdown(
                        f"### {len(bridges)} bridge paper(s) in the graph"
                    )
                    for b in bridges:
                        bp = b["paper"]
                        with st.expander(
                            f"{bp.title} (bridge score {b['score']})"
                        ):
                            st.caption(
                                f"{bp.year or '?'} · {bp.venue or 'N/A'} · "
                                f"{bp.citation_count:,} citations"
                            )
                            st.markdown(
                                f"Connects **{len(b['topic_a_neighbors'])}** "
                                f"paper(s) on *{t_a}* with "
                                f"**{len(b['topic_b_neighbors'])}** paper(s) "
                                f"on *{t_b}*."
                            )
                else:
                    st.info(
                        "No bridge papers found at the current similarity "
                        "threshold. Try lowering it to surface more edges."
                    )


# ---------------------------------------------------------------------------
# USER GUIDE
# ---------------------------------------------------------------------------
elif page == "User Guide":
    st.markdown(
        '<div class="rg-page-title">User Guide</div>'
        '<p class="rg-page-subtitle">'
        'Every page, control, status pill, icon, and legend mark in '
        'ResearchGraph — what it does, when to use it, and what to expect. '
        'Skim the table of contents to jump to a specific area.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ----- Table of contents -------------------------------------------
    st.markdown(
        """
<div class="rg-guide">
<div class="rg-toc">
  <a href="#guide-shell">Sidebar &amp; header</a>
  <a href="#guide-search">Search Papers</a>
  <a href="#guide-explorer">Graph Explorer</a>
  <a href="#guide-trajectory">Research Trajectory</a>
  <a href="#guide-insights">Insights &amp; Rankings</a>
  <a href="#guide-idea">Idea Generator</a>
</div>
""",
        unsafe_allow_html=True,
    )

    # ----- Section: shell (sidebar + top header) -----------------------
    st.markdown(
        """
<section id="guide-shell" class="rg-guide-section">
  <h2><span class="rg-guide-icon">⚙</span> App shell — sidebar &amp; header</h2>
  <p class="rg-guide-lede">
    The chrome wrapping every page. The left sidebar controls the dataset
    you're working with; the top strip shows live graph stats and which
    page is active.
  </p>

  <h3>Left sidebar</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Dataset</span>
      <span>
        Switch between bundled and live-fetched paper sets.
        <code>Sample Dataset (bundled)</code> uses the curated 70-paper
        JSON shipped with the repo (works offline). The other entries
        (<code>Transformers</code>, <code>Diffusion Models</code>,
        <code>Graph Neural Networks</code>) hit the Semantic Scholar API
        for up to 150 papers per query, with disk caching.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Similarity threshold</span>
      <span>
        TF-IDF cosine cutoff for the similarity edges. Lower values
        (~0.05) reveal more latent connections at the cost of noisier
        graphs; higher values (~0.30+) keep only the strongest semantic
        links. Citation edges are unaffected.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Graph stats line</span>
      <span>
        <b>papers · edges · component(s)</b> — total nodes, total edges
        (citation + similarity combined), and how many disconnected
        sub-graphs exist. The caption underneath splits edges by kind
        and shows graph density.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Navigate</span>
      <span>
        Page selector. Changing pages preserves your dataset choice and
        any focus state in <b>Graph Explorer</b>.
      </span>
    </li>
  </ul>

  <h3>Top header strip</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">RG monogram + tagline</span>
      <span>
        Brand block — same on every page. Click nothing here; it's just
        the product mark.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Right-side meta</span>
      <span>
        Live counts of <b>papers</b>, <b>edges</b>, and graph
        <b>density</b>, plus the name of the active page. These
        refresh whenever you change dataset or similarity threshold.
      </span>
    </li>
  </ul>
</section>
""",
        unsafe_allow_html=True,
    )

    # ----- Section: Search Papers --------------------------------------
    st.markdown(
        """
<section id="guide-search" class="rg-guide-section">
  <h2><span class="rg-guide-icon">⌕</span> Search Papers</h2>
  <p class="rg-guide-lede">
    Find papers in the active dataset by keyword and see each result
    annotated with its <b>structural role</b> in the graph — not just
    how often it's cited, but how many research threads it connects
    through citation vs. similarity links.
  </p>

  <h3>How the search works</h3>
  <p>
    The search box matches your keyword against each paper's
    <b>title</b> and <b>abstract</b> (case-insensitive substring match).
    Results are sorted by <b>citation count</b>, descending, so the most
    established work surfaces first. There's no fuzzy matching — type
    the term as it appears in the literature
    (<code>"transformer"</code>, <code>"diffusion"</code>,
    <code>"graph neural"</code>, …).
  </p>

  <h3>Reading a result row</h3>
  <p>
    Each match is rendered as a collapsed expander labelled
    <code>Title (year)</code>. Click to open it. The body has two
    columns:
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Left column · paper card</span>
      <span>
        The card displays venue, citation count, and the first few
        author names. If the paper has an <b>Open paper</b> link, the
        URL appears beneath the card. The abstract snippet is rendered
        inline (capped at ~250 chars).
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Right column · stats</span>
      <span>
        Two metrics: <b>Graph Connections</b> (total neighbors of this
        paper in the current graph) and <b>Citations</b> (raw citation
        count from the source). The caption underneath splits the
        connections by kind: <code>N citation / M similarity</code>.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Bottom caption</span>
      <span>
        A one-sentence summary positioning the paper inside its
        research cluster, derived from its top topic words plus the
        edge breakdown above.
      </span>
    </li>
  </ul>

  <h3>What to do next</h3>
  <p>
    Found a paper that interests you? Copy its title from the expander
    header, jump to <b>Graph Explorer</b> (or <b>Research Trajectory</b>)
    in the sidebar, and pick it from the focus dropdown there. Search
    intentionally does not jump for you — you may want to compare
    several results before committing one to the graph view.
  </p>

  <h3>Empty state</h3>
  <p>
    Zero results means the keyword doesn't appear in any title or
    abstract in the active dataset. Switch dataset (top-left sidebar)
    or try a broader term — keywords are case-insensitive but exact:
    <code>"transformer"</code> matches but <code>"transfromers"</code>
    will not.
  </p>
</section>
""",
        unsafe_allow_html=True,
    )

    # ----- Section: Graph Explorer -------------------------------------
    st.markdown(
        """
<section id="guide-explorer" class="rg-guide-section">
  <h2><span class="rg-guide-icon">◉</span> Graph Explorer</h2>
  <p class="rg-guide-lede">
    The interactive workspace. Anchor any paper at the center of the
    canvas, then hover, click, drag, and filter to see how the
    surrounding literature connects. Every node, edge, badge, and
    button on this page is documented below.
  </p>

  <h3>Top-of-page controls (Streamlit row)</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Focus paper</span>
      <span>
        Dropdown listing every paper in the active dataset, sorted
        alphabetically. Picking a new paper recenters the graph on it
        and pushes the previous focus onto the back-stack.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Radius (hops)</span>
      <span>
        How far out from the focus to walk when collecting nodes:
        <code>1</code> shows direct neighbors only,
        <code>2</code> adds 2nd-hop context nodes (rendered as small
        light-gray dots), <code>3</code> goes one further. Larger
        radii are capped at 60 / 150 nodes for legibility.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">← Back</span>
      <span>
        Pop the most recent previous focus off the back-stack. Greyed
        out when the stack is empty.
      </span>
    </li>
  </ul>

  <h3>Canvas controls bar (top of the iframe)</h3>
  <p class="rg-guide-lede">
    Five compact pill buttons above the graph. Each is a soft white
    chip; the two edge-filter buttons show an <b>active</b> state
    (light-blue tint) when their edge type is visible.
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
        </span>
        Fit
      </span>
      <span>
        Animate the camera to fit every visible node into view. Useful
        after dragging nodes far apart or after zooming in on a
        neighbor.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 8a5 5 0 1 0 1.5-3.5"/><path d="M3 3v3h3"/></svg>
        </span>
        Reset
      </span>
      <span>
        Clear the sticky selection (if any), restore every edge's
        original color, repaint the sidebar to show the focus paper,
        and re-fit the camera. Equivalent to "go back to the resting
        state".
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 4h10M3 8h10M3 12h7"/></svg>
        </span>
        Labels
      </span>
      <span>
        Toggle short titles on every neighbor / context node. Off by
        default — only the focus paper is labeled — to keep the canvas
        legible. Click the button to reveal labels (it turns
        light-blue), click again to hide.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 8h12"/></svg>
        </span>
        Citation
      </span>
      <span>
        Filter toggle. Active by default — citation edges (and
        citation+similarity "both" edges) are visible. Click to hide
        them; click again to bring them back.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="2 2"><path d="M2 8h12"/></svg>
        </span>
        Similarity
      </span>
      <span>
        Filter toggle. Active by default — semantic-similarity edges
        (the dashed ones) are visible. Toggle off to focus on
        citation-only structure.
      </span>
    </li>
  </ul>

  <h3>Detail sidebar (right side of the canvas)</h3>
  <p class="rg-guide-lede">
    Always visible. Shows the focus paper by default, swaps to the
    hovered paper on hover, and locks to the clicked paper on click
    (via a sticky-selection state).
  </p>
  <h4 style="font-size:12.5px;color:var(--rg-text-muted);font-weight:600;margin:10px 0 4px 0;letter-spacing:0.04em;text-transform:uppercase;">
    Status pill (top of sidebar)
  </h4>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-pill"><span class="rg-guide-pill-dot"></span>Current focus</span>
      </span>
      <span>
        Blue. The sidebar is showing the paper at the center of the
        graph — the anchor of this view. Default state with no hover
        or selection active.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-pill is-amber"><span class="rg-guide-pill-dot"></span>Hovered paper</span>
      </span>
      <span>
        Amber. You're hovering a non-focus node. Sidebar contents
        reflect that node transiently — when you move the mouse away
        the sidebar reverts to the focus.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-pill is-green"><span class="rg-guide-pill-dot"></span>Selected paper</span>
      </span>
      <span>
        Green. You clicked this node. The sidebar is locked to it
        through any subsequent hover; click empty space (or hit
        <b>Reset</b>) to release the lock.
      </span>
    </li>
  </ul>

  <h4 style="font-size:12.5px;color:var(--rg-text-muted);font-weight:600;margin:10px 0 4px 0;letter-spacing:0.04em;text-transform:uppercase;">
    Body sections
  </h4>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Title</span>
      <span>16-pt semibold, the paper's full title.</span>
    </li>
    <li>
      <span class="rg-guide-key">Meta row</span>
      <span>
        <b>Year</b> · venue · citation count, separated by faint dots.
        Year is bold; venue and citation count are muted.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Authors</span>
      <span>
        First four author names (a "+ N more" suffix when there are
        more). Hidden entirely if the paper has no author data.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Topics</span>
      <span>
        Up to six topic-word badges (cool-gray pills) extracted from
        the title + abstract. Hidden when no topics could be derived.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Why connected</span>
      <span>
        Blue callout, only shown when the sidebar is on a non-focus
        paper that has a direct edge to the focus. Explains the edge
        kind: <code>direct citation link</code>,
        <code>text similarity 0.42</code>,
        <code>citation + similarity 0.31</code>, or
        <code>shared author</code>.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Abstract</span>
      <span>
        Full abstract, scrollable inside the sidebar. Hidden if the
        paper has no abstract on file.
      </span>
    </li>
  </ul>

  <h4 style="font-size:12.5px;color:var(--rg-text-muted);font-weight:600;margin:10px 0 4px 0;letter-spacing:0.04em;text-transform:uppercase;">
    Footer actions
  </h4>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="8" cy="8" r="3"/><circle cx="8" cy="8" r="6"/></svg>
        </span>
        Set as focus
      </span>
      <span>
        Primary blue button. Recenters the graph on the currently-shown
        paper, pushing the prior focus onto the back-stack. Auto-disabled
        and labelled <b>Already focus</b> when the sidebar is showing the
        current focus paper.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-guide-icon-sm">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 2h5v5"/><path d="M14 2 7 9"/><path d="M12 9v4a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h4"/></svg>
        </span>
        Open paper
      </span>
      <span>
        Secondary chip. Opens the paper's source URL in a new tab.
        Greyed out and inert if the paper has no URL on file.
      </span>
    </li>
  </ul>

  <h3>Legend (bottom-left of the canvas)</h3>
  <p class="rg-guide-lede">
    Compact frosted-glass strip explaining the visual encoding. Three
    color dots and two line samples.
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">
        <span class="rg-legend-dot" style="background:#2f5fb3;display:inline-block;width:10px;height:10px;border-radius:50%;border:1.5px solid #fff;box-shadow:0 0 0 1px var(--rg-border);"></span>
        focus
      </span>
      <span>
        The anchor paper. Largest node, blue with a soft blue glow and
        a white outline. Pinned at the center; not draggable.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-legend-dot" style="background:#5d8b56;display:inline-block;width:10px;height:10px;border-radius:50%;border:1.5px solid #fff;box-shadow:0 0 0 1px var(--rg-border);"></span>
        citation
      </span>
      <span>
        A direct neighbor where the connection is a citation (one paper
        references the other). Muted-sage fill, white outline.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-legend-dot" style="background:#c69457;display:inline-block;width:10px;height:10px;border-radius:50%;border:1.5px solid #fff;box-shadow:0 0 0 1px var(--rg-border);"></span>
        similarity
      </span>
      <span>
        A direct neighbor where the only connection is text similarity
        above the threshold (no citation). Muted-amber fill.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-edge-sample"><span class="rg-edge-sample-line"></span></span>
        cites
      </span>
      <span>
        Solid edge — a citation link. Width is constant by default;
        the highlight pass slightly widens the connected edges.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span class="rg-edge-sample"><span class="rg-edge-sample-line is-dashed"></span></span>
        similar (text)
      </span>
      <span>
        Dashed edge — a TF-IDF cosine similarity that exceeds the
        sidebar threshold but with no underlying citation. Soft indigo.
      </span>
    </li>
  </ul>
  <p>
    Two extra node colors that don't appear in the legend strip but you
    will see in the graph: <b>muted indigo</b> for "both" neighbors
    (citation + similarity at once) and <b>soft lavender</b> for
    shared-author neighbors. Tiny <b>warm-gray</b> dots are 2nd-hop
    context nodes (only present at radius ≥ 2).
  </p>

  <h3>Interactions</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Hover a node</span>
      <span>
        Status pill turns amber, sidebar updates with that paper's
        metadata, the node grows by 6&nbsp;px, every edge that <i>isn't</i>
        connected to it fades to near-transparent grey, and every other
        non-connected node desaturates to a flat ghost color. The focus
        stays vivid throughout — it's the anchor.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Click a node</span>
      <span>
        Status pill turns green, the same dim/highlight pass applies but
        is now <b>sticky</b> through any subsequent hover, the camera
        animates to zoom in on the node (1.55× scale), and the sidebar
        locks to it. Click empty space (or hit <b>Reset</b>) to release.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Drag a node</span>
      <span>
        The physics simulator wakes up for the duration of the drag.
        Connected dynamic-smooth edges flex and curve as you pull;
        when you release, the simulator settles for ~800&nbsp;ms then
        freezes again so the canvas doesn't perpetually drift.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Pan / zoom canvas</span>
      <span>
        Click-drag empty space to pan. Mouse wheel to zoom. Hit
        <b>Fit</b> in the controls bar to recenter on everything visible.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Tooltip on hover</span>
      <span>
        A small tooltip with just the paper's <b>title</b>. Full
        metadata lives in the sidebar — the tooltip is intentionally
        minimal so the eye reads structure first.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Quick double-click</span>
      <span>
        Treated as two single-clicks on the same node — no special
        handler, no flicker. (An earlier double-click handler used to
        race the single-click and zoom-out the canvas; it has been
        removed.)
      </span>
    </li>
  </ul>

  <h3>Responsive layout</h3>
  <p>
    Below ~880&nbsp;px viewport width the sidebar collapses to a
    240&nbsp;px-tall <b>bottom drawer</b> and the canvas takes the full
    width above it. Pure CSS @media rule — no JS toggle, no extra
    button.
  </p>
</section>
""",
        unsafe_allow_html=True,
    )

    # ----- Section: Research Trajectory --------------------------------
    st.markdown(
        """
<section id="guide-trajectory" class="rg-guide-section">
  <h2><span class="rg-guide-icon">↝</span> Research Trajectory</h2>
  <p class="rg-guide-lede">
    Walk a scored, multi-hop path between two anchors — either two
    specific papers or two free-text topics — and read why the path
    is meaningful at each step. The graph treats this as a search
    problem and ranks candidate paths by semantic coherence and
    citation grounding.
  </p>

  <h3>Two modes (top tabs)</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Between Two Papers</span>
      <span>
        Pick a <b>From</b> and a <b>To</b> paper from the dropdowns,
        choose <b>Max paths</b> (1 / 2 / 3 / 5), and hit
        <b>Find Trajectories</b>. The graph enumerates simple paths up
        to length&nbsp;5 between the two endpoints, scores each, and
        returns the top&nbsp;k. Source must differ from target.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Between Two Topics</span>
      <span>
        Type two free-text keywords (e.g. <code>"transformer"</code>
        and <code>"diffusion"</code>). The engine picks up to eight
        candidate papers per topic, removes the overlap between them,
        and scores every cross-pair, returning the strongest
        trajectories. Topics must be non-empty and different.
      </span>
    </li>
  </ul>

  <h3>How a path is scored</h3>
  <p>
    Every candidate path is reduced to a single score, then a label:
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Avg. Similarity</span>
      <span>
        Mean TF-IDF cosine between consecutive papers along the path.
        High values mean each step shares topical vocabulary with the
        next — the path doesn't lurch between unrelated areas.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Citation Strength</span>
      <span>
        Fraction of the path's edges that are <i>citation</i> or
        <i>both</i> (vs. similarity-only). High values mean the route
        rides on real "X cites Y" links rather than only inferred
        textual closeness.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Short-path penalty</span>
      <span>
        1-hop paths are penalised by 0.3, 2-hop by 0.1, longer paths
        not at all — the goal is to reward paths that actually
        traverse intermediate ideas, not trivial direct neighbors.
      </span>
    </li>
  </ul>
  <p>
    The composite score is <code>avg_similarity + citation_strength
    − penalty</code>, surfacing as one of three quality badges:
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">
        <span style="background:#2e7d32;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;">Strong</span>
      </span>
      <span>score ≥ 0.80 — coherent, citation-grounded path.</span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span style="background:#e65100;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;">Moderate</span>
      </span>
      <span>0.40 ≤ score &lt; 0.80 — plausible but partially leaning
        on textual similarity.</span>
    </li>
    <li>
      <span class="rg-guide-key">
        <span style="background:#c62828;color:#fff;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600;">Weak</span>
      </span>
      <span>score &lt; 0.40 — the only route between the endpoints,
        worth treating with caution.</span>
    </li>
  </ul>

  <h3>Reading a trajectory card</h3>
  <p class="rg-guide-lede">
    Each result is a collapsible expander; the first one opens by
    default. Inside you get five blocks, in order:
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Header line</span>
      <span>
        <b>Trajectory N</b>, the colored quality badge, the numeric
        <b>score</b>, and the <b>hop count</b>.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Narrative summary</span>
      <span>
        One auto-generated sentence describing what the path reveals
        — e.g. "research on <i>X</i> evolves into <i>Y</i> through
        <i>bridging ideas</i>, a 4-hop journey across 6 years".
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Visual chain</span>
      <span>
        The papers laid out left-to-right with → arrows between them.
        Each entry shows title, citation count, and year.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Three score metrics</span>
      <span>
        <b>Avg. Similarity</b>, <b>Citation Strength</b> (rendered
        as a percentage), and <b>Path Length</b> (number of papers).
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Concept-transfer steps</span>
      <span>
        For every consecutive pair (A → B) on the path, a one-line
        description of which concepts moved from A to B — derived
        from each paper's topic words.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Why this trajectory matters</span>
      <span>
        Auto-composed reasoning that fires when applicable: a
        multi-year span (≥ 2 years between endpoints), a high
        citation-strength fraction (&gt; 50%), or strong topical
        coherence (avg. similarity &gt; 0.30).
      </span>
    </li>
  </ul>

  <h3>Empty / error states</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">"Select two different papers."</span>
      <span>You picked the same paper for both endpoints.</span>
    </li>
    <li>
      <span class="rg-guide-key">"No connecting trajectory found."</span>
      <span>
        The two endpoints sit in <b>disconnected components</b> of the
        graph. Lower the similarity threshold from the sidebar to
        stitch them together with more edges, or pick a different
        endpoint.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">"No papers match …"</span>
      <span>
        Topic-mode only. The keyword didn't hit any title or abstract.
        Try a broader term or switch dataset.
      </span>
    </li>
  </ul>

  <h3>Performance notes</h3>
  <p>
    Path enumeration is capped at <b>50 candidates per query</b> and a
    cutoff length of 5 hops, so results return in well under a second
    even on the 150-paper datasets. Topic mode internally runs the
    paper-mode search across up to <b>8 × 8 = 64</b> source-target
    pairs, dedupes by paper sequence, and keeps the top three.
  </p>
</section>
""",
        unsafe_allow_html=True,
    )

    # ----- Section: Insights & Rankings --------------------------------
    st.markdown(
        """
<section id="guide-insights" class="rg-guide-section">
  <h2><span class="rg-guide-icon">▦</span> Insights &amp; Rankings</h2>
  <p class="rg-guide-lede">
    Five tabs of structural analysis. Everything on this page is
    derived from <b>graph topology</b> — degree, betweenness, community
    detection, and similarity edges — not raw citation counts. The
    point is to surface roles that bibliometric ranking misses.
  </p>

  <h3>Tab 1 · Hub Papers</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">What it ranks</span>
      <span>
        Top 10 papers by <b>degree centrality</b> — total combined
        citation + similarity edges. The header banner highlights the
        single most-connected paper.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Why it matters</span>
      <span>
        Hubs are convergence points where multiple research threads
        meet. Removing one would leave many of its neighbors with
        nothing closer than a 2-hop relationship.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Reading a row</span>
      <span>
        Each entry expands to show the paper card, abstract, and a
        structural-role line: <code>connects to N citation neighbor(s)
        and M similarity neighbor(s)</code>. The expander header tells
        you the rank, title, year, and total connection count.
      </span>
    </li>
  </ul>

  <h3>Tab 2 · Bridge Papers</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">What it ranks</span>
      <span>
        Top 10 papers by <b>betweenness centrality</b> — how many
        shortest paths between other paper pairs run through this one.
        Score is shown to four decimals because betweenness values are
        typically small.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Why it matters</span>
      <span>
        Bridges translate concepts across community boundaries. They
        often have lower citation counts than hubs, so traditional
        ranking misses them entirely.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Structural Role Analysis</span>
      <span>
        Auto-generated narrative inside each expander explaining
        <i>which</i> communities the paper bridges, plus a fragmentation
        warning when removing the paper would split the graph into
        additional disconnected components.
      </span>
    </li>
  </ul>

  <h3>Tab 3 · Bridge Spotlight</h3>
  <p class="rg-guide-lede">
    A deep-dive on the single most important bridge paper — its
    metadata, betweenness score, and a complete inventory of which
    paper pairs depend on it for their only graph connection.
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Top metrics</span>
      <span>
        <b>Betweenness</b> (4-decimal centrality score) and
        <b>Connections</b> (total neighbors). Plus the paper card on
        the left.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Communities this paper bridges</span>
      <span>
        A list of expandable paper-pair entries — for each pair, the
        spotlight paper is the <i>only</i> hop between them. Removing
        the spotlight breaks that pair's connection. Each pair
        expander shows both papers' cards side by side and the
        bridging narrative.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Bridge Neighborhood</span>
      <span>
        A small embedded graph view showing the spotlight paper at
        the center surrounded by its 1-hop neighbors. Useful for a
        quick visual confirmation of the structural role.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Empty state</span>
      <span>
        "Graph is too small or dense for a meaningful bridge spotlight."
        Happens when no single paper has noticeably higher betweenness
        than the rest — try a different / larger dataset.
      </span>
    </li>
  </ul>

  <h3>Tab 4 · Cluster Insights</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">What it does</span>
      <span>
        Runs <b>greedy modularity community detection</b> on the
        combined citation + similarity network and returns every
        cluster of ≥ 2 papers.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Cluster header</span>
      <span>
        <code>Cluster N: theme1, theme2, … (P papers, year-range)</code>.
        Theme words are the most-frequent topic tokens across the
        cluster's papers; the year range covers earliest to latest.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Three metrics inside</span>
      <span>
        <b>Papers</b> in the cluster, <b>Internal Density</b>
        (fraction of possible edges that exist within the cluster
        sub-graph), and <b>Total Citations</b> summed across the
        cluster.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Density bands</span>
      <span>
        Density &gt; 0.50 → green callout: <i>tightly-knit
        community</i>. 0.20–0.50 → blue callout:
        <i>loosely connected subfield</i>. Below 0.20 → no callout.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Auto-summary</span>
      <span>
        At the end of every cluster, the IdeaEngine renders a
        few-sentence prose summary of the cluster's themes,
        composed from its top papers and topics.
      </span>
    </li>
  </ul>

  <h3>Tab 5 · Surprising Connections</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">What it surfaces</span>
      <span>
        Top 10 paper pairs with high TF-IDF cosine similarity but
        <b>no</b> direct citation between them. These are connections
        the graph's similarity layer reveals but bibliometric analysis
        cannot.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Why each pair is surprising</span>
      <span>
        Auto-generated reasoning that fires on whichever conditions
        apply: distant graph hops, mismatched venues (independent
        convergence across communities), large temporal gap (possible
        rediscovery), shared topic vocabulary, or zero overlap with
        latent semantic match.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Theme breakdown</span>
      <span>
        Three labels at the bottom of each pair: <b>Converging
        themes</b> (shared topic words), <b>Unique to first</b>, and
        <b>Unique to second</b>. Empty buckets render as "none".
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Empty state</span>
      <span>
        "No surprising connections at this threshold." Lower the
        similarity threshold from the sidebar to surface more
        cross-cluster latent links.
      </span>
    </li>
  </ul>
</section>
""",
        unsafe_allow_html=True,
    )

    # ----- Section: Idea Generator -------------------------------------
    st.markdown(
        """
<section id="guide-idea" class="rg-guide-section">
  <h2><span class="rg-guide-icon">✦</span> Idea Generator</h2>
  <p class="rg-guide-lede">
    Turn graph structure into concrete research-direction suggestions.
    Three modes — anchor on a paper, anchor on a topic, or bridge two
    topics. Every idea is grounded in a relationship the graph
    actually contains; the engine never invents a citation or a
    similarity that isn't there.
  </p>

  <h3>Mode 1 · From a paper</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Inputs</span>
      <span>
        An <b>Anchor paper</b> (dropdown — every paper in the active
        dataset) and a <b>Number of ideas</b> slider (1–5, default 3).
      </span>
    </li>
    <li>
      <span class="rg-guide-key">What it does</span>
      <span>
        Pulls the anchor's top related papers (combining citation
        edges, similarity edges, shared authors, and shared topic
        words into a composite score), then asks the IdeaEngine to
        write <i>n</i> cross-pollination directions seeded by those
        relationships.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Output</span>
      <span>
        Two blocks: a numbered list of ideas, then a "Why these papers
        seeded the ideas" block — for each seed paper an expander
        showing year/venue/citations and a bullet list of <b>Reasons
        related</b> (e.g. <i>"shared author Alice"</i>, <i>"text
        similarity 0.42"</i>, <i>"direct citation"</i>).
      </span>
    </li>
  </ul>

  <h3>Mode 2 · From a topic</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Inputs</span>
      <span>
        A <b>Topic keyword</b> (free text — same matching rules as
        Search Papers) and a <b>Number of ideas</b> slider (1–5).
      </span>
    </li>
    <li>
      <span class="rg-guide-key">What it does</span>
      <span>
        Surfaces papers matching the keyword, ranks them by citations,
        and asks the IdeaEngine to generate follow-up directions
        seeded by the most-cited matches.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Output</span>
      <span>
        Header tells you how many papers seeded the suggestions
        (<code>Ideas seeded by N matching paper(s)</code>), then a
        numbered list of directions.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Empty state</span>
      <span>
        "Enter a topic keyword." or "No papers match …" — switch
        dataset or broaden the term.
      </span>
    </li>
  </ul>

  <h3>Mode 3 · Bridge two topics</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Inputs</span>
      <span>
        Two free-text topics, <b>Topic A</b> and <b>Topic B</b>. Both
        must be non-empty and different.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">What it does</span>
      <span>
        Composes a bridge-research suggestion describing how a study
        could combine the two areas, then searches the graph for
        actual <b>bridge papers</b> — papers that already connect at
        least one paper from topic&nbsp;A's neighborhood with at least
        one from topic&nbsp;B's.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Output</span>
      <span>
        The synthesised proposal first, then up to five bridge-paper
        expanders. Each shows year/venue/citations and a one-liner
        about how many topic-A and topic-B neighbors that paper
        connects.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Empty state</span>
      <span>
        "No bridge papers found at the current similarity threshold."
        Lower the threshold from the sidebar — that adds more edges
        and often surfaces a previously-hidden bridge candidate.
      </span>
    </li>
  </ul>
</section>

<section class="rg-guide-section">
  <h2><span class="rg-guide-icon">⚒</span> Dataset &amp; threshold cookbook</h2>
  <p class="rg-guide-lede">
    A short field guide to the two sidebar controls that change every
    page in the app. Both decisions are reversible — change them, watch
    the header counts update live, and re-explore.
  </p>

  <h3>Choosing a dataset</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Sample Dataset (bundled)</span>
      <span>
        70 hand-curated papers shipped with the repo, no network
        required. Best for exploring the UI, demos, and offline use.
        Fully connected with internal references resolved.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Transformers / Diffusion / Graph Neural Networks</span>
      <span>
        Live Semantic Scholar fetches (up to 150 papers per topic,
        cached on disk after the first run). Better for serious
        exploration but the first load is slow and you need network
        access.
      </span>
    </li>
  </ul>

  <h3>Tuning the similarity threshold</h3>
  <p>
    The slider sets the TF-IDF cosine cutoff for similarity edges
    only. Citation edges are unaffected. Two heuristics:
  </p>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">Lower (≈ 0.05–0.12)</span>
      <span>
        Surfaces more latent connections. Useful for <b>Surprising
        Connections</b>, <b>Bridge Papers</b>, and any topic-mode
        search that's returning empty paths. Trade-off: visually
        denser graph and noisier hub rankings.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Default (0.15)</span>
      <span>
        Balanced. Good for first-pass exploration in
        <b>Graph Explorer</b> — connections are meaningful and the
        canvas stays legible.
      </span>
    </li>
    <li>
      <span class="rg-guide-key">Higher (≈ 0.25–0.40)</span>
      <span>
        Only the strongest semantic links. Useful when you want
        cluster detection to return very tight communities, or when
        a low threshold has flooded the canvas. Trade-off: paths
        between distant topics may stop resolving.
      </span>
    </li>
  </ul>

  <h3>Quick troubleshooting</h3>
  <ul class="rg-guide-list">
    <li>
      <span class="rg-guide-key">"No connecting trajectory found"</span>
      <span>Lower the similarity threshold, or pick endpoints from the
      same dataset / general subfield.</span>
    </li>
    <li>
      <span class="rg-guide-key">"No bridge papers found"</span>
      <span>Same fix — lower threshold widens the similarity layer
      and stitches more bridges into existence.</span>
    </li>
    <li>
      <span class="rg-guide-key">Canvas drifts or rotates</span>
      <span>It shouldn't — the simulator freezes after stabilization
      and only resumes during a drag. If you ever see it, drag any
      neighbor a tiny bit; that wakes the simulator and it
      re-stabilises.</span>
    </li>
    <li>
      <span class="rg-guide-key">Sidebar / labels feel cramped</span>
      <span>Below ≈ 880 px viewport width the right sidebar drops to
      a bottom drawer automatically. Widen the window to bring it
      back to the side.</span>
    </li>
  </ul>
</section>
""",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
