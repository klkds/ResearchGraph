"""Graph visualization — deterministic layouts for guaranteed clean output.

The previous version relied on physics + label-suppression heuristics to
keep the canvas readable.  In dense graphs that still resulted in
overlapping labels and jittery layouts.  This rewrite removes the
guesswork:

    - **Radial layout** for neighborhood graphs: the focus paper is
      placed at (0, 0) and direct neighbors are pinned on a circle
      around it.  Second-hop / context nodes go on a wider outer ring.
    - **Linear layout** for paths: path papers sit on a horizontal
      line, evenly spaced.  Optional context nodes sit on a faint
      arc above and below.
    - **Physics is OFF** in the rendered network.  Positions are
      fixed (``fixed: true``) and pyvis does no layout work — the
      output looks identical on every load.
    - **Labels are OFF by default** for every node except the focus
      / path.  All paper metadata still lives in the rich hover
      tooltip.  This eliminates label overlap entirely.
    - **No in-canvas legend nodes**.  The legend is rendered by
      Streamlit above the graph (``neighborhood_legend_html`` /
      ``path_legend_html``).
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from pyvis.network import Network

if TYPE_CHECKING:
    from src.paper import Paper
    from src.research_graph import ResearchGraph


# ---------------------------------------------------------------------------
# Color palette — calm, low-saturation tones for a research-dashboard feel
# ---------------------------------------------------------------------------
COLOR_FOCUS = "#2f5fb3"          # the brand blue — the focus is the anchor
COLOR_CITATION = "#5d8b56"       # muted sage (citation neighbor)
COLOR_SIMILARITY = "#c69457"     # muted amber (similarity-only neighbor)
COLOR_BOTH = "#6a5fa3"           # muted indigo (citation + similarity)
COLOR_PATH = "#b04a4a"           # muted brick (path node)
COLOR_CONTEXT = "#d6dde4"        # warm light gray (2nd-hop / context)
COLOR_SHARED_AUTHOR = "#9988c1"  # soft lavender (co-author neighbor)

EDGE_CITATION = "#7d8b96"        # cool gray (default citation edge)
EDGE_SIMILARITY = "#9ea7c9"      # soft indigo (similarity edge, dashed)
EDGE_BOTH = "#5b6878"            # darker gray for both-type edges
EDGE_PATH = "#b04a4a"            # muted brick
EDGE_CONTEXT = "#e6eaef"         # very light gray

# ---------------------------------------------------------------------------
# Layout constants (all in pyvis pixel coordinates)
# ---------------------------------------------------------------------------
INNER_RADIUS = 220   # direct-neighbor ring radius
OUTER_RADIUS = 380   # second-hop ring radius
PATH_SPACING_X = 200 # horizontal spacing for path nodes
PATH_CONTEXT_Y = 130 # vertical offset for path context arcs


# ---------------------------------------------------------------------------
# Pyvis options — barnesHut force-directed layout with stabilization,
# combined with dynamic-smooth edges. Nodes wiggle into a stable
# arrangement on load (you can see them settle), then any drag pulls
# connected nodes through spring/gravity forces and the edges flex
# with the motion. Focus / path nodes are pinned at hand-computed
# coordinates so the centerpiece stays anchored; everything else
# floats and is fully draggable.
# ---------------------------------------------------------------------------
_PYVIS_OPTIONS = """
{
  "nodes": {
    "shape": "dot",
    "borderWidth": 1.5,
    "borderWidthSelected": 4,
    "shadow": false,
    "font": {
      "size": 12,
      "face": "Arial",
      "strokeWidth": 4,
      "strokeColor": "#ffffff",
      "color": "#1f2933"
    },
    "scaling": {
      "label": {
        "enabled": false
      }
    },
    "chosen": {
      "node": true
    }
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "dynamic",
      "roundness": 0.18
    },
    "color": {
      "inherit": false,
      "highlight": "#1f2933",
      "hover": "#1f2933"
    },
    "selectionWidth": 2.4,
    "hoverWidth": 1.4,
    "chosen": {
      "edge": true
    }
  },
  "physics": {
    "enabled": true,
    "solver": "barnesHut",
    "barnesHut": {
      "gravitationalConstant": -3000,
      "centralGravity": 0.3,
      "springLength": 120,
      "springConstant": 0.04,
      "damping": 0.4,
      "avoidOverlap": 0.2
    },
    "stabilization": {
      "enabled": true,
      "iterations": 250,
      "fit": true
    }
  },
  "interaction": {
    "hover": true,
    "hoverConnectedEdges": true,
    "selectConnectedEdges": true,
    "tooltipDelay": 80,
    "dragNodes": true,
    "dragView": true,
    "zoomView": true,
    "hideEdgesOnDrag": false,
    "navigationButtons": false,
    "keyboard": false
  }
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_label(title: str, max_len: int = 24) -> str:
    """Truncate a paper title for in-graph display."""
    if not title:
        return ""
    if len(title) <= max_len:
        return title
    return title[: max_len - 3].rstrip() + "..."


# Pyvis falls back to the node id when ``label=""`` so we use a single
# space when a node should appear unlabeled.
HIDDEN_LABEL = " "


def _make_network(height: str = "560px") -> Network:
    """Create a pyvis Network with the project-wide options applied."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=False,
    )
    net.set_options(_PYVIS_OPTIONS)
    return net


# Click on a node:        animated zoom + slide-in inspect panel for that paper.
# Click on empty space:   fit graph back to view + slide panel away.
# Click "Set this as focus" button in the panel: triggers a Streamlit rerun
# by appending ``?focus_swap=<paper_id>`` to the parent URL.
# (No doubleClick handler — it used to race the single-click handler and
# caused a zoom-in / zoom-out flicker on quick double-clicks.)
_OVERLAY_CSS = """
<style>
:root {
  --rg-bg:        #f3f5f8;
  --rg-surface:   #ffffff;
  --rg-border:    #e3e8ef;
  --rg-border-strong: #d2dae4;
  --rg-text:      #1f2933;
  --rg-text-muted:#5b6b7c;
  --rg-text-faint:#8895a3;
  --rg-accent:    #2f5fb3;
  --rg-accent-soft:#eaf1fb;
  --rg-shadow-sm: 0 1px 2px rgba(20,40,80,0.05);
  --rg-shadow-md: 0 1px 2px rgba(20,40,80,0.04), 0 4px 18px rgba(20,40,80,0.06);
}

html, body {
  margin: 0; padding: 0;
  background: var(--rg-bg);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
               'Helvetica Neue', Arial, sans-serif;
  color: var(--rg-text);
  height: 100%;
  overflow: hidden;
  letter-spacing: -0.005em;
}

/* Pyvis wraps mynetwork in a .card; flatten it so we can position
   mynetwork ourselves without their bordered card frame. */
.card { border: 0 !important; box-shadow: none !important; background: transparent !important;
        width: 100% !important; height: 100% !important; padding: 0 !important; margin: 0 !important; }
#mynetwork {
  position: absolute !important;
  top: 56px !important;
  left: 0 !important;
  right: 340px !important;
  bottom: 0 !important;
  width: auto !important;
  height: auto !important;
  background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%) !important;
  border-right: 1px solid var(--rg-border) !important;
  border-top:    1px solid var(--rg-border) !important;
}

/* ========================================================
   Controls bar (top of the iframe)
   ======================================================== */
.rg-controls {
  position: fixed;
  top: 0; left: 0; right: 340px;
  height: 56px;
  display: flex; align-items: center;
  gap: 6px;
  padding: 0 14px;
  background: var(--rg-surface);
  border-bottom: 1px solid var(--rg-border);
  z-index: 30;
}
.rg-controls-group {
  display: flex; align-items: center; gap: 4px;
  padding-right: 10px; margin-right: 4px;
  border-right: 1px solid var(--rg-border);
}
.rg-controls-group:last-child { border-right: 0; }
.rg-btn {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--rg-surface);
  color: var(--rg-text);
  border: 1px solid var(--rg-border);
  border-radius: 7px;
  font-size: 12.5px;
  font-weight: 500;
  font-family: inherit;
  padding: 5px 10px;
  cursor: pointer;
  transition: all 0.14s ease;
  box-shadow: var(--rg-shadow-sm);
  user-select: none;
}
.rg-btn:hover {
  border-color: var(--rg-border-strong);
  background: #fafbfc;
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(20,40,80,0.06);
}
.rg-btn.is-active {
  background: var(--rg-accent-soft);
  border-color: #b6cdec;
  color: var(--rg-accent);
}
.rg-btn-icon { width: 14px; height: 14px; opacity: 0.8; }
.rg-controls-spacer { flex: 1; }

/* ========================================================
   Right sidebar (always visible — replaces the slide-in card)
   ======================================================== */
.rg-sidebar {
  position: fixed;
  top: 0; right: 0;
  width: 340px;
  height: 100vh;
  background: var(--rg-surface);
  border-left: 1px solid var(--rg-border);
  box-shadow: -4px 0 18px rgba(20,40,80,0.04);
  display: flex; flex-direction: column;
  z-index: 40;
}
.rg-sidebar-head {
  padding: 16px 20px 12px 20px;
  border-bottom: 1px solid var(--rg-border);
}
.rg-status {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  background: var(--rg-accent-soft);
  color: var(--rg-accent);
  border: 1px solid #d4e1f4;
  margin-bottom: 10px;
}
.rg-status .rg-status-dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--rg-accent);
}
.rg-status.is-hovered { background: #fff7e8; color: #b67e1c; border-color: #f1d9a9; }
.rg-status.is-hovered .rg-status-dot { background: #d99a3a; }
.rg-status.is-selected { background: #eaf6ec; color: #3d7c4c; border-color: #c8e3cd; }
.rg-status.is-selected .rg-status-dot { background: #5a9569; }

.rg-sidebar h3 {
  margin: 0 0 6px 0;
  font-size: 16px; font-weight: 600; line-height: 1.32;
  color: var(--rg-text);
  letter-spacing: -0.01em;
  transition: opacity 0.18s ease;
}
.rg-meta-row {
  display: flex; flex-wrap: wrap; gap: 4px 10px;
  font-size: 12.5px; color: var(--rg-text-muted);
}
.rg-meta-row b { color: var(--rg-text); font-weight: 600; }
.rg-meta-row .rg-meta-sep {
  color: var(--rg-border-strong);
}

.rg-sidebar-body {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 14px 20px 12px 20px;
  display: flex; flex-direction: column; gap: 14px;
}
.rg-section-label {
  font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--rg-text-faint);
  margin-bottom: 4px;
}
.rg-section { font-size: 12.8px; line-height: 1.5; color: var(--rg-text); }

.rg-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.rg-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  background: #eef2f7;
  border: 1px solid #e0e7ef;
  color: #3a4a5b;
  font-size: 11.5px;
  font-weight: 500;
}

.rg-connection {
  background: var(--rg-accent-soft);
  border-left: 3px solid var(--rg-accent);
  border-radius: 0 6px 6px 0;
  padding: 8px 10px;
  font-size: 12px;
  color: #2c4a78;
}

.rg-abstract {
  font-size: 12.5px;
  line-height: 1.55;
  color: #37474f;
  max-height: 220px;
  overflow-y: auto;
  padding-right: 6px;
}

.rg-sidebar-foot {
  padding: 12px 20px 16px 20px;
  border-top: 1px solid var(--rg-border);
  display: flex; gap: 8px;
}
.rg-action {
  flex: 1;
  display: inline-flex; align-items: center; justify-content: center;
  gap: 5px;
  background: var(--rg-surface);
  color: var(--rg-text);
  border: 1px solid var(--rg-border);
  border-radius: 8px;
  font-size: 12.5px; font-weight: 500;
  font-family: inherit;
  padding: 7px 10px;
  cursor: pointer;
  transition: all 0.14s ease;
  text-decoration: none;
}
.rg-action:hover {
  border-color: var(--rg-border-strong);
  background: #fafbfc;
}
.rg-action.is-primary {
  background: var(--rg-accent);
  border-color: var(--rg-accent);
  color: #ffffff;
}
.rg-action.is-primary:hover { background: #244c93; border-color: #244c93; }
.rg-action[disabled] {
  opacity: 0.55; cursor: default;
  background: #f4f6f9; color: #8895a3; border-color: var(--rg-border);
}

/* ========================================================
   Compact legend (bottom-left, inside the canvas area)
   ======================================================== */
.rg-legend {
  position: fixed;
  bottom: 12px;
  left: 12px;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(4px);
  border: 1px solid var(--rg-border);
  border-radius: 9px;
  padding: 8px 11px;
  font-size: 11.5px;
  color: var(--rg-text-muted);
  box-shadow: var(--rg-shadow-sm);
  display: flex; gap: 14px; align-items: center;
  z-index: 25;
}
.rg-legend-item { display: flex; align-items: center; gap: 6px; }
.rg-legend-dot {
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  border: 1.5px solid #ffffff;
  box-shadow: 0 0 0 1px var(--rg-border);
}
.rg-legend-line { display: inline-block; width: 22px; height: 2px; background: #7d8b96; }
.rg-legend-line.rg-dashed {
  height: 0; border-top: 2px dashed #9ea7c9; background: transparent;
}

/* ========================================================
   Responsive: stack sidebar below the canvas on narrow widths
   ======================================================== */
@media (max-width: 880px) {
  #mynetwork { right: 0 !important; bottom: 240px !important; }
  .rg-controls { right: 0; }
  .rg-sidebar {
    position: fixed; top: auto; bottom: 0; left: 0; right: 0;
    width: 100%; height: 240px;
    border-left: 0; border-top: 1px solid var(--rg-border);
    box-shadow: 0 -4px 18px rgba(20,40,80,0.05);
  }
}
</style>
"""

# Inline SVG helpers — keep tiny so they stay readable in the source.
_ICON_FIT       = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>'
_ICON_RESET     = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 8a5 5 0 1 0 1.5-3.5"/><path d="M3 3v3h3"/></svg>'
_ICON_LABELS    = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 4h10M3 8h10M3 12h7"/></svg>'
_ICON_CITATION  = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 8h12"/></svg>'
_ICON_SIMILAR   = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="2 2"><path d="M2 8h12"/></svg>'
_ICON_OPEN      = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 2h5v5"/><path d="M14 2 7 9"/><path d="M12 9v4a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h4"/></svg>'
_ICON_FOCUS     = '<svg class="rg-btn-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="8" cy="8" r="3"/><circle cx="8" cy="8" r="6"/></svg>'

_OVERLAY_HTML = f"""
<div class="rg-controls" role="toolbar" aria-label="Graph controls">
  <div class="rg-controls-group">
    <button id="rg-ctl-fit"    class="rg-btn" title="Fit graph to view">{_ICON_FIT} Fit</button>
    <button id="rg-ctl-reset"  class="rg-btn" title="Reset to focus">{_ICON_RESET} Reset</button>
    <button id="rg-ctl-labels" class="rg-btn" title="Show all paper labels">{_ICON_LABELS} Labels</button>
  </div>
  <div class="rg-controls-group">
    <button id="rg-ctl-cite"   class="rg-btn is-active" title="Show citation edges">{_ICON_CITATION} Citation</button>
    <button id="rg-ctl-sim"    class="rg-btn is-active" title="Show similarity edges">{_ICON_SIMILAR} Similarity</button>
  </div>
  <div class="rg-controls-spacer"></div>
</div>

<div class="rg-legend" aria-label="Edge legend">
  <div class="rg-legend-item">
    <span class="rg-legend-dot" style="background:#2f5fb3"></span>
    <span>focus</span>
  </div>
  <div class="rg-legend-item">
    <span class="rg-legend-dot" style="background:#5d8b56"></span>
    <span>citation</span>
  </div>
  <div class="rg-legend-item">
    <span class="rg-legend-dot" style="background:#c69457"></span>
    <span>similarity</span>
  </div>
  <div class="rg-legend-item">
    <span class="rg-legend-line"></span>
    <span>cites</span>
  </div>
  <div class="rg-legend-item">
    <span class="rg-legend-line rg-dashed"></span>
    <span>similar (text)</span>
  </div>
</div>

<aside class="rg-sidebar" aria-label="Paper details">
  <div class="rg-sidebar-head">
    <div id="rg-status" class="rg-status">
      <span class="rg-status-dot"></span>
      <span id="rg-status-label">Current focus</span>
    </div>
    <h3 id="rg-title">&nbsp;</h3>
    <div id="rg-meta" class="rg-meta-row"></div>
  </div>
  <div class="rg-sidebar-body">
    <div id="rg-section-authors" class="rg-section" style="display:none">
      <div class="rg-section-label">Authors</div>
      <div id="rg-authors"></div>
    </div>
    <div id="rg-section-tags" class="rg-section" style="display:none">
      <div class="rg-section-label">Topics</div>
      <div id="rg-tags" class="rg-tags"></div>
    </div>
    <div id="rg-section-connection" class="rg-section" style="display:none">
      <div class="rg-section-label">Why connected</div>
      <div id="rg-connection" class="rg-connection"></div>
    </div>
    <div id="rg-section-abstract" class="rg-section" style="display:none">
      <div class="rg-section-label">Abstract</div>
      <div id="rg-abstract" class="rg-abstract"></div>
    </div>
  </div>
  <div class="rg-sidebar-foot">
    <button id="rg-action-focus" class="rg-action is-primary">{_ICON_FOCUS} Set as focus</button>
    <a id="rg-action-open" class="rg-action" target="_blank" rel="noopener">{_ICON_OPEN} Open paper</a>
  </div>
</aside>
"""

_OVERLAY_JS_TEMPLATE = """
<script>
(function() {
  var FOCUS_ID = __FOCUS_ID__;
  var PANEL_DATA = __PANEL_DATA__;
  var ZOOM_SCALE = 1.55;
  var ANIM_MS = 420;

  function el(id) { return document.getElementById(id); }
  function escapeHtml(s) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(String(s == null ? '' : s)));
    return div.innerHTML;
  }

  // Streamlit's iframe sandbox is "allow-same-origin allow-scripts ..."
  // but does NOT include "allow-top-navigation", so directly assigning
  // window.parent.location.href throws SecurityError. With same-origin
  // we can still touch parent.document, so we inject a <script> tag
  // into it — the script runs in the parent's context, free of the
  // iframe sandbox, and can navigate the top page normally.
  function swapFocus(nodeId) {
    var topWin = null;
    try { topWin = window.top || window.parent; } catch (e) { topWin = window.parent; }
    try {
      var pdoc = topWin.document;
      var s = pdoc.createElement('script');
      s.text =
        "(function(){try{" +
          "var u=new URL(window.location.href);" +
          "u.searchParams.set('focus_swap'," + JSON.stringify(String(nodeId)) + ");" +
          "u.searchParams.set('page','Graph Explorer');" +
          "window.location.assign(u.toString());" +
        "}catch(e){console.error('rg focus swap failed:',e);}})();";
      (pdoc.body || pdoc.documentElement).appendChild(s);
    } catch (err) {
      console.error('ResearchGraph: cannot reach parent document to swap focus:', err);
    }
  }

  // ----------------------------------------------------------------
  // Sidebar state machine
  //
  //   stickyNodeId  =  last clicked node (null when nothing is locked)
  //   currentNodeId =  what the sidebar is showing right now
  //
  // Hover transiently changes currentNodeId (badge: "Hovered paper").
  // Click locks stickyNodeId (badge: "Selected paper" or "Current
  // focus" if it is the focus). With nothing locked and nothing
  // hovered, sidebar reverts to the focus paper.
  // ----------------------------------------------------------------
  var stickyNodeId = null;
  var currentNodeId = null;

  function statusFor(nodeId) {
    if (nodeId === FOCUS_ID) return { label: 'Current focus',  cls: '' };
    if (nodeId === stickyNodeId) return { label: 'Selected paper', cls: 'is-selected' };
    return { label: 'Hovered paper', cls: 'is-hovered' };
  }

  function paintSidebar(nodeId) {
    var data = PANEL_DATA[nodeId] || PANEL_DATA[FOCUS_ID];
    if (!data) return;
    currentNodeId = nodeId;
    var status = statusFor(nodeId);

    var statusEl = el('rg-status');
    statusEl.className = 'rg-status ' + status.cls;
    el('rg-status-label').textContent = status.label;

    el('rg-title').textContent = data.title || '';

    var meta = el('rg-meta');
    meta.innerHTML = '';
    var bits = [];
    if (data.year)   bits.push('<span><b>'+escapeHtml(data.year)+'</b></span>');
    if (data.venue)  bits.push('<span>'+escapeHtml(data.venue)+'</span>');
    bits.push('<span>'+Number(data.citations || 0).toLocaleString()+' citations</span>');
    meta.innerHTML = bits.join('<span class="rg-meta-sep">·</span>');

    var authorsSec = el('rg-section-authors');
    if (data.authors) {
      authorsSec.style.display = '';
      el('rg-authors').textContent = data.authors;
    } else {
      authorsSec.style.display = 'none';
    }

    var tagsSec = el('rg-section-tags');
    if (data.topics && data.topics.length) {
      tagsSec.style.display = '';
      el('rg-tags').innerHTML = data.topics.map(function(t) {
        return '<span class="rg-tag">'+escapeHtml(t)+'</span>';
      }).join('');
    } else {
      tagsSec.style.display = 'none';
    }

    var connSec = el('rg-section-connection');
    if (data.edge_reason) {
      connSec.style.display = '';
      el('rg-connection').textContent = data.edge_reason;
    } else {
      connSec.style.display = 'none';
    }

    var absSec = el('rg-section-abstract');
    if (data.abstract) {
      absSec.style.display = '';
      el('rg-abstract').textContent = data.abstract;
    } else {
      absSec.style.display = 'none';
    }

    // Action buttons
    var focusBtn = el('rg-action-focus');
    if (nodeId === FOCUS_ID) {
      focusBtn.disabled = true;
      focusBtn.classList.remove('is-primary');
      focusBtn.innerHTML = focusBtn.innerHTML.replace(/Set as focus|Already focus/, 'Already focus');
    } else {
      focusBtn.disabled = false;
      focusBtn.classList.add('is-primary');
      focusBtn.innerHTML = focusBtn.innerHTML.replace(/Set as focus|Already focus/, 'Set as focus');
    }
    focusBtn.onclick = function() {
      if (nodeId !== FOCUS_ID) swapFocus(nodeId);
    };

    var openBtn = el('rg-action-open');
    if (data.url) {
      openBtn.href = data.url;
      openBtn.removeAttribute('aria-disabled');
      openBtn.style.opacity = '';
      openBtn.style.pointerEvents = '';
    } else {
      openBtn.removeAttribute('href');
      openBtn.setAttribute('aria-disabled', 'true');
      openBtn.style.opacity = '0.55';
      openBtn.style.pointerEvents = 'none';
    }
  }

  // ----------------------------------------------------------------
  // Edge / node focus-mode — dim everything not connected to the
  // hovered or selected node so the eye reads the 1-hop neighborhood
  // immediately. Default state (nothing hovered) keeps every edge
  // muted so the canvas stays quiet.
  // ----------------------------------------------------------------
  var DIM_EDGE_COLOR = 'rgba(40,55,70,0.06)';
  var DIM_EDGE_WIDTH = 0.4;
  var DIM_NODE_COLOR = { background: '#e5e8ec', border: '#ffffff' };
  var ORIG_EDGES = null;          // {edgeId: {color, width, dashes, edge_kind}}
  var ORIG_NODES = null;          // {nodeId: {color, size, label}}
  var labelsVisible = false;
  var edgeFilter = { citation: true, similarity: true };

  function snapshot() {
    if (ORIG_EDGES && ORIG_NODES) return;
    ORIG_EDGES = {};
    edges.get().forEach(function(e) {
      ORIG_EDGES[e.id] = {
        color: e.color, width: e.width, dashes: e.dashes,
        // Map vis-network's edge title (set in Python) back to a kind.
        kind: (function(t) {
          if (!t) return 'other';
          t = String(t).toLowerCase();
          if (t.indexOf('similarity') === 0) return 'similarity';
          if (t.indexOf('citation + similarity') === 0) return 'both';
          if (t.indexOf('citation') === 0) return 'citation';
          if (t.indexOf('shared') === 0) return 'shared_author';
          return 'other';
        })(e.title)
      };
    });
    ORIG_NODES = {};
    nodes.get().forEach(function(n) {
      ORIG_NODES[n.id] = { color: n.color, size: n.size, label: n.label };
    });
  }

  // Edges hidden by an edge-type filter never come back via highlight.
  function isHiddenByFilter(edgeId) {
    var k = ORIG_EDGES[edgeId].kind;
    if (k === 'citation' && !edgeFilter.citation) return true;
    if (k === 'similarity' && !edgeFilter.similarity) return true;
    if (k === 'both' && !edgeFilter.citation && !edgeFilter.similarity) return true;
    return false;
  }

  function applyHighlight(nodeId) {
    snapshot();
    var connectedEdges = new Set(network.getConnectedEdges(nodeId));
    var connectedNodes = new Set(network.getConnectedNodes(nodeId));
    connectedNodes.add(nodeId);
    var edgeUpdates = [];
    Object.keys(ORIG_EDGES).forEach(function(eid) {
      if (isHiddenByFilter(eid)) {
        edgeUpdates.push({ id: eid, hidden: true });
      } else if (connectedEdges.has(eid)) {
        edgeUpdates.push({ id: eid, hidden: false,
                           color: ORIG_EDGES[eid].color,
                           width: ORIG_EDGES[eid].width });
      } else {
        edgeUpdates.push({ id: eid, hidden: false,
                           color: DIM_EDGE_COLOR, width: DIM_EDGE_WIDTH });
      }
    });
    edges.update(edgeUpdates);

    // Slight scale-up on the hovered/selected node for an elegant pop.
    var nodeUpdates = [];
    Object.keys(ORIG_NODES).forEach(function(nid) {
      var orig = ORIG_NODES[nid];
      if (nid === FOCUS_ID || connectedNodes.has(nid)) {
        var sz = orig.size;
        if (nid === nodeId && nid !== FOCUS_ID) sz = orig.size + 6;
        nodeUpdates.push({ id: nid, color: orig.color, size: sz });
      } else {
        nodeUpdates.push({ id: nid, color: DIM_NODE_COLOR, size: orig.size });
      }
    });
    nodes.update(nodeUpdates);
  }

  function restoreHighlight() {
    if (!ORIG_EDGES || !ORIG_NODES) return;
    edges.update(Object.keys(ORIG_EDGES).map(function(eid) {
      return {
        id: eid,
        hidden: isHiddenByFilter(eid),
        color: ORIG_EDGES[eid].color,
        width: ORIG_EDGES[eid].width
      };
    }));
    nodes.update(Object.keys(ORIG_NODES).map(function(nid) {
      return { id: nid, color: ORIG_NODES[nid].color, size: ORIG_NODES[nid].size };
    }));
  }

  // ----------------------------------------------------------------
  // Controls bar wiring
  // ----------------------------------------------------------------
  function bindControls() {
    el('rg-ctl-fit').addEventListener('click', function() {
      network.fit({ animation: { duration: ANIM_MS, easingFunction: 'easeInOutQuad' } });
    });
    el('rg-ctl-reset').addEventListener('click', function() {
      stickyNodeId = null;
      restoreHighlight();
      paintSidebar(FOCUS_ID);
      network.fit({ animation: { duration: ANIM_MS, easingFunction: 'easeInOutQuad' } });
    });
    el('rg-ctl-labels').addEventListener('click', function(e) {
      snapshot();
      labelsVisible = !labelsVisible;
      e.currentTarget.classList.toggle('is-active', labelsVisible);
      var updates = [];
      Object.keys(ORIG_NODES).forEach(function(nid) {
        if (nid === FOCUS_ID) return;  // focus already labeled
        var data = PANEL_DATA[nid];
        var fullTitle = data ? data.title : '';
        var short = fullTitle && fullTitle.length > 24
                    ? fullTitle.substring(0, 22).replace(/\\s+$/,'') + '\\u2026'
                    : (fullTitle || '');
        updates.push({ id: nid, label: labelsVisible ? short : ' ' });
      });
      nodes.update(updates);
    });
    el('rg-ctl-cite').addEventListener('click', function(e) {
      edgeFilter.citation = !edgeFilter.citation;
      e.currentTarget.classList.toggle('is-active', edgeFilter.citation);
      reapplyFilters();
    });
    el('rg-ctl-sim').addEventListener('click', function(e) {
      edgeFilter.similarity = !edgeFilter.similarity;
      e.currentTarget.classList.toggle('is-active', edgeFilter.similarity);
      reapplyFilters();
    });
  }

  function reapplyFilters() {
    snapshot();
    if (stickyNodeId) {
      applyHighlight(stickyNodeId);
    } else if (currentNodeId && currentNodeId !== FOCUS_ID) {
      applyHighlight(currentNodeId);
    } else {
      restoreHighlight();
    }
  }

  // ----------------------------------------------------------------
  // Network event wiring
  // ----------------------------------------------------------------
  function _attach() {
    if (typeof network === 'undefined' || !network) {
      return setTimeout(_attach, 60);
    }

    // Default state — show focus paper in the sidebar.
    paintSidebar(FOCUS_ID);
    bindControls();

    // Idle-freeze: stop physics when the layout is at rest, restart
    // for the duration of a drag so dynamic-smooth edges flex.
    var dragFreezeTimer = null;
    network.once('stabilizationIterationsDone', function() {
      try { network.stopSimulation(); } catch (e) {}
    });
    network.on('dragStart', function() {
      if (dragFreezeTimer) { clearTimeout(dragFreezeTimer); dragFreezeTimer = null; }
      try { network.startSimulation(); } catch (e) {}
    });
    network.on('dragEnd', function() {
      if (dragFreezeTimer) clearTimeout(dragFreezeTimer);
      dragFreezeTimer = setTimeout(function() {
        try { network.stopSimulation(); } catch (e) {}
        dragFreezeTimer = null;
      }, 800);
    });

    network.on('click', function(params) {
      var id = (params.nodes && params.nodes[0]) || null;
      if (id && PANEL_DATA[id]) {
        stickyNodeId = id;
        applyHighlight(id);
        paintSidebar(id);
        network.focus(id, {
          scale: ZOOM_SCALE,
          animation: { duration: ANIM_MS, easingFunction: 'easeInOutQuad' }
        });
      } else {
        // Click empty space → clear selection, re-fit, revert to focus.
        stickyNodeId = null;
        restoreHighlight();
        paintSidebar(FOCUS_ID);
        network.fit({
          animation: { duration: ANIM_MS, easingFunction: 'easeInOutQuad' }
        });
      }
    });
    network.on('hoverNode', function(params) {
      if (stickyNodeId !== null) return;  // sticky click wins
      applyHighlight(params.node);
      paintSidebar(params.node);
    });
    network.on('blurNode', function() {
      if (stickyNodeId !== null) return;
      restoreHighlight();
      paintSidebar(FOCUS_ID);
    });
  }
  _attach();
})();
</script>
"""


def _build_panel_data(
    rg: "ResearchGraph",
    center_id: str,
    node_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Compact JSON-friendly metadata for every node shown in the canvas.

    Used by the in-canvas overlay panel: clicking a node populates the panel
    from this dict (no extra round-trip to the server).
    """
    data: dict[str, dict[str, Any]] = {}
    for nid in node_ids:
        paper = rg.get_paper(nid)
        if paper is None:
            continue
        authors_str = ""
        if paper.authors:
            authors_str = ", ".join(paper.authors[:4])
            if len(paper.authors) > 4:
                authors_str += f" + {len(paper.authors) - 4} more"
        edge_reason = ""
        if nid != center_id and rg.graph.has_edge(center_id, nid):
            ed = rg.graph[center_id][nid]
            reason = _edge_reason(
                ed.get("edge_type", ""), ed.get("similarity"),
            )
            if reason:
                edge_reason = reason
        data[nid] = {
            "id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "venue": paper.venue or "",
            "citations": int(paper.citation_count or 0),
            "authors": authors_str,
            "topics": list(paper.topic_words(6)),
            "abstract": paper.abstract or "",
            "url": paper.url or "",
            "edge_reason": edge_reason,
        }
    return data


def _inject_overlay(
    html: str,
    focus_id: str | None,
    panel_data: dict[str, dict[str, Any]] | None,
) -> str:
    """Inject the overlay panel CSS / HTML / JS into the pyvis output.

    If ``panel_data`` is ``None`` (e.g. for path views) the overlay is
    omitted entirely and we keep only a minimal click-to-fit handler.
    """
    if panel_data is None:
        fallback = """
<script>
(function() {
  function _attach() {
    if (typeof network === 'undefined' || !network) {
      return setTimeout(_attach, 60);
    }
    network.on('click', function(params) {
      if (params.nodes && params.nodes.length > 0) {
        network.focus(params.nodes[0], {
          scale: 1.5,
          animation: { duration: 450, easingFunction: 'easeInOutQuad' }
        });
      } else {
        network.fit({
          animation: { duration: 450, easingFunction: 'easeInOutQuad' }
        });
      }
    });
  }
  _attach();
})();
</script>
"""
        injection = fallback
    else:
        js = (
            _OVERLAY_JS_TEMPLATE
            .replace("__FOCUS_ID__", json.dumps(focus_id))
            .replace("__PANEL_DATA__", json.dumps(panel_data))
        )
        injection = _OVERLAY_CSS + _OVERLAY_HTML + js

    if "</body>" in html:
        return html.replace("</body>", injection + "</body>")
    return html + injection


def _hover_html(paper: Paper, edge_reason: str | None = None) -> str:
    """Title-only tooltip.

    Hover is for fast identification — full metadata (year, venue,
    citations, authors, topics, abstract, connection reason) lives in
    the slide-in inspect panel that opens when the node is clicked.
    """
    return paper.title


def _edge_visuals(edge_type: str, similarity: float | None) -> dict[str, Any]:
    """Default edge styling.

    Default colors are deliberately muted (low width, soft tones) so the
    canvas reads as quiet background. The interactive layer (see
    ``_OVERLAY_JS_TEMPLATE``) re-styles edges live: hovered/selected
    nodes get their connected edges restored to vivid color while
    every other edge is faded to near-transparent grey.
    """
    if edge_type == "citation":
        return {"color": EDGE_CITATION, "width": 1.0, "dashes": False,
                "title": "Citation"}
    if edge_type == "similarity":
        return {"color": EDGE_SIMILARITY, "width": 0.8, "dashes": True,
                "title": "Similarity"}
    if edge_type == "both":
        return {"color": EDGE_BOTH, "width": 1.4, "dashes": False,
                "title": "Citation + similarity"}
    if edge_type == "shared_author":
        return {"color": COLOR_SHARED_AUTHOR, "width": 0.8, "dashes": True,
                "title": "Shared author"}
    return {"color": "#bdbdbd", "width": 0.8, "dashes": False,
            "title": edge_type}


def _neighbor_color(edge_type: str) -> str:
    return {
        "citation": COLOR_CITATION,
        "similarity": COLOR_SIMILARITY,
        "both": COLOR_BOTH,
        "shared_author": COLOR_SHARED_AUTHOR,
    }.get(edge_type, COLOR_CITATION)


def _edge_reason(edge_type: str, similarity: float | None) -> str | None:
    if edge_type == "citation":
        return "direct citation link"
    if edge_type == "similarity" and similarity:
        return f"text similarity {similarity:.2f}"
    if edge_type == "both" and similarity:
        return f"citation + similarity {similarity:.2f}"
    if edge_type == "shared_author":
        return "shared author"
    return None


def _neighbor_priority(edge_type: str, similarity: float | None,
                       citation_count: int) -> float:
    """Higher → more important neighbor (for ordering on the inner ring)."""
    type_w = {"both": 2.5, "citation": 2.0, "shared_author": 0.7}.get(
        edge_type, 1.0 + float(similarity or 0.0)
    )
    return type_w + (citation_count / 100000.0)


# ---------------------------------------------------------------------------
# Public API: neighborhood
# ---------------------------------------------------------------------------

def render_neighborhood(
    rg: ResearchGraph,
    center_id: str,
    radius: int = 1,
    max_nodes: int = 40,
    label_neighbors: bool = False,
) -> str:
    """Render an interactive neighborhood graph with a fixed radial layout.

    Layout:
        - Focus paper pinned at (0, 0).
        - Direct neighbors pinned on a circle at radius ``INNER_RADIUS``,
          ordered by importance (citation > both > similarity > etc.)
          so structurally important neighbors sit beside each other.
        - For ``radius >= 2``, second-hop / context nodes pinned on
          a wider outer ring.

    Labels:
        - Focus paper: always labeled (short title).
        - Direct neighbors: labeled only if ``label_neighbors=True``.
        - Context nodes: never labeled.

    Hover tooltips always contain the full title + metadata.
    """
    if center_id not in rg.graph:
        return "<p>No nodes to display.</p>"

    direct: list[str] = list(rg.graph.neighbors(center_id))

    # Order direct neighbors by importance so the highest-signal ones
    # come first on the ring (visually more prominent).
    direct.sort(
        key=lambda nid: _neighbor_priority(
            rg.graph[center_id][nid].get("edge_type", ""),
            rg.graph[center_id][nid].get("similarity"),
            rg.papers[nid].citation_count if nid in rg.papers else 0,
        ),
        reverse=True,
    )

    # Compute outer-ring nodes for radius >= 2 (capped).
    outer: list[str] = []
    if radius >= 2:
        seen = {center_id, *direct}
        for nid in direct:
            for nbr in rg.graph.neighbors(nid):
                if nbr not in seen:
                    seen.add(nbr)
                    outer.append(nbr)
        outer.sort(
            key=lambda nid: rg.papers[nid].citation_count
            if nid in rg.papers else 0,
            reverse=True,
        )
        # Cap outer ring so the canvas stays uncluttered.
        outer = outer[: max(0, max_nodes - 1 - len(direct))]

    nodes_to_show = {center_id, *direct, *outer}
    net = _make_network()

    # ------- Focus node (pinned at center) -------
    focus_paper = rg.get_paper(center_id)
    if focus_paper is None:
        return "<p>No nodes to display.</p>"
    net.add_node(
        center_id,
        label=short_label(focus_paper.title, max_len=22),
        title=_hover_html(focus_paper),
        color={
            "background": COLOR_FOCUS,
            "border": "#ffffff",
            "highlight": {"background": COLOR_FOCUS, "border": "#ffffff"},
            "hover":     {"background": COLOR_FOCUS, "border": "#ffffff"},
        },
        size=38,
        borderWidth=4,
        x=0, y=0,
        fixed=True,
        physics=False,
        # Soft blue glow gives the focus an elegant emphasis without
        # being loud — it reads as "this is the anchor of the view".
        shadow={
            "enabled": True,
            "color": "rgba(47,95,179,0.45)",
            "size": 22,
            "x": 0, "y": 0,
        },
        font={"size": 14, "face": "Inter, system-ui, sans-serif",
              "color": "#0d2a52", "bold": True,
              "strokeWidth": 4, "strokeColor": "#ffffff"},
    )

    # ------- Inner ring: direct neighbors -------
    n_direct = len(direct)
    # Start angle at top (-pi/2) and go clockwise so the most important
    # neighbor is at 12 o'clock.
    for i, nid in enumerate(direct):
        paper = rg.get_paper(nid)
        if paper is None:
            continue
        angle = -math.pi / 2 + (2 * math.pi * i / max(n_direct, 1))
        x = INNER_RADIUS * math.cos(angle)
        y = INNER_RADIUS * math.sin(angle)

        ed = rg.graph[center_id][nid]
        etype = ed.get("edge_type", "")
        sim = ed.get("similarity")
        color = _neighbor_color(etype)
        reason = _edge_reason(etype, sim)

        label = short_label(paper.title, max_len=20) if label_neighbors else HIDDEN_LABEL

        # The radial (x, y) acts as an *initial position* hint so the
        # barnesHut stabilization pass settles into a clean ring layout
        # instead of an arbitrary one. The node itself is fully
        # physics-driven afterwards: drag it and the spring/gravity
        # forces propagate to its neighbors and bend the edges.
        net.add_node(
            nid,
            label=label,
            title=_hover_html(paper, edge_reason=reason),
            color={
                "background": color,
                "border": "#ffffff",
                "highlight": {"background": color, "border": "#ffffff"},
                "hover":     {"background": color, "border": "#ffffff"},
            },
            size=22,
            borderWidth=2,
            x=x, y=y,
            font={"size": 11, "face": "Inter, system-ui, sans-serif",
                  "color": "#1f2933",
                  "strokeWidth": 4, "strokeColor": "#ffffff"},
        )

    # ------- Outer ring: context (2nd-hop) nodes -------
    n_outer = len(outer)
    if n_outer:
        for i, nid in enumerate(outer):
            paper = rg.get_paper(nid)
            if paper is None:
                continue
            angle = -math.pi / 2 + (2 * math.pi * i / n_outer)
            x = OUTER_RADIUS * math.cos(angle)
            y = OUTER_RADIUS * math.sin(angle)
            net.add_node(
                nid,
                label=HIDDEN_LABEL,  # context nodes never labeled
                title=_hover_html(paper),
                color=COLOR_CONTEXT,
                size=8,
                borderWidth=1,
                x=x, y=y,
                font={"size": 10, "face": "Arial",
                      "color": "#888888",
                      "strokeWidth": 2, "strokeColor": "#ffffff"},
            )

    # ------- Edges -------
    seen: set[tuple[str, str]] = set()
    for u in nodes_to_show:
        if u not in rg.graph:
            continue
        for v in rg.graph.neighbors(u):
            if v not in nodes_to_show:
                continue
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            ed = rg.graph[u][v]
            visuals = _edge_visuals(
                ed.get("edge_type", "unknown"), ed.get("similarity"),
            )
            net.add_edge(u, v, **visuals)

    panel_data = _build_panel_data(rg, center_id, list(nodes_to_show))
    return _inject_overlay(
        net.generate_html(notebook=False),
        focus_id=center_id,
        panel_data=panel_data,
    )


# ---------------------------------------------------------------------------
# Public API: path
# ---------------------------------------------------------------------------

def render_path(
    rg: ResearchGraph,
    path_paper_ids: list[str],
    context_radius: int = 1,
    max_nodes: int = 30,
) -> str:
    """Render a path with a fixed horizontal layout.

    Layout:
        - Path papers are placed on a horizontal line, left to right,
          spaced ``PATH_SPACING_X`` apart.
        - Context nodes (direct neighbors of path nodes that are not
          on the path) are placed above and below the line on a faint
          gray ring.

    Labels:
        - Path papers: short labels.
        - Context papers: never labeled.
    """
    if len(path_paper_ids) < 2:
        return "<p>Need at least two papers to visualize a path.</p>"

    net = _make_network()

    # ------- Path nodes on a centered horizontal line -------
    n = len(path_paper_ids)
    total_width = (n - 1) * PATH_SPACING_X
    x_start = -total_width / 2.0

    path_set = set(path_paper_ids)
    valid_path_ids: list[str] = []
    for i, pid in enumerate(path_paper_ids):
        paper = rg.get_paper(pid)
        if paper is None:
            continue
        valid_path_ids.append(pid)
        x = x_start + i * PATH_SPACING_X
        net.add_node(
            pid,
            label=short_label(paper.title, max_len=22),
            title=_hover_html(paper),
            color=COLOR_PATH,
            size=30,
            borderWidth=4,
            x=x, y=0,
            fixed=True,
            physics=False,
            font={"size": 12, "face": "Arial",
                  "color": "#7c1818", "bold": True,
                  "strokeWidth": 4, "strokeColor": "#ffffff"},
        )

    # ------- Context nodes on faint arcs above and below -------
    if context_radius >= 1:
        context_candidates: list[str] = []
        for pid in valid_path_ids:
            if pid in rg.graph:
                for nbr in rg.graph.neighbors(pid):
                    if nbr not in path_set and nbr not in context_candidates:
                        context_candidates.append(nbr)

        # Cap so the canvas stays clean.
        context_candidates.sort(
            key=lambda n: rg.papers[n].citation_count
            if n in rg.papers else 0,
            reverse=True,
        )
        cap = max(0, max_nodes - len(valid_path_ids))
        context_candidates = context_candidates[:cap]

        # Place candidates alternating above / below, evenly distributed
        # across the path's x range.
        cn = len(context_candidates)
        for i, nid in enumerate(context_candidates):
            paper = rg.get_paper(nid)
            if paper is None:
                continue
            # Distribute x across the path span (slightly beyond ends)
            t = (i + 0.5) / max(cn, 1)
            x = x_start - 60 + t * (total_width + 120)
            y = PATH_CONTEXT_Y if (i % 2 == 0) else -PATH_CONTEXT_Y
            net.add_node(
                nid,
                label=HIDDEN_LABEL,
                title=_hover_html(paper),
                color=COLOR_CONTEXT,
                size=7,
                borderWidth=1,
                x=x, y=y,
                font={"size": 10, "face": "Arial",
                      "color": "#888888",
                      "strokeWidth": 2, "strokeColor": "#ffffff"},
            )

    # ------- Edges -------
    path_edges: set[tuple[str, str]] = set()
    for i in range(len(valid_path_ids) - 1):
        a, b = valid_path_ids[i], valid_path_ids[i + 1]
        path_edges.add((a, b) if a < b else (b, a))

    seen: set[tuple[str, str]] = set()
    rendered_nodes = {n["id"] for n in net.nodes}
    for u in list(rendered_nodes):
        if u not in rg.graph:
            continue
        for v in rg.graph.neighbors(u):
            if v not in rendered_nodes:
                continue
            key = (u, v) if u < v else (v, u)
            if key in seen:
                continue
            seen.add(key)
            if key in path_edges:
                visuals = {
                    "color": EDGE_PATH, "width": 4,
                    "dashes": False, "title": "Path edge",
                }
            else:
                visuals = {
                    "color": EDGE_CONTEXT, "width": 1,
                    "dashes": False,
                    "title": rg.graph[u][v].get("edge_type", "context"),
                }
            net.add_edge(u, v, **visuals)

    return _inject_overlay(
        net.generate_html(notebook=False),
        focus_id=None,
        panel_data=None,
    )


# ---------------------------------------------------------------------------
# External legends (rendered by Streamlit, NOT inside the canvas)
# ---------------------------------------------------------------------------

_LEGEND_CHIP = (
    "display:inline-flex; align-items:center; gap:6px; "
    "padding:2px 10px; border-radius:11px; "
    "background:{bg}; color:#ffffff; font-size:0.78rem; "
    "font-weight:600; letter-spacing:0.2px;"
)


def neighborhood_legend_html() -> str:
    """Compact HTML legend for the neighborhood graph."""
    return (
        '<div style="display:flex; flex-wrap:wrap; gap:10px 14px; '
        'align-items:center; font-size:0.85rem; color:#37474f; '
        'margin:0.15rem 0 0.55rem 0;">'
        f'<span style="{_LEGEND_CHIP.format(bg=COLOR_FOCUS)}">Selected</span>'
        f'<span style="{_LEGEND_CHIP.format(bg=COLOR_CITATION)}">Citation</span>'
        f'<span style="{_LEGEND_CHIP.format(bg=COLOR_SIMILARITY)}">Similarity</span>'
        f'<span style="{_LEGEND_CHIP.format(bg=COLOR_BOTH)}">Both</span>'
        f'<span style="{_LEGEND_CHIP.format(bg="#90a4ae")}">Context</span>'
        '<span style="color:#cfd8dc;">|</span>'
        f'<span><span style="color:{EDGE_CITATION};font-weight:700;'
        'font-size:1.05rem;">━</span> citation edge</span>'
        f'<span><span style="color:{EDGE_SIMILARITY};font-weight:700;'
        'font-size:1.05rem;">┄</span> similarity edge</span>'
        f'<span><span style="color:{EDGE_BOTH};font-weight:700;'
        'font-size:1.05rem;">━</span> both</span>'
        '<span style="flex-basis:100%; height:0;"></span>'
        '<span style="color:#78909c; font-size:0.78rem;">'
        'Only the selected paper is labeled. Hover any dot to see the '
        'full paper title, year, venue, citation count, authors, and '
        'abstract snippet.'
        '</span>'
        '</div>'
    )


def path_legend_html() -> str:
    """Compact HTML legend for the path visualization."""
    return (
        '<div style="display:flex; flex-wrap:wrap; gap:10px 14px; '
        'align-items:center; font-size:0.85rem; color:#37474f; '
        'margin:0.15rem 0 0.55rem 0;">'
        f'<span style="{_LEGEND_CHIP.format(bg=COLOR_PATH)}">Path paper</span>'
        f'<span style="{_LEGEND_CHIP.format(bg="#90a4ae")}">Context paper</span>'
        '<span style="color:#cfd8dc;">|</span>'
        f'<span><span style="color:{EDGE_PATH};font-weight:700;'
        'font-size:1.1rem;">━━</span> path edge</span>'
        f'<span><span style="color:{EDGE_CONTEXT};font-weight:700;'
        'font-size:1.05rem;">━</span> context edge</span>'
        '<span style="flex-basis:100%; height:0;"></span>'
        '<span style="color:#78909c; font-size:0.78rem;">'
        'Only path papers are labeled. Hover any dot for the full paper '
        'metadata.'
        '</span>'
        '</div>'
    )
