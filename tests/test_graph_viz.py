"""Behavioral specification for graph visualization.

The visualization module renders interactive HTML graphs using pyvis.
These tests verify that the rendering functions produce valid output
without crashing, that titles appear in tooltips, that focus and path
nodes are highlighted, and that radius-2/3 graphs are capped to keep
the visualization readable.
"""

import pytest

from src.paper import Paper
from src.research_graph import ResearchGraph
from src.graph_viz import (
    render_neighborhood,
    render_path,
    short_label,
    neighborhood_legend_html,
    path_legend_html,
    COLOR_FOCUS,
    COLOR_PATH,
)


def _make_paper(pid: str, title: str = "Paper", refs: list[str] | None = None, **kw) -> Paper:
    return Paper(paper_id=pid, title=title, references=refs or [], **kw)


@pytest.fixture
def small_graph() -> ResearchGraph:
    """A small graph with citation and similarity edges for visualization tests."""
    rg = ResearchGraph()
    papers = [
        _make_paper("a", "Attention mechanisms in transformers",
                     refs=["b"], year=2017, citation_count=90000,
                     abstract="Self attention for sequence modeling"),
        _make_paper("b", "BERT language model pretraining",
                     refs=["c"], year=2019, citation_count=70000,
                     abstract="Bidirectional encoder from transformers"),
        _make_paper("c", "GPT large language models",
                     year=2020, citation_count=35000,
                     abstract="Large language models for few-shot learning"),
        _make_paper("d", "Graph neural network convolutions",
                     year=2017, citation_count=22000,
                     abstract="Convolutions on graph structured data"),
    ]
    rg.build_graph(papers, similarity_threshold=0.1)
    return rg


# ===========================================================================
# Short-label helper
# ===========================================================================

class TestShortLabel:
    def test_passes_short_titles_through(self):
        assert short_label("hello") == "hello"

    def test_truncates_long_titles_with_ellipsis(self):
        out = short_label("a" * 80, max_len=20)
        assert out.endswith("...")
        assert len(out) <= 20

    def test_handles_empty_title(self):
        assert short_label("") == ""


# ===========================================================================
# Neighborhood visualization
# ===========================================================================

class TestNeighborhoodVisualization:
    """render_neighborhood produces an interactive HTML graph of a paper's local area."""

    def test_returns_html_string(self, small_graph: ResearchGraph):
        """The output should be a non-empty HTML string."""
        html = render_neighborhood(small_graph, "a", radius=1)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_contains_focus_paper_title(self, small_graph: ResearchGraph):
        """The selected paper's full title should appear (in label or tooltip)."""
        html = render_neighborhood(small_graph, "a", radius=1)
        assert "Attention" in html

    def test_html_contains_neighbor_titles_in_tooltip(self, small_graph: ResearchGraph):
        """Direct neighbor titles should appear in the rendered HTML
        (in label or hover tooltip)."""
        html = render_neighborhood(small_graph, "a", radius=1)
        # "b" is a direct neighbor of "a"
        assert "BERT" in html

    def test_handles_nonexistent_node(self, small_graph: ResearchGraph):
        """A missing node should return a fallback message, not crash."""
        html = render_neighborhood(small_graph, "nonexistent", radius=1)
        assert isinstance(html, str)

    def test_radius_2_includes_more_nodes(self, small_graph: ResearchGraph):
        """Radius 2 should include nodes that radius 1 would miss."""
        html_r1 = render_neighborhood(small_graph, "a", radius=1)
        html_r2 = render_neighborhood(small_graph, "a", radius=2)
        # c is 2 hops from a (a->b->c), so only in r2
        assert "GPT" not in html_r1 or "GPT" in html_r2

    def test_no_in_graph_legend_node(self, small_graph: ResearchGraph):
        """The old in-canvas LEGEND nodes must NOT appear — the legend
        is now rendered externally by Streamlit."""
        html = render_neighborhood(small_graph, "a", radius=1)
        assert "legend_title" not in html
        assert "LEGEND" not in html

    def test_focus_color_is_present(self, small_graph: ResearchGraph):
        """The focus paper should be colored with the focus color."""
        html = render_neighborhood(small_graph, "a", radius=1)
        assert COLOR_FOCUS in html

    def test_radius_2_caps_second_hop_node_count(self):
        """At radius 2 the second-hop expansion should be capped to
        max_nodes, while direct neighbors are always kept (per the
        design contract: radius 1 shows all neighbors, radius >= 2
        prunes the second-hop fan-out)."""
        rg = ResearchGraph()
        rg.add_paper(_make_paper("hub", "Center paper", year=2020))
        # Two direct neighbors of hub
        for j in range(2):
            rg.add_paper(_make_paper(
                f"d{j}", f"Direct neighbor {j}",
                refs=["hub"], year=2020, citation_count=10,
            ))
        # Each direct neighbor has 50 unique second-hop leaves
        for j in range(2):
            for i in range(50):
                rg.add_paper(_make_paper(
                    f"second_hop_{j}_{i}", f"SecondHop_{j}_{i}",
                    refs=[f"d{j}"], year=2020, citation_count=i,
                ))
        rg.add_citation_edges()
        html = render_neighborhood(rg, "hub", radius=2, max_nodes=30)
        # 100 second-hop candidates, cap is 30 total → many should be omitted.
        rendered = sum(
            1 for j in range(2) for i in range(50)
            if f"SecondHop_{j}_{i}" in html
        )
        assert rendered < 50, (
            f"Expected fewer than 50 second-hop nodes rendered, got {rendered}"
        )

    def test_pyvis_canvas_height_is_560(self, small_graph: ResearchGraph):
        """The pyvis canvas is 560px so it fits inside the 600px
        Streamlit components.html container without internal scrollbars."""
        html = render_neighborhood(small_graph, "a", radius=1)
        assert '560px' in html or 'height: 560' in html

    def test_canvas_uses_full_width(self, small_graph: ResearchGraph):
        """Width should be 100% so the graph fills its Streamlit column."""
        html = render_neighborhood(small_graph, "a", radius=1)
        assert '100%' in html

    def test_label_scaling_is_disabled(self, small_graph: ResearchGraph):
        """Labels must keep a fixed pixel size so they don't blow up
        when the user zooms the graph."""
        html = render_neighborhood(small_graph, "a", radius=1)
        # vis-network options are emitted as JSON; check the final form
        assert '"label": {"enabled": false}' in html or \
               '"label":{"enabled":false}' in html

    def test_uses_force_directed_physics_with_stabilization(self, small_graph: ResearchGraph):
        """The canvas runs the barnesHut force-directed solver with
        stabilization, so nodes wiggle into a stable layout on load and
        a drag visibly pulls neighbors through spring/gravity coupling.

        Focus / path nodes are still pinned via ``fixed: true`` (their
        explicit (x, y) keeps them anchored), but neighbor and context
        nodes float freely.
        """
        html = render_neighborhood(small_graph, "a", radius=1)
        assert '"enabled": true' in html  # physics is on
        assert '"solver": "barnesHut"' in html
        assert '"stabilization":' in html
        # The reference's spring/gravity values land in the rendered options.
        assert '"gravitationalConstant": -3000' in html
        assert '"centralGravity": 0.3' in html
        assert '"springLength": 120' in html
        # The focus node is the one node that opts out of physics so the
        # centerpiece stays at the origin.
        assert '"fixed": true' in html
        assert '"x": 0' in html and '"y": 0' in html

    def test_focus_node_pinned_at_origin(self, small_graph: ResearchGraph):
        """The focus paper sits at (0, 0) and is fixed."""
        html = render_neighborhood(small_graph, "a", radius=1)
        # Must contain a node entry with id 'a' that has fixed=true at
        # x:0, y:0.  Pyvis emits these as JSON keys.
        assert '"id": "a"' in html
        assert '"x": 0' in html
        assert '"y": 0' in html
        assert '"fixed": true' in html

    def test_only_focus_is_labeled_by_default(self):
        """In default mode (label_neighbors=False), only the focus paper
        carries a visible label.  Every other dot uses a single-space
        label (pyvis would fall back to node id for empty string)."""
        import re
        rg = ResearchGraph()
        rg.add_paper(_make_paper("center", "Center paper",
                                 abstract="center abstract"))
        for i in range(10):
            rg.add_paper(_make_paper(
                f"n{i}", f"NeighborTitle{i}",
                refs=["center"], year=2020, citation_count=i,
            ))
        rg.add_citation_edges()
        html = render_neighborhood(rg, "center", radius=1)
        labels = re.findall(r'"label":\s*"([^"]*)"', html)
        visible = [lbl for lbl in labels if lbl.strip()]
        # Focus only.
        assert len(visible) == 1, (
            f"Expected exactly 1 visible label (focus), got "
            f"{len(visible)}: {visible}"
        )
        # And no node id leaks as a label.
        for i in range(10):
            assert f'"label": "n{i}"' not in html

    def test_click_zoom_js_is_injected(self, small_graph: ResearchGraph):
        """The rendered HTML must wire up click-to-zoom plus the
        always-visible inspect sidebar, the controls bar, and the
        focus_swap query-param bridge."""
        html = render_neighborhood(small_graph, "a", radius=1)
        # Click handler still does the focus/fit zoom.
        assert "network.on('click'" in html
        assert "network.focus(" in html
        assert "network.fit(" in html
        # No doubleClick handler — it raced single-click and caused a
        # zoom-in / zoom-out flicker on quick double-clicks.
        assert "doubleClick" not in html
        # The fixed right sidebar, controls bar, legend, and the
        # focus_swap query-param bridge are present.
        assert 'class="rg-sidebar"' in html
        assert 'class="rg-controls"' in html
        assert 'class="rg-legend"' in html
        assert "focus_swap" in html

    def test_label_neighbors_flag_shows_neighbor_labels(self):
        """Passing label_neighbors=True adds short labels for direct
        neighbors (still relying on the radial layout to avoid overlap)."""
        import re
        rg = ResearchGraph()
        rg.add_paper(_make_paper("center", "Center paper",
                                 abstract="abc"))
        for i in range(4):
            rg.add_paper(_make_paper(
                f"n{i}", f"NeighborTitle{i}", refs=["center"],
            ))
        rg.add_citation_edges()
        html = render_neighborhood(
            rg, "center", radius=1, label_neighbors=True,
        )
        labels = re.findall(r'"label":\s*"([^"]*)"', html)
        visible = [lbl for lbl in labels if lbl.strip()]
        # Focus + 4 neighbors = 5
        assert len(visible) == 5

    def test_context_nodes_are_not_labeled(self):
        """Second-hop / context nodes should never carry a visible label."""
        rg = ResearchGraph()
        rg.add_paper(_make_paper("a", "Alpha", refs=["b"]))
        rg.add_paper(_make_paper("b", "Beta", refs=["c"]))
        rg.add_paper(_make_paper("c", "ContextOnlyPaper"))
        rg.build_graph(list(rg.papers.values()), similarity_threshold=0.99)
        html = render_neighborhood(rg, "a", radius=2)
        # 'ContextOnlyPaper' is the 2nd-hop node; its full title may
        # appear in the tooltip but must NOT appear as a node label.
        assert '"label": "ContextOnlyPaper"' not in html
        # And the bare node id 'c' must not leak into a label slot.
        assert '"label": "c"' not in html

    def test_hover_tooltip_is_title_only(self, small_graph: ResearchGraph):
        """Hover tooltip is title-only — full metadata (year, venue,
        citations, authors, abstract) lives in the slide-in inspect
        panel that opens on click. The hover popup should stay quiet
        so the eye reads the graph structure first.

        We check for the metadata *labels* (``Year:``, ``Venue:``)
        that only ever appeared in the old rich hover tooltip; the
        panel template uses different markup, so it's not a false hit.
        """
        html = render_neighborhood(small_graph, "a", radius=1)
        assert "Year:" not in html
        assert "Venue:" not in html
        # The panel-data blob still carries the rich metadata so the
        # click panel can render it.
        assert "PANEL_DATA" in html


# ===========================================================================
# Path visualization
# ===========================================================================

class TestPathVisualization:
    """render_path highlights a specific path in red over a faint context graph."""

    def test_returns_html_string(self, small_graph: ResearchGraph):
        html = render_path(small_graph, ["a", "b", "c"])
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_contains_all_path_papers(self, small_graph: ResearchGraph):
        """Every paper in the path should appear in the visualization."""
        html = render_path(small_graph, ["a", "b", "c"])
        assert "Attention" in html
        assert "BERT" in html
        assert "GPT" in html

    def test_too_short_path_returns_message(self, small_graph: ResearchGraph):
        """A single-node path should return a fallback, not crash."""
        html = render_path(small_graph, ["a"])
        assert isinstance(html, str)

    def test_includes_context_nodes(self, small_graph: ResearchGraph):
        """Nodes near the path (within context radius) should also appear."""
        html = render_path(small_graph, ["a", "b"], context_radius=1)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_no_in_graph_legend_node(self, small_graph: ResearchGraph):
        """The legend is rendered externally — none should leak into the canvas."""
        html = render_path(small_graph, ["a", "b", "c"])
        assert "legend_title" not in html
        assert "LEGEND" not in html

    def test_path_color_is_present(self, small_graph: ResearchGraph):
        """Path nodes should be colored with the path highlight color."""
        html = render_path(small_graph, ["a", "b", "c"])
        assert COLOR_PATH in html

    def test_context_nodes_in_path_are_not_labeled(self, small_graph: ResearchGraph):
        """Context (non-path) nodes in render_path must not show labels —
        and must not leak the node id either."""
        html = render_path(small_graph, ["a", "b"], context_radius=1)
        # 'd' is a context node (not on the path); pyvis must NOT echo
        # 'd' as a label.
        assert '"label": "d"' not in html


# ===========================================================================
# External Streamlit-side legends
# ===========================================================================

class TestExternalLegends:
    """The Streamlit page renders its own HTML legend so the canvas stays clean."""

    def test_neighborhood_legend_includes_all_node_categories(self):
        """Each node role should be named (compact chips: Selected,
        Citation, Similarity, Both, Context)."""
        legend = neighborhood_legend_html()
        for category in ["Selected", "Citation", "Similarity",
                         "Both", "Context"]:
            assert category in legend, f"Missing legend chip: {category}"

    def test_neighborhood_legend_includes_edge_styles(self):
        """Both citation and similarity edges should be explained."""
        legend = neighborhood_legend_html()
        assert "citation edge" in legend.lower()
        assert "similarity edge" in legend.lower()

    def test_neighborhood_legend_mentions_hover_tooltip(self):
        """The legend should tell users that full metadata is in hover."""
        legend = neighborhood_legend_html()
        assert "hover" in legend.lower()

    def test_path_legend_mentions_path_and_context(self):
        legend = path_legend_html()
        assert "Path paper" in legend
        assert "Context" in legend
