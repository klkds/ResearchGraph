"""Behavioral specification for the Streamlit-free UI helpers.

These helpers live in ``src/app_helpers.py`` so that the Streamlit app
can stay thin while the formatting and rubric logic remains testable
without booting a browser.
"""

from __future__ import annotations

import pytest

from src.paper import Paper
from src.research_graph import ScoredPath
from src.app_helpers import (
    quality_badge_html,
    truncate_abstract,
    format_paper_caption,
    path_chain_text,
    trajectory_narrative,
    summarize_neighbor_counts,
    rubric_mapping_rows,
)


# ===========================================================================
# Quality badge
# ===========================================================================

class TestQualityBadge:
    """quality_badge_html maps trajectory quality labels to colored pills."""

    def test_strong_uses_green(self):
        html = quality_badge_html("Strong")
        assert "Strong" in html
        assert "#2e7d32" in html  # green

    def test_moderate_uses_orange(self):
        html = quality_badge_html("Moderate")
        assert "#e65100" in html

    def test_weak_uses_red(self):
        html = quality_badge_html("Weak")
        assert "#c62828" in html

    def test_unknown_label_falls_back_to_grey(self):
        html = quality_badge_html("???")
        assert "#616161" in html

    def test_output_is_html_span(self):
        html = quality_badge_html("Strong")
        assert html.startswith("<span") and html.endswith("</span>")


# ===========================================================================
# Abstract truncation
# ===========================================================================

class TestTruncateAbstract:
    """truncate_abstract caps long abstracts with an ellipsis."""

    def test_short_text_passes_through(self):
        assert truncate_abstract("hello", max_chars=250) == "hello"

    def test_empty_abstract_returns_empty_string(self):
        assert truncate_abstract("", max_chars=250) == ""

    def test_long_text_is_truncated_with_ellipsis(self):
        long = "a" * 300
        out = truncate_abstract(long, max_chars=250)
        assert out.endswith("...")
        assert len(out) <= 253

    def test_does_not_double_truncate_below_threshold(self):
        text = "x" * 250
        assert truncate_abstract(text, max_chars=250) == text


# ===========================================================================
# Paper caption
# ===========================================================================

class TestFormatPaperCaption:
    """format_paper_caption renders the standard one-line metadata strip."""

    def test_includes_year_venue_citations_authors(self):
        p = Paper(
            paper_id="p1", title="T", year=2017, venue="NeurIPS",
            citation_count=12345, authors=["Alice", "Bob"],
        )
        caption = format_paper_caption(p)
        assert "2017" in caption
        assert "NeurIPS" in caption
        assert "12,345" in caption
        assert "Alice" in caption

    def test_handles_missing_year_and_venue(self):
        p = Paper(paper_id="p1", title="T")
        caption = format_paper_caption(p)
        assert "?" in caption
        assert "N/A" in caption
        assert "Unknown" in caption

    def test_truncates_long_author_lists(self):
        p = Paper(
            paper_id="p1", title="T",
            authors=["A", "B", "C", "D", "E", "F"],
        )
        caption = format_paper_caption(p)
        assert "+ 3 more" in caption


# ===========================================================================
# Path chain text
# ===========================================================================

class TestPathChainText:
    """path_chain_text formats a list of papers as an arrow chain."""

    def test_single_paper_has_no_arrows(self):
        p = Paper(paper_id="p1", title="Solo", year=2020)
        out = path_chain_text([p])
        assert "->" not in out
        assert "Solo" in out

    def test_multiple_papers_are_joined_with_arrows(self):
        papers = [
            Paper(paper_id="p1", title="A", year=2017),
            Paper(paper_id="p2", title="B", year=2018),
            Paper(paper_id="p3", title="C", year=2019),
        ]
        out = path_chain_text(papers)
        assert out.count("->") == 2
        assert out.index("A") < out.index("B") < out.index("C")

    def test_includes_year_when_available(self):
        p = Paper(paper_id="p1", title="A", year=2017)
        assert "2017" in path_chain_text([p])

    def test_omits_year_when_missing(self):
        p = Paper(paper_id="p1", title="A")
        out = path_chain_text([p])
        assert "(" not in out  # no year parenthesis


# ===========================================================================
# Trajectory narrative
# ===========================================================================

def _scored(papers: list[Paper], score: float = 0.5) -> ScoredPath:
    return ScoredPath(
        papers=papers, score=score, label="Moderate",
        avg_similarity=0.4, citation_strength=0.5, length=len(papers),
    )


class TestTrajectoryNarrative:
    """trajectory_narrative summarizes a scored path in one sentence."""

    def test_two_paper_path_uses_direct_phrasing(self):
        papers = [
            Paper(paper_id="a", title="Attention", year=2017,
                  abstract="Transformers attention sequence"),
            Paper(paper_id="b", title="Diffusion", year=2020,
                  abstract="Diffusion image generation models"),
        ]
        narrative = trajectory_narrative(_scored(papers))
        assert "direct" in narrative.lower() or "structurally" in narrative.lower()

    def test_three_paper_path_mentions_hops(self):
        papers = [
            Paper(paper_id="a", title="Attention", year=2017,
                  abstract="transformer attention sequence"),
            Paper(paper_id="b", title="BERT", year=2019,
                  abstract="bidirectional transformer pretraining"),
            Paper(paper_id="c", title="Diffusion", year=2020,
                  abstract="diffusion image generation"),
        ]
        narrative = trajectory_narrative(_scored(papers))
        assert "2-hop" in narrative

    def test_includes_topic_words_from_endpoints(self):
        papers = [
            Paper(paper_id="a", title="Quantum mechanics fundamentals", year=2010,
                  abstract="quantum quantum quantum mechanics"),
            Paper(paper_id="b", title="Machine learning models", year=2020,
                  abstract="machine learning learning models"),
        ]
        narrative = trajectory_narrative(_scored(papers))
        # At least one endpoint topic word should appear in the narrative
        assert "quantum" in narrative.lower() or "machine" in narrative.lower() \
            or "learning" in narrative.lower()


# ===========================================================================
# Neighbor count bucketing
# ===========================================================================

class TestSummarizeNeighborCounts:
    """summarize_neighbor_counts buckets neighbors by edge type."""

    def test_counts_each_type(self):
        neighbors = [
            {"edge_type": "citation"},
            {"edge_type": "similarity"},
            {"edge_type": "both"},
            {"edge_type": "shared_author"},
        ]
        counts = summarize_neighbor_counts(neighbors)
        # 'both' counts toward both citation and similarity
        assert counts["citation"] == 2
        assert counts["similarity"] == 2
        assert counts["shared_author"] == 1

    def test_unknown_edge_type_lands_in_other(self):
        neighbors = [{"edge_type": "weird"}]
        counts = summarize_neighbor_counts(neighbors)
        assert counts["other"] == 1

    def test_empty_neighbors_returns_zero_counts(self):
        counts = summarize_neighbor_counts([])
        assert counts == {"citation": 0, "similarity": 0,
                          "shared_author": 0, "other": 0}


# ===========================================================================
# Rubric mapping content
# ===========================================================================

class TestRubricMapping:
    """rubric_mapping_rows is the single source of truth for the SI 507 mapping."""

    def test_returns_seven_rubric_rows(self):
        rows = rubric_mapping_rows()
        assert len(rows) == 7

    def test_all_rows_are_pairs_of_strings(self):
        for row in rubric_mapping_rows():
            assert isinstance(row, tuple) and len(row) == 2
            requirement, satisfaction = row
            assert isinstance(requirement, str) and requirement
            assert isinstance(satisfaction, str) and satisfaction

    def test_covers_all_si507_requirement_categories(self):
        joined = " ".join(r.lower() for r, _ in rubric_mapping_rows())
        for must_have in ["graph", "object-oriented", "real-world",
                          "interaction", "interface", "testing", "ambition"]:
            assert must_have in joined, f"Missing rubric category: {must_have}"
