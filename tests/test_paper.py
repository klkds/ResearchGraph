"""Behavioral specification for the Paper class.

A Paper is the fundamental node in the ResearchGraph.  These tests
document what a Paper *should do* — they serve as both verification
and specification of the data model.
"""

import pytest
from src.paper import Paper


class TestPaperConstruction:
    """A Paper can be created with minimal or full metadata."""

    def test_requires_id_and_title(self):
        """Every paper must have at least an id and a title."""
        p = Paper(paper_id="p1", title="Test Paper")
        assert p.paper_id == "p1"
        assert p.title == "Test Paper"

    def test_optional_fields_default_to_safe_values(self):
        """Omitted fields should default to empty/zero — never None."""
        p = Paper(paper_id="p1", title="Minimal")
        assert p.abstract == ""
        assert p.year is None
        assert p.authors == []
        assert p.venue == ""
        assert p.citation_count == 0
        assert p.references == []
        assert p.url == ""

    def test_accepts_full_metadata(self):
        """A paper can carry full metadata for display and analysis."""
        p = Paper(
            paper_id="42",
            title="Full Paper",
            abstract="An abstract.",
            year=2023,
            authors=["Alice", "Bob"],
            venue="NeurIPS",
            citation_count=100,
            references=["10", "11"],
            url="https://example.com",
        )
        assert p.year == 2023
        assert len(p.authors) == 2
        assert p.citation_count == 100
        assert "10" in p.references


class TestPaperValidation:
    """A Paper validates its identity fields on creation."""

    def test_rejects_empty_paper_id(self):
        """paper_id must be non-empty — it is the graph node identifier."""
        with pytest.raises(ValueError, match="paper_id"):
            Paper(paper_id="", title="T")

    def test_rejects_empty_title(self):
        """title must be non-empty — it is the primary display label."""
        with pytest.raises(ValueError, match="title"):
            Paper(paper_id="p1", title="")

    def test_negative_citation_count_coerced_to_zero(self):
        """Negative citation counts are meaningless and should be zeroed."""
        p = Paper(paper_id="p1", title="T", citation_count=-5)
        assert p.citation_count == 0

    def test_none_citation_count_coerced_to_zero(self):
        """API data sometimes has null citation counts; coerce to 0."""
        p = Paper(paper_id="p1", title="T", citation_count=None)  # type: ignore[arg-type]
        assert p.citation_count == 0

    def test_none_abstract_coerced_to_empty_string(self):
        """API data sometimes has null abstracts; coerce to empty string."""
        p = Paper(paper_id="p1", title="T", abstract=None)  # type: ignore[arg-type]
        assert p.abstract == ""


class TestPaperDisplayMethods:
    """Paper provides methods for human-readable display in the UI."""

    @pytest.fixture
    def transformer_paper(self) -> Paper:
        return Paper(
            paper_id="p1",
            title="Attention Is All You Need",
            abstract="We propose the Transformer architecture.",
            year=2017,
            authors=["Ashish Vaswani", "Noam Shazeer"],
            venue="NeurIPS",
            citation_count=95000,
        )

    def test_summary_dict_contains_all_display_fields(self, transformer_paper: Paper):
        """summary_dict should return a UI-ready dict with standard keys."""
        d = transformer_paper.summary_dict()
        expected_keys = {"Title", "Year", "Authors", "Venue", "Citations", "URL"}
        assert expected_keys == set(d.keys())

    def test_summary_dict_formats_authors_as_comma_separated(self, transformer_paper: Paper):
        """Authors should appear as a human-readable comma-separated string."""
        d = transformer_paper.summary_dict()
        assert "Vaswani" in d["Authors"]
        assert "Shazeer" in d["Authors"]

    def test_summary_dict_shows_unknown_when_no_authors(self):
        """Papers without author data should display 'Unknown'."""
        p = Paper(paper_id="x", title="T")
        assert p.summary_dict()["Authors"] == "Unknown"

    def test_short_label_includes_surname_and_year(self, transformer_paper: Paper):
        """Short labels are for graph visualization: 'Surname (Year): Title...'"""
        label = transformer_paper.short_label()
        assert "Vaswani" in label
        assert "2017" in label

    def test_short_label_truncates_long_titles(self):
        """Titles longer than 40 characters should be truncated with ellipsis."""
        p = Paper(paper_id="x", title="A" * 60, authors=["Author Name"])
        label = p.short_label()
        assert "..." in label
        assert len(label) < 80

    def test_short_label_handles_missing_authors(self):
        """Papers without authors should show 'Unknown' in the label."""
        p = Paper(paper_id="x", title="Short Title")
        assert "Unknown" in p.short_label()


class TestSimilarityFeatures:
    """Paper provides text features for TF-IDF similarity computation."""

    def test_combines_title_and_abstract(self):
        """Similarity features concatenate title and abstract for TF-IDF."""
        p = Paper(paper_id="p1", title="Deep Learning", abstract="Neural network methods")
        features = p.similarity_features()
        assert "Deep Learning" in features
        assert "Neural network" in features

    def test_works_with_empty_abstract(self):
        """Papers without abstracts should still produce similarity text."""
        p = Paper(paper_id="p1", title="Title Only")
        features = p.similarity_features()
        assert "Title Only" in features


class TestTopicWords:
    """Paper can extract salient topic words for cluster analysis."""

    def test_extracts_meaningful_words(self):
        """topic_words should return content words, not stop words."""
        p = Paper(
            paper_id="p1",
            title="Graph Neural Networks for Molecular Property Prediction",
            abstract="We propose a graph neural network architecture.",
        )
        words = p.topic_words(5)
        assert "graph" in words
        assert "neural" in words
        # Stop words should be excluded
        assert "for" not in words
        assert "the" not in words

    def test_returns_empty_for_very_short_text(self):
        """Very short titles with only stop words yield an empty list."""
        p = Paper(paper_id="p1", title="On It")
        words = p.topic_words(5)
        assert isinstance(words, list)


class TestPaperIdentity:
    """Paper identity is based on paper_id — this is critical for graph deduplication."""

    def test_same_id_means_equal(self):
        """Two Papers with the same id are equal, even with different titles."""
        p1 = Paper(paper_id="same", title="Version A")
        p2 = Paper(paper_id="same", title="Version B")
        assert p1 == p2

    def test_different_id_means_not_equal(self):
        """Two Papers with different ids are not equal, even with the same title."""
        p1 = Paper(paper_id="a", title="Same Title")
        p2 = Paper(paper_id="b", title="Same Title")
        assert p1 != p2

    def test_hash_matches_equality(self):
        """Equal Papers must hash the same (required for sets and dict keys)."""
        p1 = Paper(paper_id="same", title="A")
        p2 = Paper(paper_id="same", title="B")
        assert hash(p1) == hash(p2)

    def test_deduplication_in_sets(self):
        """Adding two Papers with the same id to a set should keep only one."""
        p1 = Paper(paper_id="1", title="A")
        p2 = Paper(paper_id="1", title="B")
        p3 = Paper(paper_id="2", title="C")
        assert len({p1, p2, p3}) == 2

    def test_not_equal_to_non_paper_types(self):
        """Comparing a Paper to a non-Paper type should return False."""
        p = Paper(paper_id="1", title="A")
        assert p != "not a paper"
        assert p != 42

    def test_repr_is_informative(self):
        """repr should include id, title, and year for debugging."""
        p = Paper(paper_id="p1", title="Test", year=2020)
        r = repr(p)
        assert "p1" in r
        assert "Test" in r
        assert "2020" in r


class TestDisplayAuthors:
    """Paper.display_authors formats authors for the Streamlit UI."""

    def test_no_authors_returns_unknown(self):
        p = Paper(paper_id="p1", title="T")
        assert p.display_authors() == "Unknown"

    def test_short_author_list_is_joined(self):
        p = Paper(paper_id="p1", title="T", authors=["Alice", "Bob"])
        assert p.display_authors() == "Alice, Bob"

    def test_long_author_list_is_truncated_with_count(self):
        p = Paper(paper_id="p1", title="T",
                  authors=["A", "B", "C", "D", "E", "F"])
        out = p.display_authors(max_authors=3)
        assert out.startswith("A, B, C")
        assert "+ 3 more" in out


class TestPaperDictRoundTrip:
    """to_dict and from_dict should round-trip a Paper without data loss."""

    def test_to_dict_contains_all_fields(self):
        p = Paper(
            paper_id="p1", title="T", abstract="abs", year=2020,
            authors=["Alice"], venue="NeurIPS", citation_count=10,
            references=["r1"], url="http://x.y",
        )
        d = p.to_dict()
        assert d["paper_id"] == "p1"
        assert d["title"] == "T"
        assert d["abstract"] == "abs"
        assert d["year"] == 2020
        assert d["authors"] == ["Alice"]
        assert d["citation_count"] == 10
        assert d["references"] == ["r1"]

    def test_round_trip_preserves_all_fields(self):
        p = Paper(
            paper_id="p1", title="T", abstract="abs", year=2020,
            authors=["Alice", "Bob"], venue="ICML", citation_count=42,
            references=["r1", "r2"], url="http://x.y",
        )
        restored = Paper.from_dict(p.to_dict())
        assert restored == p
        assert restored.title == p.title
        assert restored.abstract == p.abstract
        assert restored.year == p.year
        assert restored.authors == p.authors
        assert restored.venue == p.venue
        assert restored.citation_count == p.citation_count
        assert restored.references == p.references

    def test_from_dict_accepts_semantic_scholar_camelcase(self):
        """Cached Semantic Scholar payloads use camelCase keys."""
        raw = {
            "paperId": "p1",
            "title": "T",
            "abstract": "abs",
            "year": 2020,
            "authors": [{"name": "Alice"}],
            "venue": "NeurIPS",
            "citationCount": 50,
            "references": [{"paperId": "r1"}, {"paperId": "r2"}],
            "url": "http://x.y",
        }
        p = Paper.from_dict(raw)
        assert p.paper_id == "p1"
        assert p.authors == ["Alice"]
        assert p.references == ["r1", "r2"]
        assert p.citation_count == 50

    def test_from_dict_handles_missing_optional_fields(self):
        p = Paper.from_dict({"paper_id": "p1", "title": "T"})
        assert p.paper_id == "p1"
        assert p.year is None
        assert p.authors == []
        assert p.references == []
