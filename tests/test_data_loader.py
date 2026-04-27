"""Behavioral specification for the DataLoader class.

DataLoader is the ingestion layer: it converts raw data from any source
(Semantic Scholar API, cached JSON, bundled sample dataset) into clean
Paper objects ready for graph construction.  These tests document the
normalization guarantees and caching behavior.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.data_loader import DataLoader
from src.paper import Paper


class TestRecordNormalization:
    """DataLoader normalizes raw records from diverse sources into Papers.

    Real-world data is messy: fields may be missing, null, or formatted
    differently across sources.  The normalizer must handle all of these
    gracefully and produce valid Paper objects every time.
    """

    def test_complete_record_normalizes_cleanly(self):
        """A well-formed record should produce a Paper with all fields set."""
        raw = {
            "paperId": "abc",
            "title": "  Test Paper  ",
            "abstract": "An abstract with  extra   spaces.",
            "year": 2022,
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "venue": "ICML",
            "citationCount": 50,
            "references": [{"paperId": "ref1"}, {"paperId": "ref2"}],
            "url": "https://example.com",
        }
        loader = DataLoader()
        paper = loader.normalize_paper_record(raw)
        assert paper.paper_id == "abc"
        assert paper.title == "Test Paper"  # whitespace trimmed
        assert "extra spaces" in paper.abstract  # internal whitespace collapsed
        assert paper.authors == ["Alice", "Bob"]
        assert paper.references == ["ref1", "ref2"]

    def test_missing_optional_fields_get_safe_defaults(self):
        """A record with only id and title should still produce a valid Paper."""
        raw = {"paperId": "x", "title": "Minimal"}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.abstract == ""
        assert paper.year is None
        assert paper.authors == []
        assert paper.citation_count == 0

    def test_null_abstract_becomes_empty_string(self):
        """Semantic Scholar returns null for papers without abstracts."""
        raw = {"paperId": "x", "title": "T", "abstract": None}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.abstract == ""

    def test_null_citation_count_becomes_zero(self):
        """Some API records have citationCount: null."""
        raw = {"paperId": "x", "title": "T", "citationCount": None}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.citation_count == 0

    def test_string_references_accepted(self):
        """References may arrive as plain strings instead of nested dicts."""
        raw = {"paperId": "x", "title": "T", "references": ["r1", "r2"]}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.references == ["r1", "r2"]

    def test_empty_reference_ids_are_filtered_out(self):
        """References with empty paperId should be silently dropped."""
        raw = {"paperId": "x", "title": "T", "references": [{"paperId": ""}, {"paperId": "r1"}]}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.references == ["r1"]

    def test_empty_string_references_are_filtered_out(self):
        """Empty-string references in list form should be dropped."""
        raw = {"paperId": "x", "title": "T", "references": ["", "r1"]}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.references == ["r1"]


class TestNullListFieldsFromAPI:
    """Semantic Scholar can return ``null`` (not absent) for any list field.

    The earlier implementation crashed with ``TypeError: 'NoneType' object
    is not iterable`` because ``raw.get("references", [])`` only defaults
    when the key is *missing* — when the key is *present with value None*
    it returns None.  The normalizer must defend against this for every
    iterable field.
    """

    def test_none_references_does_not_raise(self):
        raw = {"paperId": "x", "title": "T", "references": None}
        # Must not raise TypeError
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.references == []

    def test_none_authors_does_not_raise(self):
        raw = {"paperId": "x", "title": "T", "authors": None}
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.authors == []

    def test_none_fields_of_study_does_not_raise(self):
        raw = {"paperId": "x", "title": "T", "fieldsOfStudy": None}
        paper = DataLoader().normalize_paper_record(raw)
        # No crash, and topic_words still returns a list (Paper derives
        # topics from title+abstract, not from the API field).
        assert isinstance(paper.topic_words(), list)

    def test_none_s2_fields_of_study_does_not_raise(self):
        raw = {"paperId": "x", "title": "T", "s2FieldsOfStudy": None}
        paper = DataLoader().normalize_paper_record(raw)
        assert isinstance(paper.topic_words(), list)

    def test_all_list_fields_none_simultaneously(self):
        """A real-world worst-case record where every list field is null."""
        raw = {
            "paperId": "x",
            "title": "Worst case paper",
            "abstract": None,
            "year": None,
            "authors": None,
            "venue": None,
            "citationCount": None,
            "references": None,
            "fieldsOfStudy": None,
            "s2FieldsOfStudy": None,
            "url": None,
        }
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.paper_id == "x"
        assert paper.title == "Worst case paper"
        assert paper.references == []
        assert paper.authors == []
        assert paper.citation_count == 0
        assert paper.abstract == ""
        assert paper.venue == ""
        assert isinstance(paper.topic_words(), list)

    def test_none_items_inside_authors_list_are_skipped(self):
        """Some API rows return [None, {"name": "X"}] inside authors."""
        raw = {
            "paperId": "x", "title": "T",
            "authors": [None, {"name": "Alice"}, None],
        }
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.authors == ["Alice"]

    def test_none_items_inside_references_list_are_skipped(self):
        raw = {
            "paperId": "x", "title": "T",
            "references": [None, {"paperId": "r1"}, None, "r2"],
        }
        paper = DataLoader().normalize_paper_record(raw)
        assert paper.references == ["r1", "r2"]


class TestBundledDataset:
    """The bundled sample dataset provides a realistic fallback for offline use.

    It must contain enough papers with correct structure to build a
    meaningful graph and exercise all features of the application.
    """

    @pytest.fixture
    def papers(self) -> list[Paper]:
        return DataLoader().load_local_dataset()

    def test_contains_at_least_25_papers(self, papers: list[Paper]):
        """The dataset must be large enough to demonstrate graph features."""
        assert len(papers) >= 25

    def test_all_entries_are_paper_objects(self, papers: list[Paper]):
        assert all(isinstance(p, Paper) for p in papers)

    def test_every_paper_has_a_unique_id(self, papers: list[Paper]):
        """Unique ids are required for graph node identity."""
        ids = [p.paper_id for p in papers]
        assert all(pid for pid in ids)
        assert len(set(ids)) == len(ids)

    def test_every_paper_has_a_title(self, papers: list[Paper]):
        assert all(p.title for p in papers)

    def test_some_papers_have_references(self, papers: list[Paper]):
        """Without references, no citation edges can be built."""
        papers_with_refs = [p for p in papers if p.references]
        assert len(papers_with_refs) >= 10

    def test_papers_span_multiple_years(self, papers: list[Paper]):
        """Year diversity is needed for the Research Trajectory feature."""
        years = {p.year for p in papers if p.year}
        assert len(years) >= 5


class TestCaching:
    """DataLoader caches API results to disk to avoid redundant requests.

    The cache must round-trip Paper objects faithfully: saving and loading
    should produce identical data.
    """

    def test_cache_round_trip_preserves_data(self):
        """Papers saved to cache should load back with the same attributes."""
        loader = DataLoader()
        papers = [
            Paper(paper_id="c1", title="Cached Paper", year=2021, authors=["Test"]),
            Paper(paper_id="c2", title="Another", abstract="Abstract text"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"
            loader.save_cache(papers, cache_path)
            loaded = loader.load_cache(cache_path)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].title == "Cached Paper"
        assert loaded[0].year == 2021
        assert loaded[1].abstract == "Abstract text"

    def test_nonexistent_cache_returns_none(self):
        """Missing cache file should return None, not raise an error."""
        result = DataLoader().load_cache(Path("/nonexistent/path.json"))
        assert result is None

    def test_corrupt_cache_returns_none(self):
        """Corrupt JSON should return None so the caller falls back to API."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("this is not json{{{")
            path = Path(f.name)
        result = DataLoader().load_cache(path)
        assert result is None
        path.unlink()


class TestAPIFallback:
    """DataLoader falls back to the bundled dataset when the API fails.

    This ensures the project always works, even without network access.
    """

    @patch("src.data_loader.DataLoader._api_search")
    def test_falls_back_on_network_error(self, mock_api: MagicMock):
        """A network error should trigger fallback to local data."""
        import requests
        mock_api.side_effect = requests.ConnectionError("no network")
        loader = DataLoader()
        papers = loader.fetch_from_semantic_scholar("test query")
        # Should return local dataset instead of crashing
        assert len(papers) >= 25
        assert all(isinstance(p, Paper) for p in papers)

    @patch("src.data_loader.DataLoader._api_search")
    def test_falls_back_on_timeout(self, mock_api: MagicMock):
        """A timeout should trigger fallback, not crash the application."""
        import requests
        mock_api.side_effect = requests.Timeout("timed out")
        loader = DataLoader()
        papers = loader.fetch_from_semantic_scholar("test query")
        assert len(papers) >= 25
