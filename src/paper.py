"""Paper class representing a single academic paper node in the research graph.

Each Paper is a node in the ResearchGraph.  Its attributes support three
concerns: graph construction (references link papers), text similarity
(title + abstract power TF-IDF edges), and display (authors, venue, year
appear in the Streamlit UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Paper:
    """A single academic paper — the fundamental node type in the knowledge graph.

    Identity semantics:
        Two Paper objects are equal (and hash the same) when they share
        the same ``paper_id``, regardless of other attributes.  This
        lets us safely deduplicate papers loaded from different sources.
    """

    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    citation_count: int = 0
    references: list[str] = field(default_factory=list)
    url: str = ""

    def __post_init__(self) -> None:
        """Validate and coerce fields after dataclass initialization."""
        if not self.paper_id:
            raise ValueError("paper_id must be a non-empty string")
        if not self.title:
            raise ValueError("title must be a non-empty string")
        # Coerce None → safe defaults so downstream code never needs null-checks
        if self.citation_count is None:
            self.citation_count = 0
        if self.citation_count < 0:
            self.citation_count = 0
        if self.abstract is None:
            self.abstract = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Paper to a JSON-friendly dict (round-trips with ``from_dict``)."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "authors": list(self.authors),
            "venue": self.venue,
            "citation_count": self.citation_count,
            "references": list(self.references),
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Construct a Paper from a JSON-friendly dict.

        Accepts both the canonical ``to_dict`` form and the
        Semantic-Scholar-style camelCase form (``paperId``,
        ``citationCount``) so cached API responses load directly.
        """
        paper_id = data.get("paper_id") or data.get("paperId") or ""
        citation_count = data.get("citation_count")
        if citation_count is None:
            citation_count = data.get("citationCount", 0) or 0

        authors_raw = data.get("authors", []) or []
        authors: list[str] = []
        for a in authors_raw:
            if isinstance(a, dict):
                authors.append(a.get("name", "Unknown"))
            else:
                authors.append(str(a))

        refs_raw = data.get("references", []) or []
        references: list[str] = []
        for r in refs_raw:
            if isinstance(r, dict):
                rid = r.get("paperId") or r.get("paper_id") or ""
                if rid:
                    references.append(rid)
            elif isinstance(r, str) and r:
                references.append(r)

        return cls(
            paper_id=paper_id,
            title=data.get("title", "") or "Untitled",
            abstract=data.get("abstract") or "",
            year=data.get("year"),
            authors=authors,
            venue=data.get("venue", "") or "",
            citation_count=citation_count,
            references=references,
            url=data.get("url", "") or "",
        )

    def display_authors(self, max_authors: int = 3) -> str:
        """Return a human-readable author string, truncated with '+ N more'.

        Used by the Streamlit UI; isolated here so it can be unit-tested
        without booting the app.
        """
        if not self.authors:
            return "Unknown"
        if len(self.authors) <= max_authors:
            return ", ".join(self.authors)
        head = ", ".join(self.authors[:max_authors])
        return f"{head} + {len(self.authors) - max_authors} more"

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary_dict(self) -> dict[str, Any]:
        """Return a dictionary summary suitable for display in the UI."""
        return {
            "Title": self.title,
            "Year": self.year,
            "Authors": ", ".join(self.authors) if self.authors else "Unknown",
            "Venue": self.venue or "N/A",
            "Citations": self.citation_count,
            "URL": self.url,
        }

    def short_label(self) -> str:
        """Return a compact label for graph visualization.

        Format: ``LastName (Year): Truncated title...``
        """
        if self.authors:
            # Take last whitespace-separated token as surname
            first_author = self.authors[0].split()[-1]
        else:
            first_author = "Unknown"
        year_str = str(self.year) if self.year else "?"
        max_title = 40
        short_title = self.title[:max_title] + ("..." if len(self.title) > max_title else "")
        return f"{first_author} ({year_str}): {short_title}"

    # ------------------------------------------------------------------
    # Similarity / analysis helpers
    # ------------------------------------------------------------------

    def similarity_features(self) -> str:
        """Return combined text used for computing TF-IDF similarity.

        Only title and abstract are used — venue and author names are
        excluded to keep similarity focused on intellectual content.
        """
        return f"{self.title} {self.abstract}"

    def topic_words(self, top_n: int = 5) -> list[str]:
        """Extract the most salient lowercase words from title + abstract.

        Uses a simple frequency heuristic after removing common stop words.
        Useful for quick cluster theme summaries without pulling in sklearn.
        """
        _STOP = frozenset({
            "a", "an", "the", "of", "for", "and", "in", "on", "with", "to",
            "from", "by", "is", "are", "was", "were", "be", "been", "being",
            "that", "this", "these", "those", "it", "its", "we", "our",
            "which", "can", "has", "have", "had", "do", "does", "did",
            "but", "or", "not", "no", "so", "as", "at", "if", "than",
            "also", "into", "such", "more", "between", "based", "using",
        })
        words = self.similarity_features().lower().split()
        freq: dict[str, int] = {}
        for w in words:
            token = w.strip(".,;:()[]\"'")
            if len(token) > 2 and token not in _STOP and token.isalpha():
                freq[token] = freq.get(token, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:top_n]]

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash(self.paper_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Paper):
            return NotImplemented
        return self.paper_id == other.paper_id

    def __repr__(self) -> str:
        return f"Paper(id={self.paper_id!r}, title={self.title!r}, year={self.year})"
