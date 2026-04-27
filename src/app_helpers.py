"""Pure helpers used by the Streamlit UI (no Streamlit imports).

Anything in here must be safely importable in tests without booting
Streamlit.  Extracted from ``app.py`` so the helpers can be unit-tested
and reused by alternate frontends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.paper import Paper
    from src.research_graph import ScoredPath


_QUALITY_COLORS: dict[str, str] = {
    "Strong": "#2e7d32",
    "Moderate": "#e65100",
    "Weak": "#c62828",
}


def quality_badge_html(label: str) -> str:
    """Return an inline-HTML pill colored by trajectory quality label."""
    color = _QUALITY_COLORS.get(label, "#616161")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def truncate_abstract(abstract: str, max_chars: int = 250) -> str:
    """Truncate an abstract for card display, never mid-ellipsis."""
    if not abstract:
        return ""
    if len(abstract) <= max_chars:
        return abstract
    return abstract[:max_chars].rstrip() + "..."


def format_paper_caption(paper: Paper) -> str:
    """Render a one-line metadata caption for a paper card.

    Format: ``YEAR • VENUE • CITATIONS • AUTHORS``.
    """
    year = paper.year or "?"
    venue = paper.venue or "N/A"
    citations = f"{paper.citation_count:,} citations"
    authors = paper.display_authors(max_authors=3)
    return f"{year} • {venue} • {citations} • {authors}"


def path_chain_text(papers: list[Paper]) -> str:
    """Render a path of papers as a plain-text arrow chain.

    Used in tests and for non-HTML environments.  Each paper appears as
    ``Title (Year)``.
    """
    parts: list[str] = []
    for p in papers:
        year = f" ({p.year})" if p.year else ""
        parts.append(f"{p.title}{year}")
    return " -> ".join(parts)


def trajectory_narrative(scored: ScoredPath) -> str:
    """One-sentence summary of *what* a scored trajectory reveals.

    Mirrors the language used in the Streamlit UI but is fully testable
    in isolation.
    """
    first = scored.papers[0]
    last = scored.papers[-1]
    first_topics = ", ".join(first.topic_words(3)) or "its foundational concepts"
    last_topics = ", ".join(last.topic_words(3)) or "its advanced concepts"

    if scored.length > 2:
        mid = scored.papers[scored.length // 2]
        mid_topics = ", ".join(mid.topic_words(2)) or "bridging ideas"
        if first.year and last.year and first.year != last.year:
            span = abs(last.year - first.year)
            timespan = f"{span} years of"
        elif first.year and last.year:
            timespan = "the same year of"
        else:
            timespan = ""
        timespan_str = f" across {timespan} research evolution" if timespan else ""
        return (
            f"This trajectory reveals how research on {first_topics} "
            f"evolves into {last_topics} through {mid_topics} — "
            f"a {scored.length - 1}-hop intellectual journey{timespan_str}."
        )
    return (
        f"This trajectory traces a direct connection from {first_topics} "
        f"to {last_topics}, revealing how these ideas are structurally "
        f"linked in the knowledge graph."
    )


def summarize_neighbor_counts(neighbors: list[dict]) -> dict[str, int]:
    """Bucket neighbor entries by edge type for quick UI badges."""
    counts = {"citation": 0, "similarity": 0, "shared_author": 0, "other": 0}
    for n in neighbors:
        etype = n.get("edge_type", "other")
        if etype in ("citation", "both"):
            counts["citation"] += 1
        if etype in ("similarity", "both"):
            counts["similarity"] += 1
        if etype == "shared_author":
            counts["shared_author"] += 1
        if etype not in ("citation", "similarity", "both", "shared_author"):
            counts["other"] += 1
    return counts


def rubric_mapping_rows() -> list[tuple[str, str]]:
    """Return the (requirement, satisfaction) rows for the SI 507 rubric.

    Centralized here so the README, the in-app overview page, and the
    final write-up all show the same content (and tests can verify it).
    """
    return [
        (
            "Graph / tree structure",
            "Papers are nodes; edges encode citation, TF-IDF similarity, "
            "and shared authorship. Every interactive feature reads from "
            "the NetworkX graph.",
        ),
        (
            "Object-oriented design",
            "Four core classes — Paper, DataLoader, ResearchGraph, "
            "IdeaEngine — with clean separation of concerns and full "
            "docstrings.",
        ),
        (
            "Real-world data",
            "Live Semantic Scholar API queries plus a bundled 50-paper "
            "ML/AI dataset for offline use; both are normalized through "
            "the same DataLoader pipeline.",
        ),
        (
            "Four interaction modes",
            "Search, Graph Explorer, Path Finder / Trajectory, "
            "Insights & Rankings, Idea Generator — five distinct modes "
            "plus sub-tabs for hubs, bridges, clusters, and surprises.",
        ),
        (
            "Interface",
            "Streamlit app with sidebar navigation, interactive pyvis "
            "graph visualizations, and a CLI-friendly test suite.",
        ),
        (
            "Testing",
            "150+ pytest tests covering Paper, DataLoader, "
            "ResearchGraph, IdeaEngine, graph viz, and app helpers — "
            "behavior-named so the suite reads as a specification.",
        ),
        (
            "Ambition & insight",
            "The graph itself reveals bridge papers, surprising "
            "non-cited similarities, multi-hop research trajectories, "
            "and clusters that flat citation lists cannot expose.",
        ),
    ]
