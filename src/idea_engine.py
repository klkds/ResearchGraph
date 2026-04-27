"""IdeaEngine: structured explanation and idea generation from graph outputs.

This module sits *on top* of graph results — it does not replace graph logic.
It takes Paper lists (paths, clusters, trajectories) produced by
:class:`ResearchGraph` and turns them into structured, human-readable text.

Every public method produces a Markdown string with:
    - Step-by-step concept transfer explanations for consecutive pairs
    - An overall insight summary
    - Structured formatting suitable for Streamlit rendering

Pluggable design:
    - If ``OPENAI_API_KEY`` is set, calls the OpenAI API for rich text.
    - Otherwise, deterministic template-based outputs work fully.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.paper import Paper

logger = logging.getLogger(__name__)


def _llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _call_llm(prompt: str) -> str | None:
    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("LLM call failed (%s) — falling back to templates", exc)
        return None


def _identify_concept_transfer(paper_a: Paper, paper_b: Paper) -> str:
    """Identify what conceptual thread connects two consecutive papers.

    Uses shared topic words and metadata to produce a one-sentence
    explanation of the intellectual link.  This is the core of the
    structured explanation layer.
    """
    words_a = set(paper_a.topic_words(10))
    words_b = set(paper_b.topic_words(10))
    shared = words_a & words_b
    unique_to_b = words_b - words_a

    if shared and unique_to_b:
        return (
            f"Both papers address **{', '.join(list(shared)[:3])}**. "
            f"The second paper extends into **{', '.join(list(unique_to_b)[:2])}**, "
            f"building on the foundation established by the first."
        )
    elif shared:
        return (
            f"Connected through shared focus on **{', '.join(list(shared)[:3])}**. "
            f"The later work deepens or applies the concepts from the earlier paper."
        )
    else:
        # No direct word overlap — look for structural connection
        if paper_a.venue and paper_a.venue == paper_b.venue:
            return (
                f"Both published at **{paper_a.venue}**, suggesting "
                f"they belong to the same research community, though they "
                f"approach the topic from different angles."
            )
        return (
            f"These papers are connected in the knowledge graph through "
            f"citation or latent similarity — representing a conceptual "
            f"bridge between distinct research threads."
        )


class IdeaEngine:
    """Generate structured explanations and research ideas from graph paths.

    Every public method accepts Paper lists and returns Markdown.
    The engine never queries the graph directly.
    """

    def __init__(self, use_llm: bool | None = None) -> None:
        if use_llm is None:
            self.use_llm = _llm_available()
        else:
            self.use_llm = use_llm

    # ------------------------------------------------------------------
    # Structured path explanation
    # ------------------------------------------------------------------

    def explain_path(self, path_papers: list[Paper]) -> str:
        """Produce a structured explanation of a paper path.

        Output format:
            Step 1 → Step 2: concept transfer explanation
            Step 2 → Step 3: concept transfer explanation
            ...
            Overall insight: summary of the full intellectual trajectory
        """
        if not path_papers:
            return "No papers provided."
        if len(path_papers) == 1:
            return f"Single paper: **{path_papers[0].title}** — no connections to explain."

        if self.use_llm:
            result = self._llm_explain_path(path_papers)
            if result is not None:
                return result
        return self._template_explain_path(path_papers)

    def _template_explain_path(self, papers: list[Paper]) -> str:
        lines: list[str] = []

        # Header
        lines.append(f"### Research Path ({len(papers)} papers, {len(papers)-1} steps)\n")

        # List all papers
        for i, p in enumerate(papers):
            role = "START" if i == 0 else ("END" if i == len(papers) - 1 else f"Step {i}")
            year_s = f" ({p.year})" if p.year else ""
            lines.append(f"**[{role}]** {p.title}{year_s}")

        # Step-by-step concept transfer
        lines.append("\n---\n### Step-by-Step Connections\n")
        for i in range(len(papers) - 1):
            a, b = papers[i], papers[i + 1]
            transfer = _identify_concept_transfer(a, b)
            lines.append(f"**{a.title}** → **{b.title}**")
            lines.append(f"> {transfer}\n")

        # Overall insight
        first, last = papers[0], papers[-1]
        first_topics = ", ".join(first.topic_words(3)) or "its core concepts"
        last_topics = ", ".join(last.topic_words(3)) or "its core concepts"

        lines.append("---\n### Overall Insight\n")

        if first.year and last.year and first.year != last.year:
            span = abs(last.year - first.year)
            lines.append(
                f"This path spans **{span} years** of research evolution, "
                f"from **{first.title}** ({first.year}) to "
                f"**{last.title}** ({last.year})."
            )
        else:
            lines.append(
                f"This path connects **{first.title}** to **{last.title}**."
            )

        lines.append(
            f"The intellectual trajectory moves from {first_topics} "
            f"toward {last_topics}, with each intermediate paper "
            f"serving as a conceptual stepping stone that transfers "
            f"key ideas between research areas."
        )

        if len(papers) > 2:
            mid = papers[len(papers) // 2]
            lines.append(
                f"\nThe pivotal paper in this path is **{mid.title}** — "
                f"it bridges the conceptual gap between the start and "
                f"end points of this research trajectory."
            )

        return "\n".join(lines)

    def _llm_explain_path(self, papers: list[Paper]) -> str | None:
        descs = [
            f"{i+1}. {p.title} ({p.year}): {p.abstract[:150]}"
            for i, p in enumerate(papers)
        ]
        prompt = (
            "You are an academic research advisor. Explain how these papers "
            "are intellectually connected.\n\n"
            "Papers:\n" + "\n".join(descs) + "\n\n"
            "For each consecutive pair, write:\n"
            "**Paper A** → **Paper B**: <one sentence explaining what "
            "concept or technique transfers between them>\n\n"
            "Then write:\n"
            "**Overall insight:** <2-3 sentences summarizing the full "
            "intellectual trajectory and why this path is meaningful>"
        )
        return _call_llm(prompt)

    # ------------------------------------------------------------------
    # Research idea generation
    # ------------------------------------------------------------------

    def generate_research_idea(self, path_papers: list[Paper]) -> str:
        if len(path_papers) < 2:
            return "Need at least two papers to generate a cross-cutting research idea."
        if self.use_llm:
            result = self._llm_generate_idea(path_papers)
            if result is not None:
                return result
        return self._template_generate_idea(path_papers)

    def _template_generate_idea(self, papers: list[Paper]) -> str:
        first, last = papers[0], papers[-1]
        mid = papers[len(papers) // 2] if len(papers) > 2 else None
        first_topics = ", ".join(first.topic_words(3))
        last_topics = ", ".join(last.topic_words(3))

        idea = "### Research Idea\n\n"
        idea += f"**Combining insights from:** {first.title} and {last.title}\n\n"
        if mid:
            idea += f"**Bridged through:** {mid.title}\n\n"

        idea += (
            f"**Proposed direction:** Investigate how the techniques or findings "
            f"from *{first.title}* (focusing on {first_topics}) could be adapted "
            f"or extended using the framework of *{last.title}* "
            f"(focusing on {last_topics}). "
        )
        if mid:
            idea += (
                f"The work on *{mid.title}* suggests a methodological bridge "
                f"that could enable this cross-pollination. "
            )

        idea += (
            f"\n\n**Why this is interesting:** These papers are connected in the "
            f"research graph but come from "
            f"{'different time periods' if first.year != last.year else 'related contexts'}, "
            f"suggesting an under-explored intersection. "
        )
        if first.year and last.year:
            span = abs(last.year - first.year)
            if span > 0:
                idea += (
                    f"The {span}-year gap means the later work may not have "
                    f"fully incorporated the earlier insights. "
                )
        idea += (
            f"A project combining "
            f"{'the ' + str(first.year) + ' approach' if first.year else 'the earlier approach'} "
            f"with {'the ' + str(last.year) + ' advances' if last.year else 'recent advances'} "
            f"could yield novel results."
        )
        return idea

    def _llm_generate_idea(self, papers: list[Paper]) -> str | None:
        descs = [f"- {p.title} ({p.year}): {p.abstract[:200]}" for p in papers]
        prompt = (
            "You are a creative research advisor. Given this path of connected "
            "academic papers, propose ONE novel research idea that bridges "
            "concepts from the first and last papers.\n\n"
            "Papers:\n" + "\n".join(descs) + "\n\n"
            "Format:\n"
            "**Title:** <proposed research title>\n"
            "**Key insight:** <one sentence>\n"
            "**Approach:** <2-3 sentences>\n"
            "**Why it matters:** <1-2 sentences>"
        )
        return _call_llm(prompt)

    # ------------------------------------------------------------------
    # Research trajectory narrative
    # ------------------------------------------------------------------

    def narrate_trajectory(self, papers: list[Paper]) -> str:
        if len(papers) < 2:
            return "Need at least two papers to narrate a trajectory."
        if self.use_llm:
            result = self._llm_narrate_trajectory(papers)
            if result is not None:
                return result
        return self._template_narrate_trajectory(papers)

    def _template_narrate_trajectory(self, papers: list[Paper]) -> str:
        sorted_papers = sorted(
            [p for p in papers if p.year], key=lambda p: p.year  # type: ignore[arg-type]
        )
        if len(sorted_papers) < 2:
            return "Not enough dated papers to build a trajectory."

        first_year = sorted_papers[0].year
        last_year = sorted_papers[-1].year
        span = last_year - first_year if first_year and last_year else 0

        lines = [
            f"### Research Trajectory ({first_year}–{last_year})\n",
            f"This trajectory spans **{span} years** and "
            f"**{len(sorted_papers)} papers**.\n",
        ]

        # Step-by-step evolution
        lines.append("### How the field evolved\n")
        for i in range(len(sorted_papers) - 1):
            a, b = sorted_papers[i], sorted_papers[i + 1]
            transfer = _identify_concept_transfer(a, b)
            year_gap = (b.year - a.year) if a.year and b.year else 0
            gap_str = f" ({year_gap} year{'s' if year_gap != 1 else ''} later)" if year_gap > 0 else ""
            lines.append(f"**{a.title}** ({a.year}) → **{b.title}** ({b.year}){gap_str}")
            lines.append(f"> {transfer}\n")

        # Summary
        total_citations = sum(p.citation_count for p in sorted_papers)
        most_cited = max(sorted_papers, key=lambda p: p.citation_count)
        lines.append("---\n### Summary\n")
        lines.append(
            f"**Landmark paper:** {most_cited.title} "
            f"({most_cited.citation_count:,} citations)"
        )
        lines.append(f"**Collective impact:** {total_citations:,} total citations.")

        return "\n".join(lines)

    def _llm_narrate_trajectory(self, papers: list[Paper]) -> str | None:
        descs = [
            f"- {p.title} ({p.year}, {p.citation_count:,} citations): "
            f"{p.abstract[:120]}"
            for p in sorted(papers, key=lambda p: p.year or 0)
        ]
        prompt = (
            "You are a science historian. Narrate how this research area "
            "evolved based on these chronologically sorted papers.\n\n"
            "For each consecutive pair, explain what conceptual advance "
            "or technique transfer occurred.\n\n"
            "Papers:\n" + "\n".join(descs) + "\n\n"
            "End with a 2-sentence overall insight."
        )
        return _call_llm(prompt)

    # ------------------------------------------------------------------
    # Cluster summary
    # ------------------------------------------------------------------

    def summarize_cluster(self, papers: list[Paper]) -> str:
        if not papers:
            return "No papers in cluster."
        if self.use_llm:
            result = self._llm_summarize_cluster(papers)
            if result is not None:
                return result
        return self._template_summarize_cluster(papers)

    def _template_summarize_cluster(self, papers: list[Paper]) -> str:
        venues = set(p.venue for p in papers if p.venue)
        years = [p.year for p in papers if p.year]
        year_range = f"{min(years)}–{max(years)}" if years else "unknown"
        total_citations = sum(p.citation_count for p in papers)

        lines = [
            f"### Cluster Summary ({len(papers)} papers)\n",
            f"**Year range:** {year_range}",
            f"**Total citations:** {total_citations:,}",
            f"**Venues:** {', '.join(sorted(venues)) if venues else 'N/A'}\n",
            "**Papers in this cluster:**",
        ]
        for p in sorted(papers, key=lambda x: x.citation_count, reverse=True):
            lines.append(f"- {p.title} ({p.year}) — {p.citation_count:,} citations")

        all_topics: dict[str, int] = {}
        for p in papers:
            for word in p.topic_words(5):
                all_topics[word] = all_topics.get(word, 0) + 1
        top_themes = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_themes:
            themes = ", ".join(w for w, _ in top_themes)
            lines.append(f"\n**Common themes:** {themes}")

        return "\n".join(lines)

    def _llm_summarize_cluster(self, papers: list[Paper]) -> str | None:
        titles = [f"- {p.title} ({p.year})" for p in papers]
        prompt = (
            "Summarize this cluster of related academic papers in 3-4 sentences. "
            "Identify the common theme and notable contributions.\n\n"
            "Papers:\n" + "\n".join(titles)
        )
        return _call_llm(prompt)

    # ------------------------------------------------------------------
    # Idea generation directly from a paper or a topic
    # ------------------------------------------------------------------

    def generate_ideas_from_paper(
        self,
        center: Paper,
        related: list[Paper],
        n_ideas: int = 3,
    ) -> list[str]:
        """Produce ``n_ideas`` short research suggestions seeded from one paper.

        Pairs the center paper with each related paper to surface a
        concrete cross-pollination opportunity.  Returns Markdown strings
        — one per idea — explaining the proposed direction and the
        graph relationship that motivated it.
        """
        if not center:
            return []
        if not related:
            return [
                f"**{center.title}** has no graph neighbors yet — "
                f"try lowering the similarity threshold or expanding the dataset "
                f"to surface adjacent research."
            ]

        ideas: list[str] = []
        center_topics = ", ".join(center.topic_words(3)) or "its core ideas"
        for partner in related[:n_ideas]:
            partner_topics = ", ".join(partner.topic_words(3)) or "its core ideas"
            year_gap = ""
            if center.year and partner.year and center.year != partner.year:
                year_gap = (
                    f" The {abs(center.year - partner.year)}-year gap between "
                    f"these works hints at concepts that may not have fully "
                    f"transferred yet."
                )
            ideas.append(
                f"**Bridge {center.title} with {partner.title}.** "
                f"Combine the {center_topics} angle of *{center.title}* "
                f"with the {partner_topics} framing of *{partner.title}*. "
                f"The graph already links these papers, suggesting a real "
                f"intellectual neighborhood rather than a contrived pairing."
                f"{year_gap}"
            )
        return ideas

    def generate_ideas_from_topic(
        self,
        topic: str,
        papers: list[Paper],
        n_ideas: int = 3,
    ) -> list[str]:
        """Generate research suggestions seeded from a topic and matching papers."""
        if not topic:
            return []
        if not papers:
            return [
                f'No papers match "**{topic}**" in this graph. '
                f"Try a broader keyword or load a Semantic Scholar query."
            ]

        ranked = sorted(papers, key=lambda p: p.citation_count, reverse=True)
        anchors = ranked[:n_ideas]
        ideas: list[str] = []
        for paper in anchors:
            paper_topics = ", ".join(paper.topic_words(3)) or topic
            year = f" ({paper.year})" if paper.year else ""
            ideas.append(
                f"**Extend {paper.title}{year}.** "
                f"Building on {paper_topics}, propose a follow-up that "
                f"either (a) transfers these ideas to an adjacent dataset "
                f"or domain, or (b) re-derives the result with a more "
                f"recent technique — using the citation neighborhood as a "
                f"baseline scaffold."
            )

        if len(ranked) >= 2:
            top, second = ranked[0], ranked[1]
            ideas.append(
                f"**Synthesize {top.title} and {second.title}.** "
                f"Both papers are central to the *{topic}* cluster but "
                f"approach it differently — a joint study contrasting their "
                f"assumptions would clarify which design choices actually "
                f"drive the reported gains."
            )
        return ideas[:n_ideas]

    def explain_surprising_connection(
        self, paper_a: Paper, paper_b: Paper, similarity: float,
    ) -> str:
        """Plain-English narrative for a similarity-only paper pair.

        Mirrors :meth:`ResearchGraph.describe_surprising_connection` but
        without graph-distance information — useful when the graph isn't
        directly available (e.g. tests, exports).
        """
        words_a = set(paper_a.topic_words(8))
        words_b = set(paper_b.topic_words(8))
        shared = words_a & words_b
        parts: list[str] = []
        parts.append(
            f"*{paper_a.title}* and *{paper_b.title}* share a text similarity "
            f"of **{similarity:.2f}** despite having no citation link."
        )
        if paper_a.venue and paper_b.venue and paper_a.venue != paper_b.venue:
            parts.append(
                f"They were published in different venues "
                f"({paper_a.venue} vs. {paper_b.venue}), suggesting "
                f"independent convergence rather than a shared community."
            )
        if shared:
            parts.append(
                f"Both papers address **{', '.join(list(shared)[:3])}** — "
                f"a latent overlap the citation graph alone could not surface."
            )
        return " ".join(parts)

    def suggest_bridge_research(self, topic_a: str, topic_b: str) -> str:
        """Suggest a study that would explicitly bridge two topics.

        Always mentions both topics so the output is verifiable in tests.
        """
        if not topic_a or not topic_b:
            return "Provide two non-empty topics to suggest a bridge."
        return (
            f"### Bridge Research: {topic_a} ↔ {topic_b}\n\n"
            f"Design a study that imports a technique from **{topic_a}** "
            f"into the standard benchmarks of **{topic_b}**. "
            f"Use the graph's bridge papers — those with neighbors in both "
            f"clusters — as the baseline. The novelty comes from explicitly "
            f"controlling for shared structure, so any improvement is "
            f"attributable to the cross-pollination, not to confounding "
            f"changes in dataset or evaluation.\n\n"
            f"**Why it matters:** the {topic_a} and {topic_b} communities "
            f"often re-derive each other's results in isolation; a graph-aware "
            f"bridge study makes the transfer explicit."
        )

    def summarize_research_trajectory(self, papers: list[Paper]) -> str:
        """Alias for :meth:`narrate_trajectory` matching the project spec.

        Kept as a thin wrapper so external code can use either name.
        """
        return self.narrate_trajectory(papers)
