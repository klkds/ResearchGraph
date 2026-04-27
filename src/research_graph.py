"""ResearchGraph: the central graph structure and algorithm engine.

Uses NetworkX to model academic papers as nodes and their relationships
(citations, text similarity) as edges.  Provides graph algorithms for
exploration, meaningful path discovery, centrality analysis, and
higher-level insight extraction.

Design choice — **undirected graph**:
    Real citation graphs are directed (A cites B != B cites A).  We use an
    undirected graph so that path exploration and neighbor queries work
    symmetrically in the Streamlit UI.  The ``edge_type`` attribute
    preserves the citation/similarity distinction for filtering.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import networkx as nx

from src.paper import Paper
from src.utils import compute_similarity_matrix, compute_pairwise_similarity


# -----------------------------------------------------------------------
# Scored path result
# -----------------------------------------------------------------------

@dataclass
class ScoredPath:
    """A graph path between two papers, annotated with a quality score.

    Attributes:
        papers: The ordered list of Paper objects forming the path.
        score: Composite quality score (higher = more meaningful).
        label: Human-readable quality label (Weak / Moderate / Strong).
        avg_similarity: Average pairwise text similarity along the path.
        citation_strength: Fraction of edges that are citation or both.
        length: Number of papers in the path.
    """
    papers: list[Paper]
    score: float
    label: str
    avg_similarity: float
    citation_strength: float
    length: int


class ResearchGraph:
    """Graph of academic papers with citation and similarity edges.

    This is the **central data structure** of the project.  Every
    interactive feature in the Streamlit app reads from this graph.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.papers: dict[str, Paper] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_paper(self, paper: Paper) -> None:
        """Add a single paper as a node (idempotent)."""
        self.papers[paper.paper_id] = paper
        self.graph.add_node(
            paper.paper_id,
            title=paper.title,
            year=paper.year,
            citation_count=paper.citation_count,
        )

    def build_graph(self, papers: list[Paper], similarity_threshold: float = 0.15) -> None:
        """Build the full graph from a list of papers.

        Pipeline:
            1. Add all papers as nodes.
            2. Add citation edges (where both endpoints exist in the graph).
            3. Add similarity edges (TF-IDF cosine >= *similarity_threshold*).
        """
        for paper in papers:
            self.add_paper(paper)
        self.add_citation_edges()
        self.add_similarity_edges(similarity_threshold)

    def add_citation_edges(self) -> None:
        """Create an edge for every citation relationship whose two papers are in the graph."""
        known_ids = set(self.papers.keys())
        for paper in self.papers.values():
            for ref_id in paper.references:
                if ref_id in known_ids:
                    self.graph.add_edge(
                        paper.paper_id, ref_id,
                        edge_type="citation", weight=1.0,
                    )

    def add_similarity_edges(self, similarity_threshold: float = 0.15) -> None:
        """Add edges between papers whose TF-IDF cosine similarity exceeds the threshold."""
        paper_list = list(self.papers.values())
        sim_edges = compute_similarity_matrix(paper_list, threshold=similarity_threshold)
        for id_a, id_b, score in sim_edges:
            if self.graph.has_edge(id_a, id_b):
                self.graph[id_a][id_b]["similarity"] = score
                self.graph[id_a][id_b]["edge_type"] = "both"
            else:
                self.graph.add_edge(
                    id_a, id_b,
                    edge_type="similarity", weight=score, similarity=score,
                )

    def add_shared_author_edges(self) -> int:
        """Annotate edges between papers that share at least one author.

        We don't *create* a new edge type here — instead we tag any
        existing edge (citation or similarity) with ``shared_authors``
        when the two endpoints share an author, AND we add a lightweight
        ``shared_author`` edge when there is no existing connection.
        This keeps the graph compatible with all existing analytics
        while exposing co-authorship as a queryable signal.

        Returns the number of newly added shared-author edges.
        """
        ids_by_author: dict[str, list[str]] = {}
        for paper in self.papers.values():
            for author in paper.authors:
                key = author.strip().lower()
                if not key:
                    continue
                ids_by_author.setdefault(key, []).append(paper.paper_id)

        new_edges = 0
        for author, ids in ids_by_author.items():
            if len(ids) < 2:
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if self.graph.has_edge(a, b):
                        existing = self.graph[a][b].setdefault("shared_authors", [])
                        if author not in existing:
                            existing.append(author)
                    else:
                        self.graph.add_edge(
                            a, b,
                            edge_type="shared_author",
                            weight=0.5,
                            shared_authors=[author],
                        )
                        new_edges += 1
        return new_edges

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_paper(self, paper_id: str) -> Paper | None:
        return self.papers.get(paper_id)

    def get_paper_by_title(self, title: str) -> Paper | None:
        title_lower = title.lower()
        for paper in self.papers.values():
            if paper.title.lower() == title_lower:
                return paper
        return None

    def search_papers(self, keyword: str) -> list[Paper]:
        keyword_lower = keyword.lower()
        return [
            paper for paper in self.papers.values()
            if keyword_lower in paper.title.lower()
            or keyword_lower in paper.abstract.lower()
        ]

    # ------------------------------------------------------------------
    # Neighborhood
    # ------------------------------------------------------------------

    def get_neighbors(self, paper_id: str) -> list[dict[str, Any]]:
        """Return neighbor info dicts for a given paper node."""
        if paper_id not in self.graph:
            return []
        neighbors = []
        for nbr_id in self.graph.neighbors(paper_id):
            edge_data = self.graph[paper_id][nbr_id]
            paper = self.papers.get(nbr_id)
            if paper:
                neighbors.append({
                    "paper": paper,
                    "edge_type": edge_data.get("edge_type", "unknown"),
                    "similarity": edge_data.get("similarity", None),
                    "shared_authors": list(edge_data.get("shared_authors", []) or []),
                })
        return neighbors

    def get_related_papers(
        self, paper_id: str, top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return papers related to ``paper_id`` with a score and human reasons.

        The score combines:
            - citation link (+1.0)
            - text similarity (the cosine score itself)
            - shared authors (+0.4 * count)
            - shared topic words (+0.05 per shared word, capped)

        Each result also includes a ``reasons`` list that explains *why*
        the paper is related, so the UI can surface concrete justification
        instead of opaque scores.
        """
        if paper_id not in self.graph:
            return []
        center = self.papers.get(paper_id)
        if center is None:
            return []
        center_topics = set(center.topic_words(8))
        center_authors = {a.strip().lower() for a in center.authors if a}

        results: list[dict[str, Any]] = []
        for nbr_id in self.graph.neighbors(paper_id):
            paper = self.papers.get(nbr_id)
            if paper is None:
                continue
            edge = self.graph[paper_id][nbr_id]
            edge_type = edge.get("edge_type", "unknown")
            sim = edge.get("similarity")

            score = 0.0
            reasons: list[str] = []
            if edge_type in ("citation", "both"):
                score += 1.0
                reasons.append("direct citation link")
            if sim:
                score += float(sim)
                reasons.append(f"text similarity {sim:.2f}")

            shared_authors = [
                a for a in paper.authors
                if a.strip().lower() in center_authors
            ]
            if shared_authors:
                score += 0.4 * len(shared_authors)
                preview = ", ".join(shared_authors[:2])
                reasons.append(f"shared author(s): {preview}")

            shared_topics = center_topics & set(paper.topic_words(8))
            if shared_topics:
                score += min(0.05 * len(shared_topics), 0.25)
                reasons.append(
                    f"shared topic words: {', '.join(list(shared_topics)[:3])}"
                )

            if not reasons:
                reasons.append(f"connected via {edge_type} edge")

            results.append({
                "paper": paper,
                "score": round(score, 4),
                "reasons": reasons,
                "edge_type": edge_type,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def get_topic_summary(self, top_n_words: int = 10) -> list[tuple[str, int]]:
        """Aggregate topic words across all papers and rank by frequency.

        Returns a list of ``(word, paper_count)`` tuples — useful as a
        cheap topic dashboard when no formal topic model is available.
        """
        word_doc_freq: dict[str, int] = {}
        for paper in self.papers.values():
            for word in set(paper.topic_words(8)):
                word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
        ranked = sorted(word_doc_freq.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n_words]

    def find_bridge_papers(
        self, topic_a: str, topic_b: str, top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find papers that bridge two topic keywords.

        A paper is a bridge if it has neighbors matching ``topic_a`` AND
        neighbors matching ``topic_b``.  Bridges are scored by how many
        topic-A neighbors and topic-B neighbors each one connects.
        """
        if not topic_a or not topic_b:
            return []
        ids_a = {p.paper_id for p in self.search_papers(topic_a)}
        ids_b = {p.paper_id for p in self.search_papers(topic_b)}
        if not ids_a or not ids_b:
            return []

        bridges: list[dict[str, Any]] = []
        for pid, paper in self.papers.items():
            if pid in ids_a or pid in ids_b:
                continue
            nbrs = set(self.graph.neighbors(pid))
            in_a = nbrs & ids_a
            in_b = nbrs & ids_b
            if in_a and in_b:
                score = len(in_a) + len(in_b)
                bridges.append({
                    "paper": paper,
                    "score": score,
                    "topic_a_neighbors": [self.papers[i] for i in in_a if i in self.papers],
                    "topic_b_neighbors": [self.papers[i] for i in in_b if i in self.papers],
                })

        bridges.sort(key=lambda b: b["score"], reverse=True)
        return bridges[:top_k]

    def extract_subgraph(self, center_id: str, radius: int = 1) -> nx.Graph:
        if center_id not in self.graph:
            return nx.Graph()
        nodes = {center_id}
        frontier = {center_id}
        for _ in range(radius):
            next_frontier: set[str] = set()
            for node in frontier:
                for nbr in self.graph.neighbors(node):
                    if nbr not in nodes:
                        next_frontier.add(nbr)
                        nodes.add(nbr)
            frontier = next_frontier
        return self.graph.subgraph(nodes).copy()

    # ------------------------------------------------------------------
    # Meaningful paths (replaces naive shortest-path)
    # ------------------------------------------------------------------

    def _score_path(self, id_path: list[str]) -> ScoredPath | None:
        """Score a path of paper ids for semantic coherence and relevance.

        score = avg_similarity + citation_strength - short_path_penalty

        - avg_similarity: mean pairwise TF-IDF cosine similarity of
          consecutive papers in the path.
        - citation_strength: fraction of edges that are citation or both
          (vs similarity-only), rewarding paths grounded in real citations.
        - short_path_penalty: 0.3 for 1-hop paths (often trivially
          adjacent), 0.1 for 2-hop.  No penalty for 3+ hops.
        """
        papers = [self.papers[pid] for pid in id_path if pid in self.papers]
        if len(papers) < 2:
            return None

        # Average pairwise similarity along the path
        sim_scores: list[float] = []
        citation_count = 0
        total_edges = 0
        for i in range(len(papers) - 1):
            a, b = papers[i], papers[i + 1]
            sim = compute_pairwise_similarity(a, b)
            sim_scores.append(sim)
            # Check edge type
            if self.graph.has_edge(a.paper_id, b.paper_id):
                etype = self.graph[a.paper_id][b.paper_id].get("edge_type", "")
                if etype in ("citation", "both"):
                    citation_count += 1
            total_edges += 1

        avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
        cite_strength = citation_count / total_edges if total_edges else 0.0

        hops = len(papers) - 1
        if hops <= 1:
            penalty = 0.3
        elif hops == 2:
            penalty = 0.1
        else:
            penalty = 0.0

        score = avg_sim + cite_strength - penalty
        score = round(score, 4)

        if score >= 0.8:
            label = "Strong"
        elif score >= 0.4:
            label = "Moderate"
        else:
            label = "Weak"

        return ScoredPath(
            papers=papers,
            score=score,
            label=label,
            avg_similarity=round(avg_sim, 4),
            citation_strength=round(cite_strength, 4),
            length=len(papers),
        )

    def find_meaningful_paths(
        self, source_id: str, target_id: str, top_k: int = 3
    ) -> list[ScoredPath]:
        """Find the top-k most meaningful paths between two papers.

        Unlike bare shortest-path, this method:
        - Finds multiple simple paths (up to a cutoff length).
        - Scores each for semantic coherence and citation grounding.
        - Penalizes trivial 1-hop paths.
        - Returns the best paths sorted by quality score.

        Returns an empty list if no path exists.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return []
        if source_id == target_id:
            return []

        try:
            # Collect candidate paths (simple paths up to length 6)
            raw_paths = list(
                itertools.islice(
                    nx.all_simple_paths(self.graph, source_id, target_id, cutoff=5),
                    50,  # cap at 50 candidates for performance
                )
            )
        except nx.NetworkXError:
            return []

        if not raw_paths:
            return []

        scored: list[ScoredPath] = []
        for id_path in raw_paths:
            sp = self._score_path(id_path)
            if sp is not None:
                scored.append(sp)

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def shortest_path(self, source_id: str, target_id: str) -> list[Paper] | None:
        """Find the shortest path between two papers (internal utility).

        Kept for backward compatibility and learning-path fallback.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None
        try:
            id_path = nx.shortest_path(self.graph, source=source_id, target=target_id)
            return [self.papers[pid] for pid in id_path if pid in self.papers]
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------
    # Learning path: topic A → topic B
    # ------------------------------------------------------------------

    def learning_path(
        self, topic_a: str, topic_b: str, max_steps: int = 6
    ) -> list[ScoredPath]:
        """Build a structured learning path from topic A to topic B.

        1. Find papers matching each topic.
        2. Try all source-target pairings across the two topic groups.
        3. Score and rank the paths.
        4. Return the best paths that represent plausible research
           progressions between the two topics.

        This is designed for queries like "I want to go from
        transformers to diffusion models."
        """
        papers_a = self.search_papers(topic_a)
        papers_b = self.search_papers(topic_b)

        if not papers_a or not papers_b:
            return []

        # Limit candidates to avoid combinatorial explosion
        ids_a = [p.paper_id for p in papers_a[:8]]
        ids_b = [p.paper_id for p in papers_b[:8]]

        # Remove overlap (a paper matching both topics)
        overlap = set(ids_a) & set(ids_b)
        ids_a = [pid for pid in ids_a if pid not in overlap]
        ids_b = [pid for pid in ids_b if pid not in overlap]

        if not ids_a or not ids_b:
            return []

        all_scored: list[ScoredPath] = []
        seen_signatures: set[tuple[str, str]] = set()

        for src in ids_a:
            for tgt in ids_b:
                sig = (src, tgt)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                paths = self.find_meaningful_paths(src, tgt, top_k=1)
                all_scored.extend(paths)

        all_scored.sort(key=lambda s: s.score, reverse=True)

        # Deduplicate — keep only paths with different paper sequences
        unique: list[ScoredPath] = []
        seen_seqs: set[tuple[str, ...]] = set()
        for sp in all_scored:
            seq = tuple(p.paper_id for p in sp.papers)
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                unique.append(sp)
            if len(unique) >= 3:
                break

        return unique

    # ------------------------------------------------------------------
    # Research trajectory
    # ------------------------------------------------------------------

    def research_trajectory(self, topic_keyword: str, max_papers: int = 10) -> list[Paper]:
        """Chronological trajectory for a topic through the graph."""
        matches = self.search_papers(topic_keyword)
        dated = [p for p in matches if p.year is not None]
        dated.sort(key=lambda p: p.year)  # type: ignore[arg-type]
        return dated[:max_papers]

    # ------------------------------------------------------------------
    # Centrality & ranking
    # ------------------------------------------------------------------

    def rank_by_degree(self, top_n: int = 10) -> list[tuple[Paper, int]]:
        degree_list = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
        results = []
        for pid, deg in degree_list[:top_n]:
            paper = self.papers.get(pid)
            if paper:
                results.append((paper, deg))
        return results

    def rank_by_betweenness(self, top_n: int = 10) -> list[tuple[Paper, float]]:
        bc = nx.betweenness_centrality(self.graph)
        sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        results = []
        for pid, score in sorted_bc[:top_n]:
            paper = self.papers.get(pid)
            if paper:
                results.append((paper, round(score, 4)))
        return results

    def rank_by_pagerank(self, top_n: int = 10) -> list[tuple[Paper, float]]:
        pr = nx.pagerank(self.graph)
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        results = []
        for pid, score in sorted_pr[:top_n]:
            paper = self.papers.get(pid)
            if paper:
                results.append((paper, round(score, 4)))
        return results

    # ------------------------------------------------------------------
    # Bridge paper spotlight
    # ------------------------------------------------------------------

    def bridge_paper_spotlight(self) -> dict[str, Any] | None:
        if self.graph.number_of_nodes() < 3:
            return None
        bc = nx.betweenness_centrality(self.graph)
        if not bc or max(bc.values()) == 0:
            return None
        bridge_id = max(bc, key=bc.get)  # type: ignore[arg-type]
        bridge_paper = self.papers.get(bridge_id)
        if not bridge_paper:
            return None
        neighbors = list(self.graph.neighbors(bridge_id))
        if len(neighbors) < 2:
            return None
        bridged_pairs: list[tuple[Paper, Paper]] = []
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if not self.graph.has_edge(n1, n2):
                    p1, p2 = self.papers.get(n1), self.papers.get(n2)
                    if p1 and p2:
                        bridged_pairs.append((p1, p2))
        return {
            "paper": bridge_paper,
            "betweenness": round(bc[bridge_id], 4),
            "num_connections": len(neighbors),
            "bridged_pairs": bridged_pairs[:5],
        }

    # ------------------------------------------------------------------
    # Surprising connections
    # ------------------------------------------------------------------

    def find_surprising_connections(self, top_n: int = 5) -> list[tuple[Paper, Paper, float]]:
        surprises: list[tuple[Paper, Paper, float]] = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == "similarity" and data.get("similarity"):
                p1, p2 = self.papers.get(u), self.papers.get(v)
                if p1 and p2:
                    surprises.append((p1, p2, data["similarity"]))
        surprises.sort(key=lambda x: x[2], reverse=True)
        return surprises[:top_n]

    def describe_surprising_connection(
        self, paper_a: Paper, paper_b: Paper, similarity: float,
    ) -> str:
        """Generate an insight-rich description explaining *why* a
        similarity-only connection between two papers is surprising.

        Analyzes graph distance, topic overlap, venue difference, and
        temporal gap to produce a narrative about the unexpected link.
        """
        words_a = set(paper_a.topic_words(8))
        words_b = set(paper_b.topic_words(8))
        shared = words_a & words_b
        unique_a = words_a - words_b
        unique_b = words_b - words_a

        parts: list[str] = []

        # Graph distance context
        try:
            dist = nx.shortest_path_length(self.graph, paper_a.paper_id, paper_b.paper_id)
            if dist > 2:
                parts.append(
                    f"Despite being **{dist} hops apart** in the citation graph, "
                    f"these papers share a similarity of **{similarity:.3f}** — "
                    f"an unexpected bridge across distant research communities."
                )
        except (nx.NetworkXNoPath, nx.NetworkXError):
            pass

        # Venue divergence
        if paper_a.venue and paper_b.venue and paper_a.venue != paper_b.venue:
            parts.append(
                f"Published in different venues (**{paper_a.venue}** vs. "
                f"**{paper_b.venue}**), suggesting independent convergence "
                f"on related ideas from separate research communities."
            )

        # Temporal gap
        if paper_a.year and paper_b.year:
            gap = abs(paper_a.year - paper_b.year)
            if gap >= 3:
                earlier = paper_a if paper_a.year < paper_b.year else paper_b
                later = paper_b if paper_a.year < paper_b.year else paper_a
                parts.append(
                    f"A **{gap}-year gap** separates these works — "
                    f"*{later.title}* may have independently rediscovered "
                    f"or reinvented concepts from *{earlier.title}*."
                )

        # Topic analysis
        if shared and (unique_a or unique_b):
            parts.append(
                f"They converge on **{', '.join(list(shared)[:3])}**, "
                f"but approach it from different angles: "
                f"*{paper_a.title}* through {', '.join(list(unique_a)[:2]) or 'its own lens'}, "
                f"*{paper_b.title}* through {', '.join(list(unique_b)[:2]) or 'its own lens'}."
            )
        elif not shared:
            parts.append(
                f"Remarkably, these papers have **no obvious keyword overlap**, "
                f"yet the similarity model detects a latent conceptual connection "
                f"invisible to surface-level analysis."
            )

        if not parts:
            parts.append(
                f"This similarity-only edge (score **{similarity:.3f}**) reveals a "
                f"hidden intellectual kinship between papers that never cite each other."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Cluster detection
    # ------------------------------------------------------------------

    def detect_clusters(self, min_size: int = 2) -> list[dict]:
        """Detect research clusters using greedy modularity community detection.

        Returns a list of cluster dicts, each containing:
            - papers: list of Paper objects in the cluster
            - theme_words: most common topic words across the cluster
            - year_range: (min_year, max_year)
            - total_citations: sum of citation counts
            - internal_density: edge density within the cluster subgraph
        """
        if self.graph.number_of_nodes() < 3:
            return []

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(self.graph))
        except Exception:
            # Fallback: connected components
            communities = list(nx.connected_components(self.graph))

        clusters: list[dict] = []
        for comm in communities:
            papers_in = [self.papers[pid] for pid in comm if pid in self.papers]
            if len(papers_in) < min_size:
                continue

            # Theme words across all papers in cluster
            word_freq: dict[str, int] = {}
            for p in papers_in:
                for w in p.topic_words(5):
                    word_freq[w] = word_freq.get(w, 0) + 1
            theme_words = [w for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]]

            years = [p.year for p in papers_in if p.year]
            year_range = (min(years), max(years)) if years else (None, None)

            # Internal density
            sub = self.graph.subgraph(comm)
            internal_density = round(nx.density(sub), 4)

            clusters.append({
                "papers": sorted(papers_in, key=lambda p: p.citation_count, reverse=True),
                "theme_words": theme_words,
                "year_range": year_range,
                "total_citations": sum(p.citation_count for p in papers_in),
                "internal_density": internal_density,
            })

        # Sort clusters by total citations (proxy for importance)
        clusters.sort(key=lambda c: c["total_citations"], reverse=True)
        return clusters

    def describe_bridge_role(self, paper_id: str) -> dict | None:
        """Analyze the structural role a bridge paper plays in the graph.

        Returns a dict with:
            - communities_connected: list of (theme_words, paper_count) for
              each cluster the bridge paper touches
            - fragmentation_risk: how many components would result if this
              paper were removed
            - role_description: narrative explanation of the bridge role
        """
        if paper_id not in self.graph:
            return None
        paper = self.papers.get(paper_id)
        if not paper:
            return None

        neighbors = list(self.graph.neighbors(paper_id))
        if len(neighbors) < 2:
            return None

        # Check fragmentation: what happens if we remove this node?
        test_graph = self.graph.copy()
        original_components = nx.number_connected_components(test_graph)
        test_graph.remove_node(paper_id)
        new_components = nx.number_connected_components(test_graph)
        fragmentation_risk = new_components - original_components

        # Group neighbors by their cluster membership
        clusters = self.detect_clusters(min_size=1)
        cluster_map: dict[str, int] = {}
        for idx, cl in enumerate(clusters):
            for p in cl["papers"]:
                cluster_map[p.paper_id] = idx

        neighbor_clusters: dict[int, list[Paper]] = {}
        for nbr_id in neighbors:
            cl_idx = cluster_map.get(nbr_id, -1)
            neighbor_clusters.setdefault(cl_idx, [])
            nbr_paper = self.papers.get(nbr_id)
            if nbr_paper:
                neighbor_clusters[cl_idx].append(nbr_paper)

        communities_connected: list[tuple[list[str], int]] = []
        for cl_idx, cl_papers in neighbor_clusters.items():
            if cl_idx >= 0 and cl_idx < len(clusters):
                themes = clusters[cl_idx]["theme_words"][:3]
                count = len(clusters[cl_idx]["papers"])
            else:
                themes = []
                for p in cl_papers:
                    themes.extend(p.topic_words(2))
                themes = list(dict.fromkeys(themes))[:3]
                count = len(cl_papers)
            communities_connected.append((themes, count))

        # Build narrative
        paper_topics = ", ".join(paper.topic_words(3)) or "its concepts"
        if len(communities_connected) >= 2:
            comm_strs = [
                f"**{', '.join(t)}** ({c} papers)" if t else f"a group of {c} papers"
                for t, c in communities_connected[:3]
            ]
            role_desc = (
                f"**{paper.title}** acts as a critical connector between "
                f"{' and '.join(comm_strs)}. "
            )
            if fragmentation_risk > 0:
                role_desc += (
                    f"Removing it would **fragment the graph into "
                    f"{fragmentation_risk} additional component(s)**, "
                    f"severing the intellectual link between these research areas. "
                )
            role_desc += (
                f"Through its focus on {paper_topics}, this paper translates "
                f"concepts across community boundaries — making it a structural "
                f"keystone in the knowledge graph."
            )
        else:
            role_desc = (
                f"**{paper.title}** has high betweenness centrality, "
                f"meaning many shortest paths between other papers pass through it. "
                f"Its work on {paper_topics} serves as a shared reference point "
                f"across the graph."
            )

        return {
            "communities_connected": communities_connected,
            "fragmentation_risk": fragmentation_risk,
            "role_description": role_desc,
        }

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def export_graph_json(self, path: str) -> None:
        """Serialize the graph (papers + edges) to a JSON file.

        The format round-trips with :meth:`load_graph_json` and preserves
        every node and edge with all attributes — so a graph can be
        rebuilt without re-running similarity computation.
        """
        import json
        from pathlib import Path

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "attributes": dict(data),
            })
        payload = {
            "papers": [p.to_dict() for p in self.papers.values()],
            "edges": edges,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_graph_json(cls, path: str) -> "ResearchGraph":
        """Reconstruct a ResearchGraph from a JSON file written by export_graph_json."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        rg = cls()
        for paper_dict in payload.get("papers", []):
            rg.add_paper(Paper.from_dict(paper_dict))
        for edge in payload.get("edges", []):
            attrs = edge.get("attributes", {})
            rg.graph.add_edge(edge["source"], edge["target"], **attrs)
        return rg

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        num_citation = sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d.get("edge_type") in ("citation", "both")
        )
        num_similarity = sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d.get("edge_type") in ("similarity", "both")
        )
        components = nx.number_connected_components(self.graph)
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "citation_edges": num_citation,
            "similarity_edges": num_similarity,
            "connected_components": components,
            "density": round(nx.density(self.graph), 4),
        }
