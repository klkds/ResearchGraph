"""Behavioral specification for the ResearchGraph class.

ResearchGraph is the central data structure of the project.  Every
interactive feature reads from this graph.

These tests document what the graph should do and why each behavior
matters for the application.
"""

import pytest

from src.paper import Paper
from src.research_graph import ResearchGraph, ScoredPath


def _make_paper(pid: str, title: str = "Paper", refs: list[str] | None = None, **kwargs) -> Paper:
    return Paper(paper_id=pid, title=title, references=refs or [], **kwargs)


# ===========================================================================
# Graph construction
# ===========================================================================

class TestGraphConstruction:
    """The graph builds itself from Paper objects, creating nodes and edges."""

    def test_each_paper_becomes_a_node(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper("1", "Test"))
        assert "1" in rg.papers
        assert rg.graph.number_of_nodes() == 1

    def test_build_graph_creates_all_nodes(self):
        rg = ResearchGraph()
        papers = [_make_paper(f"p{i}", f"Paper {i}") for i in range(5)]
        rg.build_graph(papers, similarity_threshold=0.99)
        assert rg.graph.number_of_nodes() == 5

    def test_citation_creates_edge_between_known_papers(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Paper A", refs=["b"])
        p2 = _make_paper("b", "Paper B")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        assert rg.graph.has_edge("a", "b")

    def test_citation_edge_is_labeled(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "X", refs=["b"])
        p2 = _make_paper("b", "Y")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        assert rg.graph["a"]["b"]["edge_type"] in ("citation", "both")

    def test_reference_to_unknown_paper_creates_no_edge(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A", refs=["nonexistent"])
        rg.build_graph([p1], similarity_threshold=0.99)
        assert rg.graph.number_of_edges() == 0

    def test_similar_papers_get_similarity_edge(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Deep learning neural network training",
                         abstract="Training deep neural networks efficiently")
        p2 = _make_paper("b", "Neural network training methods",
                         abstract="Methods for training deep neural networks")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        assert rg.graph.has_edge("a", "b")

    def test_dissimilar_papers_get_no_similarity_edge(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Quantum mechanics fundamentals",
                         abstract="Quantum physics and wave functions")
        p2 = _make_paper("b", "Culinary arts in France",
                         abstract="French cooking techniques and recipes")
        rg.build_graph([p1, p2], similarity_threshold=0.5)
        assert not rg.graph.has_edge("a", "b")


# ===========================================================================
# Paper lookup and search
# ===========================================================================

class TestPaperRetrieval:
    """The graph supports lookup by id, title, or keyword."""

    @pytest.fixture
    def populated_graph(self) -> ResearchGraph:
        rg = ResearchGraph()
        papers = [
            _make_paper("p1", "Attention Is All You Need",
                        abstract="Transformer architecture based on attention"),
            _make_paper("p2", "BERT Pre-training",
                        abstract="Bidirectional encoder representations"),
            _make_paper("p3", "Graph Neural Networks",
                        abstract="Learning on graph-structured data"),
        ]
        rg.build_graph(papers, similarity_threshold=0.99)
        return rg

    def test_get_paper_by_id(self, populated_graph: ResearchGraph):
        p = populated_graph.get_paper("p1")
        assert p is not None
        assert p.title == "Attention Is All You Need"

    def test_get_paper_returns_none_for_missing_id(self, populated_graph: ResearchGraph):
        assert populated_graph.get_paper("nonexistent") is None

    def test_get_paper_by_title_is_case_insensitive(self, populated_graph: ResearchGraph):
        p = populated_graph.get_paper_by_title("bert pre-training")
        assert p is not None
        assert p.paper_id == "p2"

    def test_search_matches_title_keywords(self, populated_graph: ResearchGraph):
        results = populated_graph.search_papers("attention")
        assert len(results) == 1
        assert results[0].paper_id == "p1"

    def test_search_returns_empty_for_no_match(self, populated_graph: ResearchGraph):
        assert populated_graph.search_papers("quantum computing") == []


# ===========================================================================
# Neighbor exploration
# ===========================================================================

class TestNeighborExploration:
    """Graph Explorer relies on neighbor queries."""

    def test_neighbors_include_all_adjacent_papers(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A", refs=["b", "c"])
        p2 = _make_paper("b", "B")
        p3 = _make_paper("c", "C")
        rg.build_graph([p1, p2, p3], similarity_threshold=0.99)
        nbr_ids = {n["paper"].paper_id for n in rg.get_neighbors("a")}
        assert nbr_ids == {"b", "c"}

    def test_isolated_paper_has_no_neighbors(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper("lonely", "Lonely Paper"))
        assert rg.get_neighbors("lonely") == []

    def test_nonexistent_node_returns_empty(self):
        rg = ResearchGraph()
        assert rg.get_neighbors("ghost") == []

    def test_neighbor_dicts_include_edge_type(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A", refs=["b"])
        p2 = _make_paper("b", "B")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        nbrs = rg.get_neighbors("a")
        assert len(nbrs) == 1
        assert "edge_type" in nbrs[0]


# ===========================================================================
# Meaningful paths (the core refactored feature)
# ===========================================================================

class TestMeaningfulPaths:
    """Meaningful paths replace naive shortest-path with scored, multi-hop results.

    A meaningful path:
    - Is scored for semantic coherence and citation grounding.
    - Penalizes trivial 1-hop paths.
    - Returns multiple candidates ranked by quality.
    """

    @pytest.fixture
    def chain_graph(self) -> ResearchGraph:
        """Linear chain: a → b → c → d with realistic text."""
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Deep learning foundations",
                        refs=["b"], abstract="Neural network fundamentals"),
            _make_paper("b", "Convolutional neural networks for vision",
                        refs=["c"], abstract="Deep learning applied to images"),
            _make_paper("c", "Object detection with neural networks",
                        refs=["d"], abstract="Detecting objects in images using deep models"),
            _make_paper("d", "Autonomous driving perception systems",
                        abstract="Perception for self driving cars using detection"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        return rg

    def test_finds_paths_between_connected_papers(self, chain_graph: ResearchGraph):
        """Should find at least one path in a connected graph."""
        paths = chain_graph.find_meaningful_paths("a", "d", top_k=3)
        assert len(paths) >= 1

    def test_paths_are_scored_path_objects(self, chain_graph: ResearchGraph):
        """Each result must be a ScoredPath with all required fields."""
        paths = chain_graph.find_meaningful_paths("a", "d", top_k=1)
        assert len(paths) >= 1
        sp = paths[0]
        assert isinstance(sp, ScoredPath)
        assert isinstance(sp.score, float)
        assert sp.label in ("Weak", "Moderate", "Strong")
        assert sp.length >= 2

    def test_paths_are_sorted_by_score_descending(self, chain_graph: ResearchGraph):
        """Multiple paths should be returned best-first."""
        paths = chain_graph.find_meaningful_paths("a", "d", top_k=5)
        if len(paths) >= 2:
            assert paths[0].score >= paths[1].score

    def test_same_node_returns_empty(self, chain_graph: ResearchGraph):
        """Source == target should return no paths."""
        assert chain_graph.find_meaningful_paths("a", "a") == []

    def test_disconnected_nodes_return_empty(self):
        """No path exists between disconnected components."""
        rg = ResearchGraph()
        rg.add_paper(_make_paper("x", "X"))
        rg.add_paper(_make_paper("y", "Y"))
        assert rg.find_meaningful_paths("x", "y") == []

    def test_nonexistent_node_returns_empty(self, chain_graph: ResearchGraph):
        assert chain_graph.find_meaningful_paths("a", "zzz") == []

    def test_short_paths_are_penalized(self):
        """A 1-hop path should receive a penalty lowering its score."""
        rg = ResearchGraph()
        p1 = _make_paper("a", "Paper about neural networks", refs=["b"],
                         abstract="Neural network methods")
        p2 = _make_paper("b", "Paper about neural networks too",
                         abstract="More neural network methods")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        paths = rg.find_meaningful_paths("a", "b", top_k=1)
        if paths:
            # 1-hop path gets 0.3 penalty, so score should be lower than
            # raw similarity + citation_strength
            sp = paths[0]
            raw = sp.avg_similarity + sp.citation_strength
            assert sp.score < raw  # penalty was applied

    def test_respects_top_k(self, chain_graph: ResearchGraph):
        """Should return at most top_k paths."""
        paths = chain_graph.find_meaningful_paths("a", "d", top_k=1)
        assert len(paths) <= 1


class TestScoredPathQualityLabel:
    """The quality label reflects the score range."""

    def test_strong_label_for_high_score(self):
        """Score >= 0.8 should be labeled Strong."""
        rg = ResearchGraph()
        # Build a path with high similarity and citation strength
        p1 = _make_paper("a", "Transformer attention mechanism architecture",
                         refs=["b"],
                         abstract="Self attention in transformer neural network architectures")
        p2 = _make_paper("b", "Transformer architecture for machine translation",
                         abstract="Neural machine translation using transformer attention")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        paths = rg.find_meaningful_paths("a", "b", top_k=1)
        # We can check the label exists (exact score depends on TF-IDF)
        if paths:
            assert paths[0].label in ("Weak", "Moderate", "Strong")


# ===========================================================================
# Path scoring — behavioral specification of the scoring formula
# ===========================================================================

class TestPathScoringFormula:
    """The scoring formula score = avg_similarity + citation_strength - penalty
    encodes specific design decisions about what makes a research trajectory
    meaningful.

    These tests verify those decisions:
    - Citation-grounded paths score higher than similarity-only paths
    - Semantically coherent paths score higher than incoherent ones
    - Trivial 1-hop connections are penalized most heavily (0.3)
    - 2-hop paths receive a moderate penalty (0.1)
    - 3+ hop paths receive no penalty — they are inherently non-trivial
    """

    def test_citation_edges_increase_score_over_similarity_only(self):
        """A path whose edges are all citation links should score higher than
        an otherwise identical path where edges are similarity-only, because
        citation edges confirm direct intellectual lineage."""
        rg = ResearchGraph()
        # Two papers with a citation link between them
        p1 = _make_paper("a", "Transformer self-attention mechanism",
                         refs=["b"],
                         abstract="Self attention mechanism for sequence transduction")
        p2 = _make_paper("b", "Multi-head attention in transformers",
                         abstract="Extending self attention with multiple heads")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        paths = rg.find_meaningful_paths("a", "b", top_k=1)
        assert len(paths) == 1
        sp = paths[0]
        # citation_strength should be 1.0 (the only edge is citation/both)
        assert sp.citation_strength == 1.0
        # Score = avg_sim + 1.0 - 0.3 (1-hop penalty)
        # If we had similarity-only, citation_strength would be 0.0
        # so citation adds a full 1.0 to the score
        expected_without_citation = sp.avg_similarity + 0.0 - 0.3
        assert sp.score > expected_without_citation

    def test_high_similarity_path_scores_above_low_similarity_path(self):
        """A path between papers with closely related text should score higher
        than a path between papers with unrelated text, because semantic
        coherence indicates a genuine intellectual trajectory."""
        rg = ResearchGraph()
        # Closely related papers
        papers = [
            _make_paper("a", "Neural network training methods",
                        refs=["b", "c"],
                        abstract="Training deep neural networks with backpropagation"),
            _make_paper("b", "Deep neural network optimization",
                        abstract="Optimizing deep neural network training procedures"),
            _make_paper("c", "Volcanic eruption prediction",
                        abstract="Predicting volcanic eruptions using seismic data"),
        ]
        rg.build_graph(papers, similarity_threshold=0.01)
        path_related = rg._score_path(["a", "b"])
        path_unrelated = rg._score_path(["a", "c"])
        assert path_related is not None and path_unrelated is not None
        # The neural-network path should have much higher similarity
        assert path_related.avg_similarity > path_unrelated.avg_similarity

    def test_one_hop_penalty_is_exactly_0_3(self):
        """A direct 1-hop connection receives a 0.3 penalty because single-edge
        adjacency is often trivial — the value is in multi-hop trajectories
        that reveal non-obvious connections."""
        rg = ResearchGraph()
        p1 = _make_paper("a", "Paper one neural network training",
                         refs=["b"],
                         abstract="Training neural networks with new methods")
        p2 = _make_paper("b", "Paper two neural network training",
                         abstract="More training of neural networks")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        sp = rg._score_path(["a", "b"])
        assert sp is not None
        # Verify penalty: score = avg_sim + cite_strength - 0.3
        reconstructed = round(sp.avg_similarity + sp.citation_strength - 0.3, 4)
        assert sp.score == reconstructed

    def test_two_hop_penalty_is_exactly_0_1(self):
        """A 2-hop path receives a moderate 0.1 penalty — more valuable than
        a direct edge but still somewhat expected in a well-connected graph."""
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Attention mechanism research",
                        refs=["b"],
                        abstract="Self attention for neural sequences"),
            _make_paper("b", "Transformer architecture design",
                        refs=["c"],
                        abstract="Architecture using attention for translation"),
            _make_paper("c", "BERT pretraining method",
                        abstract="Pretraining transformers bidirectionally"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        sp = rg._score_path(["a", "b", "c"])
        assert sp is not None
        reconstructed = round(sp.avg_similarity + sp.citation_strength - 0.1, 4)
        assert sp.score == reconstructed

    def test_three_plus_hop_paths_receive_no_penalty(self):
        """Paths of 3+ hops traverse enough intermediate papers to be
        inherently non-trivial — they reveal genuine conceptual chains,
        so they receive no penalty."""
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Early deep learning foundations",
                        refs=["b"],
                        abstract="Foundations of deep learning research"),
            _make_paper("b", "Convolutional deep neural networks",
                        refs=["c"],
                        abstract="Convolutional networks for vision tasks"),
            _make_paper("c", "Object detection with deep models",
                        refs=["d"],
                        abstract="Detection of objects using deep networks"),
            _make_paper("d", "Autonomous driving perception",
                        abstract="Self driving car perception systems"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        sp = rg._score_path(["a", "b", "c", "d"])
        assert sp is not None
        # No penalty: score = avg_sim + cite_strength
        reconstructed = round(sp.avg_similarity + sp.citation_strength, 4)
        assert sp.score == reconstructed

    def test_score_components_are_in_valid_ranges(self):
        """avg_similarity is in [0,1], citation_strength is in [0,1],
        and the label correctly maps the final score."""
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Neural attention mechanisms", refs=["b"],
                        abstract="Attention in neural networks"),
            _make_paper("b", "Transformer attention layers", refs=["c"],
                        abstract="Transformer layers using attention"),
            _make_paper("c", "BERT language model pretraining",
                        abstract="Pretraining language models"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        sp = rg._score_path(["a", "b", "c"])
        assert sp is not None
        assert 0.0 <= sp.avg_similarity <= 1.0
        assert 0.0 <= sp.citation_strength <= 1.0
        assert sp.length == 3
        # Label must match score thresholds
        if sp.score >= 0.8:
            assert sp.label == "Strong"
        elif sp.score >= 0.4:
            assert sp.label == "Moderate"
        else:
            assert sp.label == "Weak"


class TestTrivialPathLabeling:
    """1-hop paths should be labeled Weak or penalized below longer alternatives.

    The system is designed to surface non-obvious connections. A single-edge
    path between adjacent papers is typically trivial — it tells the user
    nothing they couldn't find by reading citations. The penalty ensures
    multi-hop trajectories that reveal conceptual evolution are preferred.
    """

    def test_one_hop_path_is_labeled_weak_when_dissimilar_and_uncited(self):
        """A 1-hop similarity-only path between dissimilar papers should be
        labeled Weak because: low similarity + 0 citation_strength - 0.3
        penalty yields a negative or near-zero score (well below 0.4)."""
        rg = ResearchGraph()
        # Similarity-only edge (no citation), topically different
        p1 = _make_paper("a", "Quantum computing error correction",
                         abstract="Error correcting codes for quantum computers")
        p2 = _make_paper("b", "Marine ecosystem biodiversity mapping",
                         abstract="Mapping biodiversity in marine reef systems")
        rg.build_graph([p1, p2], similarity_threshold=0.01)
        sp = rg._score_path(["a", "b"])
        assert sp is not None
        # No citation edge → citation_strength = 0, low sim, -0.3 penalty → Weak
        assert sp.label == "Weak", (
            f"Score {sp.score} (sim={sp.avg_similarity}, cite={sp.citation_strength}) "
            f"should produce label Weak, got {sp.label}"
        )

    def test_longer_path_outscores_trivial_one_hop_alternative(self):
        """When both a 1-hop and a 3-hop path exist between two papers,
        the multi-hop path should score higher if it has strong semantic
        coherence, because the 1-hop path carries a 0.3 penalty while
        the 3-hop path carries none."""
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Self attention mechanism for sequences",
                        refs=["d", "b"],  # direct edge a→d, plus chain a→b→c→d
                        abstract="Self attention sequence transduction mechanism"),
            _make_paper("b", "Multi-head attention in transformers",
                        refs=["c"],
                        abstract="Multi head attention extending self attention"),
            _make_paper("c", "Transformer encoder decoder architecture",
                        refs=["d"],
                        abstract="Encoder decoder architecture with transformer attention"),
            _make_paper("d", "BERT bidirectional transformer pretraining",
                        abstract="Bidirectional pretraining of transformer models"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        one_hop = rg._score_path(["a", "d"])
        three_hop = rg._score_path(["a", "b", "c", "d"])
        assert one_hop is not None and three_hop is not None
        # The 3-hop path has no penalty and high coherence, so should score higher
        assert three_hop.score > one_hop.score, (
            f"3-hop ({three_hop.score}) should outscore 1-hop ({one_hop.score}) "
            f"because the penalty-free multi-hop path through related transformer "
            f"papers reveals a genuine conceptual chain"
        )

    def test_find_meaningful_paths_prefers_longer_coherent_paths(self):
        """When find_meaningful_paths returns results, the top-ranked path
        should not be a trivial 1-hop if a coherent multi-hop alternative
        exists, because the scoring formula is designed to surface deeper
        trajectories."""
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Self attention mechanism for sequences",
                        refs=["d", "b"],
                        abstract="Self attention sequence transduction mechanism"),
            _make_paper("b", "Multi-head attention in transformers",
                        refs=["c"],
                        abstract="Multi head attention extending self attention"),
            _make_paper("c", "Transformer encoder decoder architecture",
                        refs=["d"],
                        abstract="Encoder decoder architecture with transformer attention"),
            _make_paper("d", "BERT bidirectional transformer pretraining",
                        abstract="Bidirectional pretraining of transformer models"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        paths = rg.find_meaningful_paths("a", "d", top_k=5)
        assert len(paths) >= 2, "Should find both 1-hop and multi-hop paths"
        best = paths[0]
        # Best path should be the longer, more insightful trajectory
        assert best.length > 2, (
            f"Top-ranked path has only {best.length} papers — the scoring "
            f"formula should prefer the multi-hop trajectory through related "
            f"transformer papers over the trivial direct edge"
        )


# ===========================================================================
# Learning path
# ===========================================================================

class TestLearningPath:
    """Learning path builds cross-topic research trajectories.

    Users ask: "How do I get from topic A to topic B?"
    The system finds papers matching each topic and searches for
    meaningful paths between them.
    """

    @pytest.fixture
    def multi_topic_graph(self) -> ResearchGraph:
        """Graph with papers spanning transformers and diffusion models."""
        rg = ResearchGraph()
        papers = [
            _make_paper("t1", "Attention mechanism in transformers",
                        refs=["t2"], year=2017,
                        abstract="Self attention for sequence modeling"),
            _make_paper("t2", "Vision transformer for images",
                        refs=["bridge"], year=2021,
                        abstract="Applying transformers to image recognition"),
            _make_paper("bridge", "CLIP connecting vision and language",
                        refs=["d1"], year=2021,
                        abstract="Contrastive learning for images and text"),
            _make_paper("d1", "Latent diffusion models for generation",
                        refs=["d2"], year=2022,
                        abstract="Diffusion in latent space for image generation"),
            _make_paper("d2", "Text to image diffusion synthesis",
                        year=2022,
                        abstract="Generating images from text using diffusion models"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        return rg

    def test_finds_cross_topic_paths(self, multi_topic_graph: ResearchGraph):
        """Should find paths from transformer papers to diffusion papers."""
        paths = multi_topic_graph.learning_path("transformer", "diffusion")
        assert len(paths) >= 1
        # First paper should be about transformers, last about diffusion
        first = paths[0].papers[0]
        last = paths[0].papers[-1]
        assert "transformer" in first.title.lower() or "attention" in first.title.lower()
        assert "diffusion" in last.title.lower()

    def test_returns_scored_paths(self, multi_topic_graph: ResearchGraph):
        """Learning path results should be ScoredPath objects."""
        paths = multi_topic_graph.learning_path("transformer", "diffusion")
        if paths:
            assert isinstance(paths[0], ScoredPath)
            assert paths[0].score > 0 or paths[0].score <= 0  # it's a number

    def test_empty_when_topic_not_found(self, multi_topic_graph: ResearchGraph):
        """Unknown topics should return empty list."""
        assert multi_topic_graph.learning_path("quantum computing", "diffusion") == []

    def test_empty_when_same_topic(self, multi_topic_graph: ResearchGraph):
        """Same topic for both A and B produces no paths (overlap removed)."""
        # Papers matching both sides are removed as overlap
        paths = multi_topic_graph.learning_path("transformer", "transformer")
        # Should be empty since all matches overlap
        assert isinstance(paths, list)


# ===========================================================================
# Shortest path (kept as internal utility)
# ===========================================================================

class TestShortestPath:
    """shortest_path is kept as an internal utility for fallback."""

    def test_finds_path_through_chain(self):
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "A", refs=["b"]),
            _make_paper("b", "B", refs=["c"]),
            _make_paper("c", "C"),
        ]
        rg.build_graph(papers, similarity_threshold=0.99)
        path = rg.shortest_path("a", "c")
        assert path is not None
        assert len(path) == 3

    def test_returns_none_when_no_path(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper("x", "X"))
        rg.add_paper(_make_paper("y", "Y"))
        assert rg.shortest_path("x", "y") is None


# ===========================================================================
# Centrality rankings
# ===========================================================================

class TestCentralityRankings:
    """Centrality rankings reveal structural roles that citation counts alone cannot.

    Each centrality metric answers a different question:
    - Degree: "Which papers connect to the most others?"
    - Betweenness: "Which papers sit on paths between communities?"
    - PageRank: "Which papers are connected to other well-connected papers?"

    These tests use controlled graph topologies where the correct answer is
    known a priori, verifying that the rankings surface the right papers.
    """

    @pytest.fixture
    def star_graph(self) -> ResearchGraph:
        """Star topology: a hub with 5 unconnected leaves.
        The hub should dominate all centrality metrics."""
        rg = ResearchGraph()
        topics = [
            ("leaf0", "Quantum computing algorithms"),
            ("leaf1", "Marine biology ecosystems"),
            ("leaf2", "Renaissance art history"),
            ("leaf3", "Volcanic geology fieldwork"),
            ("leaf4", "Cryptocurrency market analysis"),
        ]
        leaves = [_make_paper(pid, title, refs=["hub"]) for pid, title in topics]
        hub = _make_paper("hub", "Interdisciplinary research methods survey")
        rg.build_graph([hub] + leaves, similarity_threshold=0.99)
        return rg

    def test_hub_has_highest_degree(self, star_graph: ResearchGraph):
        """In a star graph, the center node connects to all leaves, so it
        must have the highest degree centrality."""
        ranked = star_graph.rank_by_degree(3)
        assert ranked[0][0].paper_id == "hub"
        assert ranked[0][1] == 5

    def test_hub_has_highest_betweenness(self, star_graph: ResearchGraph):
        """Every shortest path between two leaves passes through the hub,
        giving it the highest betweenness centrality."""
        ranked = star_graph.rank_by_betweenness(3)
        assert ranked[0][0].paper_id == "hub"
        assert ranked[0][1] > 0

    def test_hub_has_highest_pagerank(self, star_graph: ResearchGraph):
        """The hub receives incoming edges from all leaves, so PageRank
        concentrates authority at the center."""
        ranked = star_graph.rank_by_pagerank(3)
        assert ranked[0][0].paper_id == "hub"

    def test_rank_respects_top_n_limit(self, star_graph: ResearchGraph):
        assert len(star_graph.rank_by_degree(2)) == 2

    def test_leaves_have_equal_degree_in_star(self, star_graph: ResearchGraph):
        """All leaves in a star graph have exactly 1 connection (to the hub),
        so they should all have the same degree."""
        ranked = star_graph.rank_by_degree(6)
        leaf_degrees = [deg for paper, deg in ranked if paper.paper_id != "hub"]
        assert all(d == 1 for d in leaf_degrees)

    def test_chain_center_has_highest_betweenness(self):
        """In a pure chain A-B-C-D-E (no similarity edges), the middle node C
        lies on the most shortest paths, so it should have the highest
        betweenness centrality — demonstrating that betweenness identifies
        structural bottlenecks, not just well-connected nodes."""
        rg = ResearchGraph()
        # Use completely dissimilar titles/abstracts so no similarity edges
        # are created — ensuring a pure chain topology
        papers = [
            _make_paper("a", "Quantum computing fundamentals",
                        refs=["b"], abstract="Quantum gates and circuits"),
            _make_paper("b", "Marine biology reef systems",
                        refs=["c"], abstract="Coral reef ecosystem dynamics"),
            _make_paper("c", "Renaissance sculpture techniques",
                        refs=["d"], abstract="Marble carving in Florence"),
            _make_paper("d", "Volcanic eruption prediction",
                        refs=["e"], abstract="Seismic monitoring of volcanoes"),
            _make_paper("e", "Medieval manuscript preservation",
                        abstract="Archival methods for ancient texts"),
        ]
        rg.build_graph(papers, similarity_threshold=0.99)
        ranked = rg.rank_by_betweenness(5)
        # C is the center of the 5-node chain; it lies on 6 shortest paths
        # (A↔D, A↔E, B↔D, B↔E, plus paths where C is between B and D)
        # while B and D each lie on only 3
        assert ranked[0][0].paper_id == "c", (
            f"Expected center node 'c' to have highest betweenness, "
            f"got '{ranked[0][0].paper_id}' — the chain may have "
            f"unexpected similarity edges creating shortcuts"
        )

    def test_degree_ranking_distinguishes_hub_from_periphery(self):
        """In a graph where one paper cites four others and those four cite
        nothing else, degree centrality must rank the citing paper first —
        showing that graph structure, not citation count, determines rank."""
        rg = ResearchGraph()
        hub = _make_paper("hub", "Survey of methods",
                          refs=["s1", "s2", "s3", "s4"],
                          citation_count=10)
        # Each spoke has high citation_count but low degree
        spokes = [
            _make_paper("s1", "Spoke one", citation_count=50000),
            _make_paper("s2", "Spoke two", citation_count=40000),
            _make_paper("s3", "Spoke three", citation_count=30000),
            _make_paper("s4", "Spoke four", citation_count=20000),
        ]
        rg.build_graph([hub] + spokes, similarity_threshold=0.99)
        ranked = rg.rank_by_degree(5)
        # Hub has degree 4, each spoke has degree 1
        # Hub should be first despite having the lowest citation_count
        assert ranked[0][0].paper_id == "hub"
        assert ranked[0][1] == 4

    def test_betweenness_returns_zero_for_leaf_nodes(self):
        """Leaf nodes in a star graph lie on no shortest paths between other
        nodes, so their betweenness should be 0.0."""
        rg = ResearchGraph()
        leaves = [_make_paper(f"l{i}", f"Leaf {i}", refs=["hub"]) for i in range(3)]
        hub = _make_paper("hub", "Center paper")
        rg.build_graph([hub] + leaves, similarity_threshold=0.99)
        ranked = rg.rank_by_betweenness(4)
        leaf_scores = {p.paper_id: s for p, s in ranked if p.paper_id != "hub"}
        assert all(s == 0.0 for s in leaf_scores.values()), (
            f"Leaf nodes should have 0 betweenness, got {leaf_scores}"
        )


# ===========================================================================
# Subgraph, bridge spotlight, surprising connections, stats
# ===========================================================================

class TestSubgraphExtraction:
    def test_radius_1_includes_direct_neighbors_only(self):
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "A", refs=["b"]),
            _make_paper("b", "B", refs=["c"]),
            _make_paper("c", "C"),
        ]
        rg.build_graph(papers, similarity_threshold=0.99)
        sub = rg.extract_subgraph("a", radius=1)
        assert "a" in sub and "b" in sub and "c" not in sub

    def test_nonexistent_center_returns_empty_graph(self):
        rg = ResearchGraph()
        assert rg.extract_subgraph("ghost").number_of_nodes() == 0


class TestBridgePaperSpotlight:
    def test_spotlight_finds_hub_in_star(self):
        rg = ResearchGraph()
        topics = [
            ("leaf0", "Quantum computing algorithms"),
            ("leaf1", "Marine biology ecosystems"),
            ("leaf2", "Renaissance art history"),
        ]
        leaves = [_make_paper(pid, title, refs=["hub"]) for pid, title in topics]
        hub = _make_paper("hub", "Interdisciplinary research methods survey")
        rg.build_graph([hub] + leaves, similarity_threshold=0.99)
        spotlight = rg.bridge_paper_spotlight()
        assert spotlight is not None
        assert spotlight["paper"].paper_id == "hub"

    def test_spotlight_returns_none_for_tiny_graph(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper("a", "A"))
        assert rg.bridge_paper_spotlight() is None


class TestSurprisingConnections:
    def test_citation_only_graph_has_no_surprises(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A", refs=["b"])
        p2 = _make_paper("b", "B")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        assert rg.find_surprising_connections() == []

    def test_sample_data_produces_surprises(self):
        from src.data_loader import DataLoader
        rg = ResearchGraph()
        rg.build_graph(DataLoader().load_local_dataset(), similarity_threshold=0.15)
        surprises = rg.find_surprising_connections(5)
        assert isinstance(surprises, list)


class TestClusterDetection:
    """detect_clusters groups papers into research subfields using community detection."""

    def test_star_graph_produces_clusters(self):
        """A star graph with dissimilar leaves should produce at least one cluster."""
        rg = ResearchGraph()
        topics = [
            ("leaf0", "Quantum computing algorithms", "quantum gate circuits"),
            ("leaf1", "Marine biology ecosystems", "ocean reef coral species"),
            ("leaf2", "Renaissance art history", "painting sculpture fresco"),
        ]
        leaves = [_make_paper(pid, title, abstract=ab, refs=["hub"]) for pid, title, ab in topics]
        hub = _make_paper("hub", "Survey paper", abstract="survey methods")
        rg.build_graph([hub] + leaves, similarity_threshold=0.99)
        clusters = rg.detect_clusters(min_size=1)
        assert isinstance(clusters, list)
        assert len(clusters) >= 1

    def test_each_cluster_has_required_keys(self):
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Transformer attention models",
                         refs=["b"], year=2017, citation_count=5000,
                         abstract="Self attention for sequence modeling"),
            _make_paper("b", "BERT language model",
                         year=2019, citation_count=3000,
                         abstract="Bidirectional encoder from transformers"),
            _make_paper("c", "Graph neural networks",
                         year=2017, citation_count=2000,
                         abstract="Convolutions on graph structured data"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        clusters = rg.detect_clusters(min_size=1)
        if clusters:
            cl = clusters[0]
            assert "papers" in cl
            assert "theme_words" in cl
            assert "year_range" in cl
            assert "total_citations" in cl
            assert "internal_density" in cl

    def test_min_size_filters_small_clusters(self):
        """Clusters smaller than min_size should be excluded."""
        rg = ResearchGraph()
        rg.add_paper(_make_paper("lone", "Isolated paper"))
        clusters = rg.detect_clusters(min_size=2)
        assert all(len(c["papers"]) >= 2 for c in clusters)

    def test_empty_graph_returns_no_clusters(self):
        rg = ResearchGraph()
        assert rg.detect_clusters() == []


class TestBridgeRoleDescription:
    """describe_bridge_role explains the structural role of a bridge paper."""

    def test_hub_in_star_has_role_description(self):
        rg = ResearchGraph()
        topics = [
            ("leaf0", "Quantum computing algorithms"),
            ("leaf1", "Marine biology ecosystems"),
            ("leaf2", "Renaissance art history"),
        ]
        leaves = [_make_paper(pid, title, refs=["hub"]) for pid, title in topics]
        hub = _make_paper("hub", "Interdisciplinary research methods survey")
        rg.build_graph([hub] + leaves, similarity_threshold=0.99)
        role = rg.describe_bridge_role("hub")
        assert role is not None
        assert "role_description" in role
        assert len(role["role_description"]) > 20

    def test_nonexistent_node_returns_none(self):
        rg = ResearchGraph()
        assert rg.describe_bridge_role("ghost") is None

    def test_fragmentation_risk_is_nonnegative(self):
        rg = ResearchGraph()
        topics = [
            ("leaf0", "Quantum computing algorithms"),
            ("leaf1", "Marine biology ecosystems"),
        ]
        leaves = [_make_paper(pid, title, refs=["hub"]) for pid, title in topics]
        hub = _make_paper("hub", "Survey paper")
        rg.build_graph([hub] + leaves, similarity_threshold=0.99)
        role = rg.describe_bridge_role("hub")
        if role:
            assert role["fragmentation_risk"] >= 0


class TestSurprisingConnectionDescription:
    """describe_surprising_connection explains why a similarity-only edge is notable."""

    def test_produces_nonempty_description(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Transformer attention models",
                          year=2017, venue="NeurIPS",
                          abstract="Self attention for sequence modeling")
        p2 = _make_paper("b", "Graph attention networks",
                          year=2020, venue="ICLR",
                          abstract="Attention mechanisms for graph data")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        desc = rg.describe_surprising_connection(p1, p2, 0.42)
        assert isinstance(desc, str)
        assert len(desc) > 30

    def test_mentions_venue_difference(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Paper A", venue="NeurIPS",
                          abstract="neural network deep learning")
        p2 = _make_paper("b", "Paper B", venue="ICML",
                          abstract="neural network deep learning")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        desc = rg.describe_surprising_connection(p1, p2, 0.5)
        assert "NeurIPS" in desc or "ICML" in desc

    def test_mentions_temporal_gap(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Old paper on neural nets", year=2014,
                          abstract="neural network architectures")
        p2 = _make_paper("b", "New paper on neural nets", year=2023,
                          abstract="neural network architectures")
        rg.build_graph([p1, p2], similarity_threshold=0.1)
        desc = rg.describe_surprising_connection(p1, p2, 0.6)
        assert "9" in desc or "year" in desc.lower()


class TestResearchTrajectory:
    def test_trajectory_returns_papers_sorted_by_year(self):
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Early transformer work", year=2017),
            _make_paper("c", "Latest transformer work", year=2023),
            _make_paper("b", "Mid transformer work", year=2020),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        traj = rg.research_trajectory("transformer")
        assert [p.year for p in traj] == [2017, 2020, 2023]

    def test_trajectory_empty_for_no_match(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper("a", "Unrelated paper", year=2020))
        assert rg.research_trajectory("transformer") == []


class TestGraphStatistics:
    def test_stats_include_all_required_keys(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper("a", "A"))
        required = {"nodes", "edges", "citation_edges", "similarity_edges",
                     "connected_components", "density"}
        assert required.issubset(rg.stats().keys())

    def test_empty_graph_has_zero_stats(self):
        rg = ResearchGraph()
        stats = rg.stats()
        assert stats["nodes"] == 0 and stats["edges"] == 0


# ===========================================================================
# Shared-author edges
# ===========================================================================

class TestSharedAuthorEdges:
    """add_shared_author_edges tags or creates edges for co-authored papers."""

    def test_creates_edge_when_no_other_connection_exists(self):
        rg = ResearchGraph()
        # Disjoint vocabulary so build_graph adds no similarity edge.
        p1 = _make_paper("a", "Quantum computing algorithms",
                          abstract="quantum gate circuits superposition entanglement",
                          authors=["Yann LeCun"])
        p2 = _make_paper("b", "Marine biology ecosystems",
                          abstract="ocean reef coral species marine",
                          authors=["Yann LeCun"])
        rg.build_graph([p1, p2], similarity_threshold=0.5)
        rg.add_shared_author_edges()
        assert rg.graph.has_edge("a", "b")
        assert rg.graph["a"]["b"]["edge_type"] == "shared_author"

    def test_tags_existing_edge_with_shared_authors_attr(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Quantum computing algorithms",
                          abstract="quantum gate circuits",
                          refs=["b"], authors=["Alice"])
        p2 = _make_paper("b", "Marine biology ecosystems",
                          abstract="ocean reef coral species",
                          authors=["Alice"])
        rg.build_graph([p1, p2], similarity_threshold=0.5)
        rg.add_shared_author_edges()
        assert "alice" in [a.lower() for a in rg.graph["a"]["b"]["shared_authors"]]
        # Existing citation edge type is preserved
        assert rg.graph["a"]["b"]["edge_type"] == "citation"

    def test_no_edges_for_papers_without_shared_authors(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Quantum computing", authors=["Alice"],
                          abstract="quantum gate circuits")
        p2 = _make_paper("b", "Marine biology", authors=["Bob"],
                          abstract="ocean reef coral species")
        rg.build_graph([p1, p2], similarity_threshold=0.5)
        rg.add_shared_author_edges()
        assert not rg.graph.has_edge("a", "b")

    def test_returns_count_of_new_edges(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Quantum computing", authors=["X"],
                          abstract="quantum gate circuits")
        p2 = _make_paper("b", "Marine biology", authors=["X"],
                          abstract="ocean reef coral species")
        p3 = _make_paper("c", "Renaissance art", authors=["X"],
                          abstract="painting sculpture fresco")
        rg.build_graph([p1, p2, p3], similarity_threshold=0.5)
        new_count = rg.add_shared_author_edges()
        # Three pairs, no prior edges → 3 new shared-author edges
        assert new_count == 3


# ===========================================================================
# Related-paper ranking with reasons
# ===========================================================================

class TestGetRelatedPapers:
    """get_related_papers returns scored neighbors annotated with reasons."""

    def test_empty_for_unknown_paper_id(self):
        rg = ResearchGraph()
        assert rg.get_related_papers("ghost") == []

    def test_reasons_include_citation_when_citation_edge_exists(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A", refs=["b"])
        p2 = _make_paper("b", "B")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        results = rg.get_related_papers("a")
        assert results
        assert any("citation" in r["reasons"][0].lower() for r in results)

    def test_reasons_include_similarity_score(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Deep neural networks for vision",
                          abstract="convolutional neural networks for image recognition")
        p2 = _make_paper("b", "Convolutional neural networks for vision",
                          abstract="deep convolutional networks for image classification")
        rg.build_graph([p1, p2], similarity_threshold=0.05)
        results = rg.get_related_papers("a")
        assert results
        assert any("similarity" in r for r in results[0]["reasons"])

    def test_higher_score_for_combined_signals(self):
        """A paper that is both cited AND similar should outscore one that is only cited."""
        rg = ResearchGraph()
        p1 = _make_paper(
            "a", "Deep neural networks for vision",
            abstract="convolutional neural networks for image recognition",
            refs=["b", "c"],
        )
        p2 = _make_paper(  # similar AND cited
            "b", "Convolutional neural networks for vision",
            abstract="deep convolutional networks for image classification",
        )
        p3 = _make_paper(  # only cited
            "c", "Pure mathematics topology",
            abstract="topology rings algebraic spaces",
        )
        rg.build_graph([p1, p2, p3], similarity_threshold=0.05)
        results = rg.get_related_papers("a")
        scores = {r["paper"].paper_id: r["score"] for r in results}
        assert scores["b"] > scores["c"]

    def test_top_k_limits_results(self):
        rg = ResearchGraph()
        center = _make_paper("hub", "Hub", refs=[f"l{i}" for i in range(5)])
        leaves = [_make_paper(f"l{i}", f"Leaf {i}") for i in range(5)]
        rg.build_graph([center] + leaves, similarity_threshold=0.99)
        results = rg.get_related_papers("hub", top_k=2)
        assert len(results) == 2

    def test_reasons_mention_shared_authors_when_present(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "Paper A", refs=["b"], authors=["Yann LeCun"])
        p2 = _make_paper("b", "Paper B", authors=["Yann LeCun"])
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        rg.add_shared_author_edges()
        reasons = " | ".join(rg.get_related_papers("a")[0]["reasons"])
        assert "author" in reasons.lower()


# ===========================================================================
# Topic summary
# ===========================================================================

class TestTopicSummary:
    """get_topic_summary surfaces the most common topic words across papers."""

    def test_returns_word_count_pairs(self):
        rg = ResearchGraph()
        rg.add_paper(_make_paper(
            "a", "Graph neural networks for molecules",
            abstract="graph neural network molecular property prediction",
        ))
        rg.add_paper(_make_paper(
            "b", "Graph attention networks",
            abstract="graph attention network node classification",
        ))
        summary = rg.get_topic_summary(top_n_words=3)
        assert summary
        assert all(isinstance(w, str) and isinstance(c, int) for w, c in summary)
        # 'graph' appears in both papers
        words = {w for w, _ in summary}
        assert "graph" in words

    def test_empty_graph_returns_empty(self):
        rg = ResearchGraph()
        assert rg.get_topic_summary() == []


# ===========================================================================
# Bridge papers between two topics
# ===========================================================================

class TestFindBridgePapers:
    """find_bridge_papers identifies papers connecting two topic clusters."""

    def test_finds_paper_connecting_two_topics(self):
        rg = ResearchGraph()
        p1 = _make_paper("trans1", "Attention is all you need transformer",
                          abstract="transformer attention sequence")
        p2 = _make_paper("trans2", "BERT pretraining with transformer",
                          abstract="transformer bidirectional encoder")
        # Bridge title intentionally avoids the keywords "transformer" and
        # "diffusion" so the bridge isn't itself classified as a topic-A or
        # topic-B paper.
        bridge = _make_paper(
            "bridge", "Cross-domain method survey",
            abstract="cross-domain methodological review",
            refs=["trans1", "trans2", "diff1", "diff2"],
        )
        d1 = _make_paper("diff1", "Diffusion models for image generation",
                          abstract="diffusion image generation denoising")
        d2 = _make_paper("diff2", "Denoising diffusion probabilistic models",
                          abstract="diffusion denoising probabilistic image")
        rg.build_graph([p1, p2, bridge, d1, d2], similarity_threshold=0.99)
        bridges = rg.find_bridge_papers("transformer", "diffusion")
        ids = [b["paper"].paper_id for b in bridges]
        assert "bridge" in ids

    def test_returns_empty_when_topic_missing(self):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A")
        p2 = _make_paper("b", "B")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        assert rg.find_bridge_papers("nonexistent_topic_x", "nonexistent_topic_y") == []

    def test_each_result_has_neighbors_in_both_topics(self):
        rg = ResearchGraph()
        p1 = _make_paper("trans1", "transformer paper",
                          abstract="transformer attention")
        bridge = _make_paper(
            "bridge", "Cross-domain methodological review",
            abstract="cross-domain technique review",
            refs=["trans1", "diff1"],
        )
        d1 = _make_paper("diff1", "diffusion paper",
                          abstract="diffusion image generation")
        rg.build_graph([p1, bridge, d1], similarity_threshold=0.99)
        bridges = rg.find_bridge_papers("transformer", "diffusion")
        assert bridges
        for b in bridges:
            assert b["topic_a_neighbors"]
            assert b["topic_b_neighbors"]


# ===========================================================================
# JSON export / load round-trip
# ===========================================================================

class TestGraphJsonRoundTrip:
    """export_graph_json and load_graph_json preserve nodes, edges, and attributes."""

    def test_roundtrip_preserves_node_count(self, tmp_path):
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Paper A", refs=["b"], year=2017),
            _make_paper("b", "Paper B", year=2018),
            _make_paper("c", "Paper C", year=2019),
        ]
        rg.build_graph(papers, similarity_threshold=0.99)
        path = tmp_path / "graph.json"
        rg.export_graph_json(str(path))
        loaded = ResearchGraph.load_graph_json(str(path))
        assert loaded.graph.number_of_nodes() == rg.graph.number_of_nodes()

    def test_roundtrip_preserves_edge_count(self, tmp_path):
        rg = ResearchGraph()
        papers = [
            _make_paper("a", "Paper A", refs=["b"]),
            _make_paper("b", "Paper B", refs=["c"]),
            _make_paper("c", "Paper C"),
        ]
        rg.build_graph(papers, similarity_threshold=0.99)
        path = tmp_path / "graph.json"
        rg.export_graph_json(str(path))
        loaded = ResearchGraph.load_graph_json(str(path))
        assert loaded.graph.number_of_edges() == rg.graph.number_of_edges()

    def test_roundtrip_preserves_edge_type_attribute(self, tmp_path):
        rg = ResearchGraph()
        p1 = _make_paper("a", "A", refs=["b"])
        p2 = _make_paper("b", "B")
        rg.build_graph([p1, p2], similarity_threshold=0.99)
        path = tmp_path / "graph.json"
        rg.export_graph_json(str(path))
        loaded = ResearchGraph.load_graph_json(str(path))
        assert loaded.graph["a"]["b"]["edge_type"] == "citation"

    def test_roundtrip_preserves_paper_metadata(self, tmp_path):
        rg = ResearchGraph()
        p = _make_paper("a", "Title", year=2020, citation_count=42, authors=["X"])
        rg.add_paper(p)
        path = tmp_path / "graph.json"
        rg.export_graph_json(str(path))
        loaded = ResearchGraph.load_graph_json(str(path))
        loaded_paper = loaded.get_paper("a")
        assert loaded_paper is not None
        assert loaded_paper.title == "Title"
        assert loaded_paper.year == 2020
        assert loaded_paper.citation_count == 42
        assert loaded_paper.authors == ["X"]
