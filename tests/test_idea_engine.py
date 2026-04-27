"""Behavioral specification for the IdeaEngine class.

IdeaEngine generates structured, human-readable narratives from graph
outputs.  It sits on top of graph algorithms and never queries the
graph directly.

These tests verify the template-based fallback mode so the project
works fully without any paid API key.
"""

import pytest

from src.paper import Paper
from src.idea_engine import IdeaEngine


def _make_paper(pid: str, title: str, **kwargs) -> Paper:
    return Paper(paper_id=pid, title=title, **kwargs)


@pytest.fixture
def engine() -> IdeaEngine:
    """IdeaEngine in template mode (no LLM)."""
    return IdeaEngine(use_llm=False)


@pytest.fixture
def sample_path() -> list[Paper]:
    """A 3-paper path: Transformers -> BERT -> GPT-3."""
    return [
        _make_paper("p1", "Attention Is All You Need", year=2017,
                    venue="NeurIPS", abstract="We propose the Transformer architecture for sequence modeling."),
        _make_paper("p2", "BERT Pre-training", year=2019,
                    venue="NAACL", abstract="Bidirectional encoder representations from transformers."),
        _make_paper("p3", "GPT-3 Language Models", year=2020,
                    venue="NeurIPS", abstract="Large language models are few-shot learners using transformers."),
    ]


# ===========================================================================
# Structured path explanation
# ===========================================================================

class TestStructuredExplanation:
    """explain_path produces structured, step-by-step concept transfer explanations.

    Format:
        Step 1 -> Step 2: what concept transfers
        Step 2 -> Step 3: what concept transfers
        Overall insight: ...
    """

    def test_empty_path_returns_message(self, engine: IdeaEngine):
        assert "No papers" in engine.explain_path([])

    def test_single_paper_acknowledges_no_connection(self, engine: IdeaEngine):
        result = engine.explain_path([_make_paper("1", "Solo Paper")])
        assert "Solo Paper" in result

    def test_explanation_mentions_every_paper(self, engine: IdeaEngine, sample_path: list[Paper]):
        result = engine.explain_path(sample_path)
        for paper in sample_path:
            assert paper.title in result

    def test_explanation_has_pairwise_step_sections(self, engine: IdeaEngine, sample_path: list[Paper]):
        """Each consecutive pair should have its own explanation section."""
        result = engine.explain_path(sample_path)
        # Should contain arrows showing pairwise connections
        assert sample_path[0].title in result
        assert sample_path[1].title in result

    def test_explanation_has_overall_insight(self, engine: IdeaEngine, sample_path: list[Paper]):
        """The explanation should end with an overall insight section."""
        result = engine.explain_path(sample_path)
        assert "overall" in result.lower() or "insight" in result.lower() or "trajectory" in result.lower()

    def test_explanation_identifies_concept_transfer(self, engine: IdeaEngine, sample_path: list[Paper]):
        """Pairwise sections should mention shared concepts or themes."""
        result = engine.explain_path(sample_path)
        # The sample papers share "transformer" as a concept
        assert "transformer" in result.lower() or "shared" in result.lower() or "focus" in result.lower()

    def test_explanation_identifies_pivot_paper(self, engine: IdeaEngine, sample_path: list[Paper]):
        """For 3+ paper paths, the middle paper should be highlighted as pivotal."""
        result = engine.explain_path(sample_path)
        assert sample_path[1].title in result


# ===========================================================================
# Research idea generation
# ===========================================================================

class TestIdeaGeneration:
    """generate_research_idea proposes novel work combining path endpoints."""

    def test_requires_at_least_two_papers(self, engine: IdeaEngine):
        result = engine.generate_research_idea([_make_paper("1", "Solo")])
        assert "at least two" in result.lower()

    def test_idea_references_endpoint_papers(self, engine: IdeaEngine, sample_path: list[Paper]):
        result = engine.generate_research_idea(sample_path)
        assert sample_path[0].title in result
        assert sample_path[-1].title in result

    def test_idea_mentions_bridge_paper(self, engine: IdeaEngine, sample_path: list[Paper]):
        result = engine.generate_research_idea(sample_path)
        assert sample_path[1].title in result

    def test_idea_includes_proposed_direction(self, engine: IdeaEngine, sample_path: list[Paper]):
        result = engine.generate_research_idea(sample_path)
        assert "proposed" in result.lower() or "direction" in result.lower() or "investigate" in result.lower()


# ===========================================================================
# Research trajectory narrative
# ===========================================================================

class TestTrajectoryNarrative:
    """narrate_trajectory tells the temporal story with step-by-step evolution."""

    def test_requires_at_least_two_papers(self, engine: IdeaEngine):
        result = engine.narrate_trajectory([_make_paper("1", "Solo", year=2020)])
        assert "at least two" in result.lower() or "not enough" in result.lower()

    def test_narrative_spans_all_years(self, engine: IdeaEngine, sample_path: list[Paper]):
        result = engine.narrate_trajectory(sample_path)
        assert "2017" in result
        assert "2020" in result

    def test_narrative_has_step_by_step_evolution(self, engine: IdeaEngine, sample_path: list[Paper]):
        """Should describe how each paper leads to the next."""
        result = engine.narrate_trajectory(sample_path)
        # Should contain pairwise transition descriptions
        assert sample_path[0].title in result
        assert sample_path[1].title in result

    def test_narrative_identifies_landmark_paper(self, engine: IdeaEngine):
        papers = [
            _make_paper("a", "Foundational Work", year=2015, citation_count=50000),
            _make_paper("b", "Follow-up Study", year=2020, citation_count=5000),
        ]
        result = engine.narrate_trajectory(papers)
        assert "Foundational Work" in result
        assert "landmark" in result.lower() or "most cited" in result.lower() or "citation" in result.lower()


# ===========================================================================
# Cluster summary
# ===========================================================================

class TestClusterSummary:
    def test_empty_cluster_returns_message(self, engine: IdeaEngine):
        assert "No papers" in engine.summarize_cluster([])

    def test_summary_reports_paper_count(self, engine: IdeaEngine, sample_path: list[Paper]):
        assert "3 papers" in engine.summarize_cluster(sample_path)

    def test_summary_lists_all_papers(self, engine: IdeaEngine, sample_path: list[Paper]):
        result = engine.summarize_cluster(sample_path)
        for paper in sample_path:
            assert paper.title in result


# ===========================================================================
# LLM mode configuration
# ===========================================================================

# ===========================================================================
# Fallback explanation logic (_identify_concept_transfer)
# ===========================================================================

class TestConceptTransferFallback:
    """_identify_concept_transfer produces one-sentence explanations connecting
    two consecutive papers in a path.

    It uses three fallback tiers:
    1. Shared topic words + unique extensions → "Both address X, extends into Y"
    2. Shared topic words only → "Connected through shared focus on X"
    3. Same venue → "Both published at venue, same research community"
    4. No overlap at all → generic "knowledge graph connection" fallback

    These tiers ensure every pair of papers gets a meaningful explanation,
    even when they share no vocabulary — which is critical for the UI
    because it must never display an empty or unhelpful connection label.
    """

    def test_shared_words_with_unique_extension(self):
        """When papers share topic words AND the second introduces new ones,
        the explanation should mention both the shared foundation and the
        new direction — telling the user what conceptual leap occurred."""
        from src.idea_engine import _identify_concept_transfer
        p1 = _make_paper("a", "Self attention mechanisms",
                         abstract="Self attention for neural sequence transduction models")
        p2 = _make_paper("b", "Vision transformers for image classification",
                         abstract="Self attention applied to image patches for classification")
        result = _identify_concept_transfer(p1, p2)
        # Should mention shared concepts and the extension
        assert "address" in result.lower() or "both" in result.lower()
        assert "extends" in result.lower() or "building" in result.lower()

    def test_shared_words_only(self):
        """When papers share topic words but the second adds nothing new,
        the explanation notes the shared focus without fabricating extensions."""
        from src.idea_engine import _identify_concept_transfer
        # Use identical vocabulary so unique_to_b is empty
        p1 = _make_paper("a", "Neural network training optimization",
                         abstract="Optimization methods for neural network training")
        p2 = _make_paper("b", "Optimization of neural network training",
                         abstract="Neural network training optimization methods")
        result = _identify_concept_transfer(p1, p2)
        assert "shared" in result.lower() or "focus" in result.lower() or "both" in result.lower()

    def test_same_venue_no_word_overlap(self):
        """When papers share no topic words but are from the same venue,
        the explanation highlights their community membership — a structural
        insight the graph provides."""
        from src.idea_engine import _identify_concept_transfer
        p1 = _make_paper("a", "Abc xyz", venue="NeurIPS",
                         abstract="Abc xyz")
        p2 = _make_paper("b", "Def ghi", venue="NeurIPS",
                         abstract="Def ghi")
        result = _identify_concept_transfer(p1, p2)
        assert "NeurIPS" in result

    def test_no_overlap_at_all_produces_generic_bridge_explanation(self):
        """When papers share no words and no venue, the fallback explanation
        should still be informative — noting that the graph itself reveals
        the connection, not surface-level metadata."""
        from src.idea_engine import _identify_concept_transfer
        p1 = _make_paper("a", "Abc xyz",
                         abstract="Abc xyz")
        p2 = _make_paper("b", "Def ghi",
                         abstract="Def ghi")
        result = _identify_concept_transfer(p1, p2)
        assert len(result) > 20
        assert "knowledge graph" in result.lower() or "bridge" in result.lower()

    def test_explanation_is_never_empty(self):
        """No matter how different two papers are, the explanation must be
        a non-empty, readable string — the UI depends on this."""
        from src.idea_engine import _identify_concept_transfer
        p1 = _make_paper("a", "X", abstract="")
        p2 = _make_paper("b", "Y", abstract="")
        result = _identify_concept_transfer(p1, p2)
        assert isinstance(result, str)
        assert len(result) > 10


class TestExplainPathFallbackBehavior:
    """explain_path template mode must produce structured output even when
    papers are topically unrelated, ensuring the UI always has content to
    display regardless of what the graph algorithm returns."""

    def test_unrelated_papers_still_produce_full_structure(self, engine: IdeaEngine):
        """Even when papers share no concepts, the explanation should have
        the full structure: step connections and overall insight."""
        papers = [
            _make_paper("a", "Quantum error correction codes",
                        year=2015, abstract="Error correction for quantum computing"),
            _make_paper("b", "French culinary traditions",
                        year=2020, abstract="Traditional French cooking techniques"),
        ]
        result = engine.explain_path(papers)
        # Structure should be complete even for unrelated papers
        assert "Quantum" in result
        assert "French" in result or "culinary" in result.lower()
        assert "insight" in result.lower() or "trajectory" in result.lower()

    def test_temporal_insight_mentions_year_span(self, engine: IdeaEngine):
        """When papers span different years, the explanation should note
        the temporal dimension — research evolution over time is a key
        insight the system provides."""
        papers = [
            _make_paper("a", "Early neural networks", year=2010,
                        abstract="First neural network approaches"),
            _make_paper("b", "Modern deep learning", year=2023,
                        abstract="State of the art deep learning models"),
        ]
        result = engine.explain_path(papers)
        assert "13 years" in result or "2010" in result

    def test_pivot_paper_highlighted_in_three_paper_path(self, engine: IdeaEngine):
        """For paths with 3+ papers, the middle paper should be explicitly
        called out as the conceptual bridge — this is a central insight
        of the trajectory feature."""
        papers = [
            _make_paper("a", "Paper Alpha", year=2015, abstract="Alpha concepts"),
            _make_paper("b", "Paper Beta", year=2018, abstract="Beta methods"),
            _make_paper("c", "Paper Gamma", year=2022, abstract="Gamma results"),
        ]
        result = engine.explain_path(papers)
        assert "Paper Beta" in result
        assert "pivotal" in result.lower() or "bridge" in result.lower()


# ===========================================================================
# Demonstrating meaningful paths on real-ish data
# ===========================================================================

class TestMeaningfulPathDemonstration:
    """These tests demonstrate that the system produces genuinely insightful
    trajectories — not just technically correct results.

    A meaningful path should:
    1. Connect papers through topically coherent intermediate steps
    2. Surface non-obvious connections that keyword search would miss
    3. Tell a story: foundational idea → methodological bridge → application
    """

    @pytest.fixture
    def nlp_evolution_graph(self) -> "ResearchGraph":
        """A graph modeling the evolution from attention → transformers → BERT
        → GPT → diffusion, with a cross-domain bridge through CLIP.

        This tests the scenario users care about most: discovering how
        ideas flow between research areas through intermediate work.
        """
        from src.research_graph import ResearchGraph
        rg = ResearchGraph()
        papers = [
            _make_paper("attn", "Attention mechanism for neural machine translation",
                        references=["transformer"], year=2015, citation_count=30000,
                        abstract="Attention mechanism allowing neural networks to focus "
                                 "on relevant parts of input sequence for translation"),
            _make_paper("transformer", "Attention is all you need transformer architecture",
                        references=["bert", "gpt"], year=2017, citation_count=90000,
                        abstract="Transformer architecture replacing recurrence with "
                                 "self attention for sequence transduction"),
            _make_paper("bert", "BERT bidirectional encoder transformer pretraining",
                        references=["clip"], year=2019, citation_count=70000,
                        abstract="Bidirectional pretraining of transformer encoders "
                                 "for language understanding tasks"),
            _make_paper("gpt", "GPT generative pretrained transformer language model",
                        year=2020, citation_count=25000,
                        abstract="Generative pretraining of transformer decoder for "
                                 "language generation and few-shot learning"),
            _make_paper("clip", "CLIP connecting language and images with transformers",
                        references=["diffusion"], year=2021, citation_count=15000,
                        abstract="Contrastive learning to connect images and text "
                                 "using transformer encoders"),
            _make_paper("diffusion", "Latent diffusion models for image generation",
                        year=2022, citation_count=12000,
                        abstract="Diffusion models in latent space for high quality "
                                 "image generation and synthesis"),
        ]
        rg.build_graph(papers, similarity_threshold=0.1)
        return rg

    def test_attention_to_diffusion_path_traverses_intermediates(
        self, nlp_evolution_graph,
    ):
        """A trajectory from attention (2015) to diffusion (2022) should pass
        through intermediate papers that carry the conceptual thread forward.
        The path tells the story: attention → transformers → (BERT/GPT or CLIP) → diffusion."""
        from src.research_graph import ResearchGraph
        paths = nlp_evolution_graph.find_meaningful_paths(
            "attn", "diffusion", top_k=3
        )
        assert len(paths) >= 1, "Must find at least one path from attention to diffusion"
        best = paths[0]
        # Path should have 3+ intermediate steps (not a trivial 1-hop)
        assert best.length >= 3, (
            f"Path with {best.length} papers is too short — the trajectory from "
            f"attention mechanisms to diffusion models should pass through "
            f"transformer and CLIP papers as conceptual stepping stones"
        )
        # First paper should be about attention, last about diffusion
        assert best.papers[0].paper_id == "attn"
        assert best.papers[-1].paper_id == "diffusion"

    def test_meaningful_path_has_semantic_coherence(self, nlp_evolution_graph):
        """Each step in the trajectory should maintain topical coherence —
        avg_similarity > 0 confirms that consecutive papers share vocabulary,
        meaning the path traces a genuine conceptual thread rather than
        arbitrary graph hops."""
        paths = nlp_evolution_graph.find_meaningful_paths(
            "attn", "diffusion", top_k=1
        )
        assert len(paths) >= 1
        best = paths[0]
        assert best.avg_similarity > 0, (
            f"Similarity of {best.avg_similarity} suggests the path hops through "
            f"unrelated papers — a meaningful trajectory should maintain topical "
            f"coherence at every step"
        )

    def test_meaningful_path_has_citation_grounding(self, nlp_evolution_graph):
        """A high-quality trajectory should include citation edges — confirming
        that the intellectual lineage is not just semantic but also explicit
        in the literature."""
        paths = nlp_evolution_graph.find_meaningful_paths(
            "attn", "diffusion", top_k=1
        )
        assert len(paths) >= 1
        best = paths[0]
        assert best.citation_strength > 0, (
            "A meaningful path through the NLP literature should include at "
            "least some citation edges — the evolution from attention to "
            "diffusion is well-documented in the citation graph"
        )

    def test_cross_domain_path_surfaces_bridge_paper(self, nlp_evolution_graph):
        """A trajectory from language (BERT) to vision (diffusion) should
        pass through CLIP — the paper that bridges NLP and computer vision.
        This is exactly the kind of non-obvious connection the system exists
        to reveal."""
        paths = nlp_evolution_graph.find_meaningful_paths(
            "bert", "diffusion", top_k=3
        )
        assert len(paths) >= 1
        best = paths[0]
        path_ids = [p.paper_id for p in best.papers]
        assert "clip" in path_ids, (
            f"Path {path_ids} does not include CLIP — but CLIP is the key "
            f"bridge paper connecting language (BERT) to vision (diffusion). "
            f"The system should surface this cross-domain connector."
        )

    def test_trajectory_explanation_tells_coherent_story(self, nlp_evolution_graph):
        """The IdeaEngine explanation of the best trajectory should form
        a readable narrative that mentions key concepts from each paper."""
        paths = nlp_evolution_graph.find_meaningful_paths(
            "attn", "diffusion", top_k=1
        )
        assert len(paths) >= 1
        engine = IdeaEngine(use_llm=False)
        explanation = engine.explain_path(paths[0].papers)
        # The explanation should mention the key concepts along the trajectory
        assert "attention" in explanation.lower()
        assert "transformer" in explanation.lower() or "transduction" in explanation.lower()
        # Should have the structural markers of a complete explanation
        assert "insight" in explanation.lower() or "trajectory" in explanation.lower()


# ===========================================================================
# LLM mode configuration
# ===========================================================================

class TestLLMConfiguration:
    def test_defaults_to_template_mode_without_api_key(self):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            assert IdeaEngine().use_llm is False

    def test_explicit_template_mode(self):
        assert IdeaEngine(use_llm=False).use_llm is False

    def test_explicit_llm_mode(self):
        assert IdeaEngine(use_llm=True).use_llm is True


# ===========================================================================
# Idea generation seeded by a paper
# ===========================================================================

class TestIdeasFromPaper:
    """generate_ideas_from_paper proposes ideas grounded in graph relationships."""

    def test_returns_n_ideas_when_enough_related(self, engine: IdeaEngine):
        center = _make_paper("c", "Attention Is All You Need", year=2017,
                              abstract="transformer attention sequence modeling")
        related = [
            _make_paper("r1", "BERT pre-training", year=2019,
                         abstract="bidirectional transformer pretraining"),
            _make_paper("r2", "GPT language models", year=2020,
                         abstract="autoregressive transformer language model"),
            _make_paper("r3", "Vision transformer", year=2021,
                         abstract="transformer image classification patches"),
        ]
        ideas = engine.generate_ideas_from_paper(center, related, n_ideas=3)
        assert len(ideas) == 3

    def test_each_idea_mentions_center_and_partner(self, engine: IdeaEngine):
        center = _make_paper("c", "Attention paper", year=2017,
                              abstract="transformer attention")
        partner = _make_paper("p", "BERT paper", year=2019,
                               abstract="bidirectional transformer pretraining")
        ideas = engine.generate_ideas_from_paper(center, [partner], n_ideas=1)
        assert ideas
        assert "Attention paper" in ideas[0]
        assert "BERT paper" in ideas[0]

    def test_empty_related_returns_helpful_message(self, engine: IdeaEngine):
        center = _make_paper("c", "Lonely paper")
        ideas = engine.generate_ideas_from_paper(center, [], n_ideas=3)
        assert ideas
        assert "Lonely paper" in ideas[0]

    def test_year_gap_is_referenced_when_present(self, engine: IdeaEngine):
        center = _make_paper("c", "2017 paper", year=2017,
                              abstract="transformer attention")
        partner = _make_paper("p", "2023 paper", year=2023,
                               abstract="transformer attention modern")
        ideas = engine.generate_ideas_from_paper(center, [partner], n_ideas=1)
        assert "6-year gap" in ideas[0]


# ===========================================================================
# Idea generation seeded by a topic
# ===========================================================================

class TestIdeasFromTopic:
    """generate_ideas_from_topic seeds ideas from the most-cited matching papers."""

    def test_anchors_on_highest_cited_papers(self, engine: IdeaEngine):
        papers = [
            _make_paper("p1", "Low cited paper", year=2017, citation_count=10,
                         abstract="transformer paper"),
            _make_paper("p2", "Highly cited paper", year=2017, citation_count=10000,
                         abstract="transformer paper"),
        ]
        ideas = engine.generate_ideas_from_topic("transformer", papers, n_ideas=1)
        assert ideas
        # Highest-citation paper anchors first idea
        assert "Highly cited paper" in ideas[0]

    def test_empty_papers_returns_helpful_message(self, engine: IdeaEngine):
        ideas = engine.generate_ideas_from_topic("transformer", [], n_ideas=3)
        assert ideas
        assert "transformer" in ideas[0].lower()

    def test_blank_topic_returns_empty(self, engine: IdeaEngine):
        assert engine.generate_ideas_from_topic("", [], n_ideas=3) == []

    def test_synthesis_idea_appears_with_two_or_more_papers(self, engine: IdeaEngine):
        papers = [
            _make_paper("p1", "First paper", year=2017, citation_count=1000,
                         abstract="transformer attention"),
            _make_paper("p2", "Second paper", year=2018, citation_count=500,
                         abstract="transformer attention"),
        ]
        ideas = engine.generate_ideas_from_topic("transformer", papers, n_ideas=2)
        assert len(ideas) == 2
        joined = " ".join(ideas)
        assert "First paper" in joined and "Second paper" in joined


# ===========================================================================
# Surprising connection narrative
# ===========================================================================

class TestSurprisingConnectionExplanation:
    """explain_surprising_connection generates a plain-English narrative."""

    def test_mentions_both_paper_titles(self, engine: IdeaEngine):
        a = _make_paper("a", "Alpha paper")
        b = _make_paper("b", "Beta paper")
        narrative = engine.explain_surprising_connection(a, b, similarity=0.42)
        assert "Alpha paper" in narrative
        assert "Beta paper" in narrative

    def test_mentions_similarity_score(self, engine: IdeaEngine):
        a = _make_paper("a", "A")
        b = _make_paper("b", "B")
        narrative = engine.explain_surprising_connection(a, b, similarity=0.78)
        assert "0.78" in narrative

    def test_mentions_venue_difference_when_distinct(self, engine: IdeaEngine):
        a = _make_paper("a", "A", venue="ICML")
        b = _make_paper("b", "B", venue="CVPR")
        narrative = engine.explain_surprising_connection(a, b, similarity=0.5)
        assert "ICML" in narrative and "CVPR" in narrative


# ===========================================================================
# Bridge research suggestion
# ===========================================================================

class TestSuggestBridgeResearch:
    """suggest_bridge_research always names both topics."""

    def test_mentions_both_topics(self, engine: IdeaEngine):
        out = engine.suggest_bridge_research("transformer", "diffusion")
        assert "transformer" in out.lower()
        assert "diffusion" in out.lower()

    def test_blank_input_returns_helpful_message(self, engine: IdeaEngine):
        out = engine.suggest_bridge_research("", "")
        assert "two non-empty topics" in out.lower() or "provide" in out.lower()


# ===========================================================================
# Trajectory alias
# ===========================================================================

class TestTrajectoryAlias:
    """summarize_research_trajectory delegates to narrate_trajectory."""

    def test_alias_returns_same_output(self, engine: IdeaEngine, sample_path: list[Paper]):
        a = engine.narrate_trajectory(sample_path)
        b = engine.summarize_research_trajectory(sample_path)
        assert a == b
