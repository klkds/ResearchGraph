# ResearchGraph — SI 507 Final Project Write-up

**Target grade: A.**

## 1. Project summary

ResearchGraph models academic papers as a connected knowledge graph and lets a user explore it through a polished, dashboard-style Streamlit app. Papers are nodes; citations, TF-IDF similarity, and shared authorship are edges. Every interaction in the app — search, neighborhood exploration, trajectory finding, ranking, bridge detection, idea generation — reads from the graph rather than a flat list. The graph is the insight: it surfaces structural roles (hubs, bridges, clusters), latent connections (high similarity with no citation), and multi-hop trajectories that no keyword search could expose.

## 2. Domain and dataset

**Domain.** Modern AI/ML research — transformers, diffusion models, graph neural networks, reinforcement learning, vision-language models, retrieval-augmented generation, contrastive self-supervision, and foundational embeddings.

**Datasets.**
- A bundled JSON dataset (`data/sample_papers.json`) of **70 real papers** spanning 2012–2023. Each record carries id, title, abstract, year, authors, venue, citation count, references, and URL. The reference graph is internally consistent so the citation network is non-trivial offline.
- Live Semantic Scholar API queries (`/graph/v1/paper/search`) with fields `paperId,title,abstract,year,authors,venue,citationCount,references,url`. The free endpoint caps each response at 100 results, so the loader **paginates with `offset`** and dedupes by paper id when the requested limit exceeds 100 (the app defaults to 150 per topic). Cached responses are written to `data/cache/`. No paid API key is required.
- A deterministic builder script (`scripts/build_sample_dataset.py`) is shipped alongside the bundled JSON. It is **idempotent** — running it again with no seed-list changes produces a byte-identical file — and demonstrates how the curated set was assembled.

**Access complexity.** Both data sources flow through the same `DataLoader.normalize_paper_record` method, which handles missing fields, null abstracts, mixed reference formats (string vs. dict), and missing citation counts. API failures fall back to the bundled dataset transparently via a single `requests.RequestException` catch.

## 3. Graph structure

- **Nodes:** every paper in the loaded dataset. Each node carries `title`, `year`, and `citation_count` as attributes for export.
- **Edges:**
  - `citation` — paper A's reference list contains paper B (and both are in the dataset).
  - `similarity` — TF-IDF cosine similarity between the concatenated title + abstract texts exceeds a user-configurable threshold (default 0.15).
  - `both` — the same pair qualifies as citation and similarity.
  - `shared_author` — exposed as a tag on existing edges, and as a new edge type when no other connection exists, when two papers share an author.
- **Edge attributes:** `edge_type`, `weight`, `similarity` (when applicable), `shared_authors` (when applicable).
- **Algorithms used:** BFS shortest path, all-simple-paths with custom scoring, degree / betweenness / PageRank centralities, greedy modularity community detection, component-removal analysis for the bridge spotlight, TF-IDF + cosine similarity for the similarity layer.

## 4. Why a graph is the right structure

A flat list of papers can rank by citation count, but it cannot answer questions like:

- *Which papers bridge two research communities?* — answered by betweenness centrality, then by checking what the graph would look like if the paper were removed.
- *What is the conceptual chain from "Attention Is All You Need" to "Denoising Diffusion Probabilistic Models"?* — answered by enumerating multi-hop simple paths and scoring each by semantic coherence and citation grounding (avg similarity + citation strength − short-path penalty).
- *Which paper pairs share latent intellectual content but no citation link?* — answered by intersecting the similarity edges with the absence of citation edges, plus a narrative that explains the structural distance, venue gap, and topic overlap.
- *Which clusters of papers form natural subfields?* — answered by greedy modularity community detection on the combined citation + similarity graph.

These insights all rely on graph topology, not on any single paper's metadata.

## 5. Object-oriented design

Five modules, four core domain classes, plus a small UI helper module and a separate visualization module:

- **`Paper`** (`src/paper.py`). A dataclass with `__post_init__` validation, `to_dict` / `from_dict` for round-trip serialization, `summary_dict` and `display_authors` for UI display, `similarity_features` for TF-IDF, `topic_words` for cluster analysis, and identity based on `paper_id`.
- **`DataLoader`** (`src/data_loader.py`). Owns the ingestion lifecycle: `fetch_from_semantic_scholar`, `_api_search` (paginated), `load_local_dataset`, `normalize_paper_record`, `save_cache`, `load_cache`. Catches `requests.RequestException` specifically and falls back gracefully.
- **`ResearchGraph`** (`src/research_graph.py`). The central data structure. Methods: `build_graph`, `add_paper`, `add_citation_edges`, `add_similarity_edges`, `add_shared_author_edges`, `get_paper`, `get_neighbors`, `get_related_papers` (returns scores + plain-English reasons), `find_meaningful_paths`, `learning_path`, `shortest_path`, `rank_by_degree` / `_betweenness` / `_pagerank`, `bridge_paper_spotlight`, `find_bridge_papers`, `find_surprising_connections`, `describe_surprising_connection`, `detect_clusters`, `describe_bridge_role`, `research_trajectory`, `get_topic_summary`, `export_graph_json` / `load_graph_json`, `stats`. A `ScoredPath` dataclass captures path metadata.
- **`IdeaEngine`** (`src/idea_engine.py`). Sits on top of graph results, never queries the graph directly. Methods: `explain_path`, `generate_research_idea`, `narrate_trajectory` / `summarize_research_trajectory`, `summarize_cluster`, `generate_ideas_from_paper`, `generate_ideas_from_topic`, `explain_surprising_connection`, `suggest_bridge_research`. Pluggable: tries OpenAI when `OPENAI_API_KEY` is set, falls back to deterministic templates otherwise.
- **`graph_viz`** (`src/graph_viz.py`). pyvis-based HTML rendering plus a complete in-canvas dashboard layer. `render_neighborhood` and `render_path` produce HTML that wraps the pyvis canvas with: a fixed right-side detail sidebar, a top-of-iframe controls bar (Fit / Reset / Labels / Citation filter / Similarity filter), a compact bottom-left legend, status pills (Current focus / Hovered paper / Selected paper), and JS for hover-dim / select-highlight, drag-to-flex dynamic-smooth edges, and a parent-document `<script>` injection bridge that lets a clicked node trigger a Streamlit rerun via a query-param even though the iframe sandbox blocks `top.location` writes.
- **`app_helpers`** (`src/app_helpers.py`). Pure functions extracted from the UI: `quality_badge_html`, `truncate_abstract`, `format_paper_caption`, `path_chain_text`, `trajectory_narrative`, `summarize_neighbor_counts`, `rubric_mapping_rows`. Exists so the helpers can be unit-tested without booting Streamlit.

The classes have clear single responsibilities and a one-direction dependency: `app.py` depends on everything; `IdeaEngine` depends on `Paper`; `ResearchGraph` depends on `Paper` and `utils`; `DataLoader` depends on `Paper` and `utils`; `graph_viz` depends on `ResearchGraph` and `Paper`.

## 6. Interaction modes

The Streamlit app exposes **seven** top-level pages, several with internal tabs:

1. **Home / Project Overview** — a polished landing: hero band with a gradient-tinted background, four live stat tiles (papers / edges / components / density), and a six-card feature grid with one card per other page (including the User Guide).
2. **Search Papers** — keyword search with per-result graph context (degree, citation vs. similarity neighbors, abstract snippet). Results are sorted by citation count.
3. **Graph Explorer** — the interactive workspace.
   - Top Streamlit row: focus dropdown, hop radius (1 / 2 / 3, capped at 60 / 150 nodes), and a back-stack button.
   - In-canvas chrome rendered inside the pyvis iframe: a controls bar with Fit, Reset, Labels toggle, and Citation / Similarity filters; an always-visible right detail sidebar with status pill (blue *Current focus* / amber *Hovered paper* / green *Selected paper*), title, meta row, Authors / Topics / Why connected / Abstract sections, and Set as focus / Open paper actions; a bottom-left legend strip.
   - Force-directed barnesHut physics with an idle-freeze cycle (simulator stabilises on load and halts; a drag wakes it for the duration of the gesture, then re-freezes after release). Hover dims every non-connected edge and node so the eye reads the 1-hop neighborhood instantly.
4. **Research Trajectory** — two tabs: paper-to-paper and topic-to-topic. Each runs the all-simple-paths + scoring algorithm and renders the top-K trajectories with quality badges (Strong / Moderate / Weak), a visual arrow chain, three score metrics (Avg. Similarity, Citation Strength, Path Length), step-by-step concept transfer for every consecutive pair, and an auto-composed "Why this trajectory matters" block.
5. **Insights & Rankings** — five tabs: Hub Papers (degree), Bridge Papers (betweenness), Bridge Spotlight (deep dive with a list of paper pairs that depend on the spotlight for their only connection, plus an embedded sub-graph), Cluster Insights (greedy modularity + density-band callouts), Surprising Connections (similarity-only edges with narrative reasoning).
6. **Idea Generator** — three tabs: from a paper (uses `get_related_papers` for graph-grounded suggestions plus a "Why these papers seeded the ideas" disclosure), from a topic, or to bridge two topics with real bridge-paper candidates pulled from the graph.
7. **User Guide** — a complete walkthrough of every page, every control button (rendered with the actual inline-SVG icons used on the page), every status pill (rendered with the live styling), every legend mark, every interaction, and a closing dataset / similarity-threshold cookbook with band-by-band guidance.

The shell that wraps every page (cool-gray canvas, Inter typography, top header strip with live stats, a workspace card around the canvas, and a scroll-to-top hook on every page change) is part of the experience too: the app reads as a research-intelligence dashboard rather than a default Streamlit demo.

## 7. Testing strategy

`pytest -q` runs **248 behavior-named tests** across six files. Reading the test file headers tells you what the system does:

- **`test_paper.py`** — construction, validation (rejects empty id/title, coerces None/negative citations), display methods (`summary_dict`, `short_label`, `display_authors`), similarity features, topic word extraction, identity semantics, dict round-trip via `to_dict` / `from_dict`.
- **`test_data_loader.py`** — record normalization for messy inputs (null abstract, null citation count, mixed reference formats, empty references), bundled dataset integrity (size, uniqueness, year diversity), cache round-trip, corrupt-cache safety, API fallback on `ConnectionError` and `Timeout` (mocked).
- **`test_research_graph.py`** — node and edge creation, citation / similarity / shared-author edges, paper retrieval, neighbor exploration, meaningful path scoring (penalty constants verified exactly), shortest path edge cases, centrality rankings on canonical topologies (star, chain), subgraph extraction, bridge spotlight, cluster detection, surprising connections, research trajectory, related-paper scoring with reasons, JSON export/load round-trip preserves nodes and edges, topic summary, find_bridge_papers.
- **`test_idea_engine.py`** — path explanation (empty / single / multi paper, pairwise sections, overall insight, pivot identification), research idea generation, trajectory narration (year range, era grouping, landmark identification), cluster summary, paper-seeded idea generation, topic-seeded idea generation, surprising connection narration, bridge research suggestion, trajectory alias.
- **`test_graph_viz.py`** — neighborhood and path rendering produce non-empty HTML; the force-directed physics options and per-node `physics: false` attributes both land in the rendered HTML; the focus node is pinned at the origin; no `doubleClick` handler is present (regression for the old single/double-click flicker bug); the dashboard sidebar / controls bar / legend markup is all present; the focus-swap query-param bridge is wired; tooltip text is title-only (rich metadata moved to the click sidebar); graceful handling of missing nodes / single-node paths.
- **`test_app_helpers.py`** — quality badge color logic, abstract truncation, caption formatting, path chain text, trajectory narrative content, neighbor count bucketing, rubric-mapping completeness.

Tests are deterministic and offline. API tests use `monkeypatch` / `MagicMock`; no real network calls.

## 8. What makes the project ambitious

- **The graph is the insight, not a decoration.** Every page is a question the graph answers; nothing is computed off a flat table.
- **Three integrated edge types** (four if `shared_author` counts) plus a custom path-scoring formula that combines avg similarity, citation strength, and a short-path penalty. Trivial 1-hop paths are explicitly down-weighted.
- **Bridge analysis goes beyond betweenness.** The bridge spotlight runs a hypothetical removal and reports how many components would form, lists which paper pairs would lose their only connection, and explains the role narratively.
- **Surprising connections** specifically target similarity-only edges with no citation — the kind of latent intellectual kinship that no citation index can produce — with a structured narrative about graph distance, venue gap, temporal gap, and topic overlap.
- **Polished, dashboard-style UX.** A cool-gray canvas with Inter type and soft shadows; an always-visible right detail sidebar (not a slide-in tooltip) inside the pyvis iframe; a top controls bar with edge-type filters and a labels toggle; a compact legend that doesn't clip; force-directed physics with an idle-freeze cycle so the canvas never perpetually rotates; dim-on-hover / sticky-on-click highlighting that lets the eye read the 1-hop neighborhood instantly; and a soft blue glow around the focus node as the visual anchor of the view. Mobile-responsive: below 880 px the sidebar drops to a bottom drawer.
- **Engineering details that survived several rewrites.** The focus-swap action ships a script element from the iframe into the parent document so it can navigate the top window even though the Streamlit iframe sandbox blocks `top.location` writes. Global CSS is injected via the same parent-document bridge because Streamlit's markdown sanitiser strips `<style>` tags. Every page-change resets the parent's scroll position via a unique-payload `components.html` so consecutive identical navigations aren't deduped by Streamlit. The deterministic dataset builder is idempotent. The `_api_search` paginates and dedupes when callers ask for more than the API's per-call cap of 100.
- **A complete in-app User Guide** documenting every page, control, badge, icon, and legend mark — rendered with the exact CSS swatches the app uses, so the documentation visually matches what's on screen.
- **A behavior-named test suite** that doubles as living documentation.
- **Optional LLM integration** for richer narratives, with a fully-functional template fallback so the project always works offline and without paid APIs.

## 9. How to run

```
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

The app opens at `http://localhost:8501` on the **Home / Project Overview** landing — hero, live stats, feature cards. The bundled 70-paper dataset is the default; the sidebar can switch to a live Semantic Scholar query (cached on disk after the first run). Tests run offline with no network calls. New users should jump straight to **User Guide** in the sidebar — every control, badge, and icon is documented there.

Optional: rebuild the bundled dataset.

```
python scripts/build_sample_dataset.py
```

Idempotent — running again with no seed-list changes leaves the JSON byte-identical.

## 10. Limitations

- The citation graph is **undirected** for symmetric path exploration; direction is preserved as an attribute but not used for ranking.
- **Semantic Scholar's free tier rate-limits aggressively** (HTTP 429 from `/graph/v1/paper/search`). The loader sends a meaningful `User-Agent`, retries 429s up to four times with exponential backoff that honors the server's `Retry-After` header, and reads `SEMANTIC_SCHOLAR_API_KEY` (or `S2_API_KEY`) from the environment to send as `x-api-key` for the higher authenticated rate limit. When all retries are exhausted, the loader falls back to the bundled JSON and the UI shows an amber **Bundled (live fetch failed)** pill in the top header so the user sees what happened.
- The **TF-IDF similarity model is sensitive to the threshold** parameter and shorter abstracts. The User Guide includes a band-by-band threshold cookbook; sentence-transformer embeddings would be more robust.
- **Author disambiguation** is name-based; the same person under two spellings would be treated as two authors.
- The **bundled dataset is curated** rather than scraped — a deliberate trade-off for grader reproducibility. The live API path exercises the real ingestion pipeline.
- **No persistent user state** across browser sessions; Streamlit's session state lives only as long as the open tab.
