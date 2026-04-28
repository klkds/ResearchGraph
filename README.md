# ResearchGraph: Academic Paper Discovery as a Knowledge Graph

ResearchGraph turns academic papers into a navigable knowledge network. Papers
are nodes. Citations, TF-IDF similarity, and shared authorship are edges. The
result is a force-directed canvas that exposes hubs, bridges, clusters, hidden
connections, and multi-hop research trajectories — the structural insights a
flat keyword search can never reveal.

The app ships as a polished, dashboard-style Streamlit interface with seven
pages, a fixed right-side detail sidebar, in-canvas controls, a calm
cool-gray design system, and a complete in-app **User Guide** documenting
every page, control, status pill, icon, and legend mark.

---

## Why a graph?

Academic knowledge is a network, not a list. Papers cite each other, share
methodologies, and build on common foundations. Keyword search returns an
isolated ranked list. A graph reveals the *structure* of knowledge itself.

| What flat search shows | What the graph reveals |
|---|---|
| A ranked list of papers matching a keyword | **Relationships** between papers (citation, similarity, shared authors) |
| No way to see how two distant topics connect | **Scored multi-hop paths** between any two papers or two topics |
| Citation count as the only importance metric | **Hubs** (degree), **bridges** (betweenness), and **structural keystones** — three structurally distinct roles |
| Only papers that match the query | **Surprising connections** — papers textually similar but never cite each other, visible *only* because the graph adds a similarity layer |
| A static list with no temporal structure | **Research trajectories** — how a topic evolved chronologically through the connected network |

> **Concrete example.** Search would never link *Attention Is All You Need*
> (2017) to *Denoising Diffusion Probabilistic Models* (2020). The graph
> walks the path
> *Attention → Vision Transformer → CLIP → Latent Diffusion → DDPM*,
> revealing how the attention mechanism — originally designed for NLP —
> became foundational to image generation through a chain of adaptations.

---

## Features at a glance

### Seven pages

| # | Page | What it does |
|---|---|---|
| 1 | **Home / Project Overview** | Hero landing with live graph stats and a tour of every feature card. |
| 2 | **Search Papers** | Keyword search across titles + abstracts; each result is annotated with its structural role (citation vs. similarity neighbor counts). |
| 3 | **Graph Explorer** | The interactive workspace. Anchor a paper, drag the network, hover to spotlight 1-hop neighbors, click to lock the inspect sidebar, "Set as focus" to recenter. Top controls bar (Fit / Reset / Labels / Citation filter / Similarity filter), bottom-left legend, status pill (Current focus / Hovered / Selected). |
| 4 | **Research Trajectory** | Scored multi-hop paths between two papers or two topics, with concept-transfer narrative for each step and a Strong / Moderate / Weak quality badge. |
| 5 | **Insights & Rankings** | Five tabs of structural analysis: Hub Papers, Bridge Papers, Bridge Spotlight (with embedded sub-graph), Cluster Insights (greedy modularity + density bands), Surprising Connections. |
| 6 | **Idea Generator** | Three modes — anchor on a paper, anchor on a topic, or bridge two topics. Every idea is grounded in a graph relationship that already exists. |
| 7 | **User Guide** | Detailed walk-through of every page, control, badge, icon, and legend mark. The fastest way to learn the app. |

### UX polish you'll notice

- **Calm cool-gray canvas** with Inter type, soft shadows, rounded
  workspace cards.
- **Always-visible right sidebar** in Graph Explorer — not a
  slide-in tooltip — with explicit sections (Authors, Topics,
  Why connected, Abstract) and compact action buttons.
- **Force-directed barnesHut physics** with an idle-freeze cycle:
  the simulator stabilises on load, then halts; drags wake it for
  the duration of the gesture, then it settles and freezes again.
  No perpetual rotation.
- **Dim-on-hover / highlight-on-select.** Hovered (amber pill) or
  selected (green pill) nodes get their connected edges restored to
  vivid color while every other edge fades to near-transparent grey.
- **Soft blue glow** on the focus node — the anchor of every view.
- **Edge-type filters** in the canvas controls bar. Solid edges =
  citation, dashed = semantic similarity.
- **Bottom-drawer responsive layout** — below ~880 px the sidebar
  drops below the canvas automatically.
- **Scroll resets to top** on every page change.

---

## Architecture

```
ResearchGraph/
├── app.py                       # Streamlit entry point (7 pages)
├── requirements.txt
├── data/
│   ├── sample_papers.json       # 70 curated ML/AI papers (offline-ready)
│   └── cache/                   # Semantic Scholar API cache
├── scripts/
│   └── build_sample_dataset.py  # Deterministic builder for the bundled set
├── src/
│   ├── paper.py                 # Paper dataclass + validation + topic words
│   ├── data_loader.py           # API fetch + paginate + cache + JSON fallback
│   ├── research_graph.py        # NetworkX graph + every algorithm
│   ├── graph_viz.py             # pyvis canvas + dashboard sidebar/controls/legend
│   ├── idea_engine.py           # Template / LLM narrative engine
│   ├── app_helpers.py           # Rubric helpers (testable in isolation)
│   └── utils.py                 # TF-IDF similarity helpers
└── tests/
    ├── test_paper.py
    ├── test_data_loader.py
    ├── test_research_graph.py
    ├── test_graph_viz.py
    ├── test_idea_engine.py
    └── test_app_helpers.py
```

### Core classes

| Class | Responsibility | Key design choice |
|---|---|---|
| `Paper` | Dataclass with validation, display methods, and topic-word extraction. Identity is `paper_id`. | `__post_init__` validates id/title non-empty and coerces None / negative citation counts. |
| `DataLoader` | Fetches from Semantic Scholar (paginated for limits > 100), loads bundled JSON, normalizes messy API records, manages disk cache. | Catches `requests.RequestException` and falls back to the bundled dataset on any API error or rate limit. |
| `ResearchGraph` | Builds and queries the NetworkX graph: citation edges, similarity edges, neighborhoods, multi-hop scored paths, centrality rankings, bridge spotlight, clusters, surprising connections. | Undirected graph for symmetric path exploration; `edge_type` attribute preserves the citation / similarity / both / shared_author distinction. |
| `IdeaEngine` | Generates explanations, ideas, trajectory narratives, and cluster summaries. | Pluggable: tries LLM first, falls back to deterministic templates on any error. Never queries the graph directly. |

### Graph design

- **Three edge types** — `citation` (from reference lists),
  `similarity` (TF-IDF cosine above a configurable threshold), and
  `shared_author`. Edges that qualify as both citation + similarity
  are tagged `both`.
- **Undirected.** Citation directionality is stored as edge data for
  optional filtering. The trade-off — losing "who cited whom" — buys
  symmetric, intuitive path-finding.
- **Per-node attributes** mirrored onto the NetworkX node so the graph
  can be exported and reasoned about without the `Paper` objects.
- **Configurable similarity threshold** in the sidebar (TF-IDF cosine,
  default 0.15). Lower → denser, surfaces latent connections; higher
  → tight clusters, may disconnect distant topics.

### Algorithms used

| Algorithm | Used by |
|---|---|
| BFS / shortest path | Graph Explorer subgraph extraction |
| `nx.all_simple_paths` + custom scoring | Research Trajectory (avg similarity + citation strength − short-path penalty) |
| Degree centrality | Hub Papers ranking |
| Betweenness centrality | Bridge Papers ranking + Bridge Spotlight |
| Component-removal analysis | Bridge Spotlight fragmentation warning |
| Greedy modularity community detection | Cluster Insights, with connected-components fallback |
| TF-IDF + cosine similarity | Similarity edges and topic-similarity scoring |

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/klkds/ResearchGraph
cd ResearchGraph
pip install -r requirements.txt
```

The app runs **fully offline** with the bundled 70-paper dataset and
deterministic template-based idea generation. No paid API keys required.

### Optional: LLM-powered idea generation
```bash
export OPENAI_API_KEY="sk-..."
```
Without this, the Idea Generator uses smart template outputs — fully
functional, just less prosaic.

### Optional: avoid Semantic Scholar rate limits
The free `/graph/v1/paper/search` endpoint throttles anonymous traffic
aggressively (you'll see `HTTP 429` in the terminal and an amber
**Bundled (live fetch failed)** pill in the app header). To unlock the
higher authenticated rate limit, request a free API key from
[Semantic Scholar](https://www.semanticscholar.org/product/api#api-key)
and export it before launching:
```bash
export SEMANTIC_SCHOLAR_API_KEY="..."     # or S2_API_KEY
```
The loader sends it as the `x-api-key` header on every request, retries
429s with backoff that honors the server's `Retry-After`, and pages
through results in 100-paper chunks. With or without a key, the first
successful fetch for a topic is cached on disk under `data/cache/`, so
subsequent loads bypass the API entirely (the header pill turns blue:
**Cached · Semantic Scholar**).

### Optional: rebuild the bundled dataset
```bash
python scripts/build_sample_dataset.py
```
The builder is **idempotent**: re-running with no seed-list changes
produces a byte-identical JSON.

---

## Running the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the left sidebar to switch
datasets, adjust the similarity threshold, and navigate between pages.
**New here?** Jump straight to **User Guide** in the nav — every
control, badge, and icon is documented there.

---

## Running tests

```bash
pytest -q
```

**248 tests** at last count, organised by behavior. The test suite
doubles as living documentation: each test name describes what the
system should do, not just what is being tested.

| File | Covers |
|---|---|
| `test_paper.py` | Construction, validation, equality / hashing, topic-word extraction, display |
| `test_data_loader.py` | Record normalization (complete / missing / null / mixed-shape), bundled JSON integrity, cache round-trips, API fallback on network errors |
| `test_research_graph.py` | Node + edge creation, retrieval, neighbors with edge metadata, shortest path / disconnected / self-loop / missing-node, centrality on a star topology, subgraph extraction, bridge spotlight, surprising connections, JSON round-trip |
| `test_graph_viz.py` | Force-directed physics options present, focus pinned at origin, no `doubleClick` handler (regression for the old flicker bug), in-canvas sidebar / controls / legend markup all present |
| `test_idea_engine.py` | Path explanation, idea generation, trajectory narrative, cluster summary, LLM-mode configuration |
| `test_app_helpers.py` | Rubric-mapping helper functions |

---

## Example workflow

1. `streamlit run app.py` → open `http://localhost:8501`.
2. Land on **Home / Project Overview** — read the hero, eyeball the
   stat tiles (papers, edges, components, density), skim the feature
   cards.
3. Open **Search Papers**, search `"transformer"`, expand a result
   and note its citation vs. similarity neighbor counts.
4. Open **Graph Explorer**, pick *Attention Is All You Need* from the
   focus dropdown:
   - **Hover** any neighbor → status pill turns amber, sidebar
     updates, only that node's edges stay vivid.
   - **Click** the same neighbor → status pill turns green, locked.
   - **Drag** any neighbor → connected edges flex via dynamic-smooth
     physics, then settle and freeze.
   - Click **Labels** in the controls bar to reveal short titles.
   - Toggle **Citation** / **Similarity** filters to see citation-only
     vs. similarity-only structure.
   - Hit **Set as focus** in the sidebar to recenter on the inspected
     paper. Use **← Back** to return.
5. Open **Research Trajectory**, set From = *Attention Is All You
   Need*, To = *Denoising Diffusion Probabilistic Models*. Read the
   scored Strong / Moderate / Weak trajectories and the
   concept-transfer narrative for each hop.
6. Open **Insights & Rankings** → cycle through Hub Papers, Bridge
   Spotlight (with its embedded sub-graph), Cluster Insights, and
   Surprising Connections.
7. Open **Idea Generator** → anchor on a paper or topic and read the
   generated research directions, each grounded in a real graph
   relationship.
8. Stuck on what a control does? Open **User Guide** — last item in
   the nav.

---

## Limitations

- **Simplified citation model.** Citation edges are undirected and
  limited to papers within the loaded dataset. Real citation graphs
  are directed and orders of magnitude larger.
- **Semantic Scholar rate limits.** The free API caps each `/search`
  response at 100 results; the loader paginates and dedupes when
  `limit` exceeds that, but heavy use can still hit rate limits — at
  which point the loader falls back to the bundled JSON gracefully.
- **Threshold sensitivity.** TF-IDF threshold strongly affects graph
  density. Too low → noisy hubs; too high → disconnected components.
  The sidebar slider is there to explore the trade-off; the User
  Guide's *Dataset & threshold cookbook* gives band-by-band guidance.
- **Template-mode idea generation.** Without an API key, idea text is
  functional but formulaic. LLM mode produces richer prose.
- **Author disambiguation.** Names are matched on lowercased strings.
  Real-world disambiguation would need ORCID-style identifiers.
- **No persistent user data.** Sessions are not saved across reruns;
  Streamlit's session-state lifetime is the open browser tab.

---

## Future work

- Directed citation graph with in-degree / out-degree influence flow.
- Embedding-based similarity (sentence transformers) to replace
  TF-IDF for stronger semantic matches.
- Saved explorations / shareable URLs for trajectories and insights.
- PDF / Markdown export for paths and ideas.
- Recommendation surface based on a user's exploration history.

---

## SI 507 requirement mapping

| Requirement | How this project satisfies it |
|---|---|
| **Graph / tree structure** | NetworkX graph with paper nodes and three edge types (citation, TF-IDF similarity, shared authorship). Every interactive feature reads from this graph. |
| **Object-oriented design** | Four core classes — `Paper`, `DataLoader`, `ResearchGraph`, `IdeaEngine` — each with a clear single responsibility. A `ScoredPath` dataclass captures path metadata. UI helpers isolated in `src/app_helpers.py`. |
| **Real-world data** | Live Semantic Scholar API (paginated up to 150 results) plus a bundled 70-paper ML/AI dataset spanning transformers, diffusion, GNNs, RL, vision-language models, RAG. Both flow through the same `DataLoader` normalizer. |
| **Four+ interaction modes** | Seven pages: Project Overview, Search Papers, Graph Explorer, Research Trajectory (paper↔paper and topic↔topic), Insights & Rankings (5 tabs), Idea Generator (3 modes), User Guide. |
| **Interface** | Streamlit dashboard with sidebar nav, fixed right detail panel, in-canvas controls + legend, force-directed pyvis visualization, plus a CLI-friendly test suite. |
| **Testing** | 248 pytest tests, behavior-organised — reading the test names tells you what the system does. Covers all four classes, the visualization layer, the app helpers, and the rubric content. |
| **Ambition & insight** | Bridge papers whose removal would fragment the graph, similarity-only edges that expose missed citations, multi-hop scored trajectories with concept-transfer narratives, modularity-based cluster detection, and graph-grounded idea generation. |

---

## Acknowledgments

- [Semantic Scholar API](https://api.semanticscholar.org/) — paper metadata
- [NetworkX](https://networkx.org/) — graph algorithms
- [pyvis](https://pyvis.readthedocs.io/) / [vis-network](https://visjs.github.io/vis-network/) — interactive force-directed canvas
- [Streamlit](https://streamlit.io/) — interactive UI runtime
- [scikit-learn](https://scikit-learn.org/) — TF-IDF text similarity
- [Inter](https://rsms.me/inter/) — typeface
