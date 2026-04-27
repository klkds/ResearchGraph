"""DataLoader: fetches, caches, normalizes, and loads academic paper data.

Supports the Semantic Scholar API as the primary data source, with a
local JSON fallback dataset bundled in the repository for offline use.

Data flow:
    1. Check local cache for previous results.
    2. If not cached, query Semantic Scholar API.
    3. If API fails (network, rate-limit, timeout), fall back to bundled JSON.
    4. Normalize every raw record into a Paper object.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from src.paper import Paper
from src.utils import clean_text

logger = logging.getLogger(__name__)

# Semantic Scholar API — free tier, no key required for basic queries
_SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = "paperId,title,abstract,year,authors,venue,citationCount,references,url"

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_CACHE_DIR = _DATA_DIR / "cache"
_SAMPLE_PATH = _DATA_DIR / "sample_papers.json"


class DataLoader:
    """Fetches, caches, and normalizes paper data from multiple sources.

    The class owns the full lifecycle of raw data → Paper objects:
      - ``fetch_from_semantic_scholar`` hits the live API with cache + fallback.
      - ``load_local_dataset`` reads the bundled sample JSON.
      - ``normalize_paper_record`` converts any raw dict into a Paper.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Records the provenance of the most recent successful load:
        #   "live"        — fetched fresh from Semantic Scholar
        #   "cached"      — served from the on-disk cache for that query
        #   "bundled"     — loaded from the curated local sample JSON
        #   "fallback:api-error"  — live fetch raised, fell back to bundled
        #   "fallback:empty"      — live fetch returned an empty list
        # ``last_error`` carries the underlying exception text when the
        # most recent fetch fell back, so the UI can show a useful note.
        self.last_source: str = "none"
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_from_semantic_scholar(
        self, query: str, limit: int = 150
    ) -> list[Paper]:
        """Fetch papers from Semantic Scholar by keyword query.

        Returns cached results when available.  Falls back to the bundled
        local dataset if the API is unreachable or rate-limited.

        After the call, inspect ``self.last_source`` ("live" / "cached"
        / "fallback:api-error" / "fallback:empty") and ``self.last_error``
        to learn how the result was produced — the UI uses this to tell
        the user whether they're really looking at live data.
        """
        self.last_error = ""
        cache_path = self._cache_path_for(query)
        cached = self.load_cache(cache_path)
        if cached is not None:
            self.last_source = "cached"
            return cached

        try:
            papers = self._api_search(query, limit)
        except requests.RequestException as exc:
            logger.warning(
                "Semantic Scholar API error: %s — falling back to local dataset",
                exc,
            )
            self.last_source = "fallback:api-error"
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.load_local_dataset(_keep_source=True)

        if not papers:
            # API returned an empty list — treat as a soft failure so
            # the UI can surface it instead of silently showing the
            # bundled set.
            self.last_source = "fallback:empty"
            self.last_error = "Semantic Scholar returned no results for this query."
            return self.load_local_dataset(_keep_source=True)

        self.save_cache(papers, cache_path)
        self.last_source = "live"
        return papers

    def load_local_dataset(
        self,
        path: str | Path | None = None,
        *,
        _keep_source: bool = False,
    ) -> list[Paper]:
        """Load papers from a local JSON file (bundled sample dataset).

        ``_keep_source`` is an internal flag the live-fetch fallback path
        sets so it can preserve its more informative source label
        (``fallback:api-error`` / ``fallback:empty``) instead of being
        overwritten with the generic ``"bundled"``.
        """
        if not _keep_source:
            self.last_source = "bundled"
            self.last_error = ""
        path = Path(path) if path else _SAMPLE_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        return [self.normalize_paper_record(r) for r in raw_records]

    def normalize_paper_record(self, raw: dict[str, Any]) -> Paper:
        """Convert a raw API or JSON record into a :class:`Paper`.

        Handles inconsistencies across data sources:
        - Authors may be dicts (``{"name": "..."}``), plain strings, ``None``,
          or missing entirely.
        - References may be dicts (``{"paperId": "..."}``), plain strings,
          ``None``, or missing entirely.
        - Fields like ``abstract``, ``citationCount``, ``references``,
          ``authors``, ``fieldsOfStudy`` and ``s2FieldsOfStudy`` may all be
          ``None`` rather than absent — Semantic Scholar returns ``null``
          for missing list fields, so ``raw.get(key, [])`` is not enough.
        """
        # Defensive: every list-like field may be present-but-None.
        # ``or []`` collapses both missing and None to an empty list.
        authors_raw = raw.get("authors") or []
        references_raw = raw.get("references") or []
        fields_raw = raw.get("fieldsOfStudy") or []
        s2_fields_raw = raw.get("s2FieldsOfStudy") or []

        authors: list[str] = []
        for a in authors_raw:
            if not a:
                continue
            if isinstance(a, dict):
                name = a.get("name")
                if name:
                    authors.append(name)
            else:
                authors.append(str(a))

        references: list[str] = []
        for ref in references_raw:
            if not ref:
                continue
            if isinstance(ref, dict):
                rid = ref.get("paperId") or ""
                if rid:
                    references.append(rid)
            elif isinstance(ref, str):
                references.append(ref)

        # Topic / field-of-study tags are not stored on Paper (the Paper
        # class derives topics from title+abstract), but we still iterate
        # defensively in case downstream code reads them and to guard
        # against the same None-list crash.
        topics: list[str] = []
        for f in fields_raw:
            if not f:
                continue
            if isinstance(f, dict):
                cat = f.get("category") or f.get("name")
                if cat:
                    topics.append(str(cat))
            elif isinstance(f, str):
                topics.append(f)
        for f in s2_fields_raw:
            if not f:
                continue
            if isinstance(f, dict):
                cat = f.get("category") or f.get("name")
                if cat and cat not in topics:
                    topics.append(str(cat))
            elif isinstance(f, str) and f not in topics:
                topics.append(f)

        return Paper(
            paper_id=raw.get("paperId") or raw.get("paper_id") or "unknown",
            title=clean_text(raw.get("title") or "Untitled"),
            abstract=clean_text(raw.get("abstract") or ""),
            year=raw.get("year"),
            authors=authors,
            venue=raw.get("venue") or "",
            citation_count=raw.get("citationCount") or 0,
            references=references,
            url=raw.get("url") or "",
        )

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def save_cache(self, papers: list[Paper], path: Path) -> None:
        """Serialize a list of Paper objects to a JSON cache file."""
        records = []
        for p in papers:
            records.append({
                "paperId": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
                "year": p.year,
                "authors": [{"name": a} for a in p.authors],
                "venue": p.venue,
                "citationCount": p.citation_count,
                "references": [{"paperId": r} for r in p.references],
                "url": p.url,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def load_cache(self, path: Path) -> list[Paper] | None:
        """Load papers from a cache file, or return None if not cached."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
            return [self.normalize_paper_record(r) for r in raw_records]
        except (json.JSONDecodeError, KeyError):
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_path_for(self, query: str) -> Path:
        """Derive a deterministic cache filename from a query string."""
        safe_name = "".join(c if c.isalnum() else "_" for c in query.lower())
        return self.cache_dir / f"{safe_name}.json"

    # ------------------------------------------------------------------
    # Semantic Scholar request helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ss_headers() -> dict[str, str]:
        """Headers attached to every Semantic Scholar request.

        A meaningful User-Agent matters: Semantic Scholar throttles
        anonymous traffic with the default ``python-requests/...`` UA
        more aggressively than identified clients. If the user has set
        ``SEMANTIC_SCHOLAR_API_KEY`` (free, opt-in), we send it as the
        ``x-api-key`` header to unlock the higher authenticated rate
        limit. Without a key, the public limit still applies.
        """
        h = {
            "User-Agent": "ResearchGraph/1.0 (Streamlit knowledge-graph explorer)",
            "Accept": "application/json",
        }
        key = (os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
               or os.environ.get("S2_API_KEY")
               or "").strip()
        if key:
            h["x-api-key"] = key
        return h

    @staticmethod
    def _retry_after_seconds(resp: requests.Response, attempt: int) -> float:
        """Compute how long to sleep before retrying a 429.

        Honour the server's ``Retry-After`` header when present
        (RFC 7231 — either an int or an HTTP-date), otherwise fall back
        to exponential backoff with jitter so multiple clients
        retrying at once don't all hit the API on the same beat.
        """
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return max(1.0, float(ra))
            except ValueError:
                pass  # HTTP-date form — uncommon for SS; fall through
        # 2s, 4s, 8s … with up to ±0.4s jitter
        return (2.0 ** attempt) + random.uniform(0, 0.4)

    def _ss_get(
        self, url: str, params: dict[str, Any], *, max_retries: int = 4,
    ) -> requests.Response:
        """GET wrapper that retries 429 / 5xx with backoff.

        Raises ``requests.HTTPError`` on a final failure so the
        outer ``fetch_from_semantic_scholar`` catch path still runs.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    url, params=params, headers=self._ss_headers(),
                    timeout=20,
                )
            except requests.RequestException as exc:
                # Network-level error — back off and retry
                last_exc = exc
                if attempt == max_retries - 1:
                    raise
                time.sleep((2.0 ** attempt) + random.uniform(0, 0.4))
                continue

            # 429 = rate-limited. Retry, honoring Retry-After when given.
            if resp.status_code == 429 and attempt < max_retries - 1:
                wait = self._retry_after_seconds(resp, attempt)
                logger.warning(
                    "Semantic Scholar 429 (rate-limited). Retry %d/%d "
                    "after %.1fs.", attempt + 1, max_retries - 1, wait,
                )
                time.sleep(wait)
                continue
            # 5xx = transient server problem. Retry too.
            if 500 <= resp.status_code < 600 and attempt < max_retries - 1:
                wait = (2.0 ** attempt) + random.uniform(0, 0.4)
                logger.warning(
                    "Semantic Scholar %d. Retry %d/%d after %.1fs.",
                    resp.status_code, attempt + 1, max_retries - 1, wait,
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        # Unreachable in practice — the loop either returns or raises —
        # but appease the type checker.
        if last_exc is not None:
            raise last_exc
        raise requests.RequestException("Semantic Scholar request failed")

    def _api_search(self, query: str, limit: int) -> list[Paper]:
        """Execute a search against the Semantic Scholar API.

        The free ``/search`` endpoint caps each response at 100 results,
        so when ``limit`` exceeds that we paginate with ``offset`` and
        merge (deduplicating by paper id). Each request goes through
        ``_ss_get`` which retries 429s with backoff so the rate-limited
        first attempt isn't a hard failure.
        """
        page_size = 100
        papers: list[Paper] = []
        seen: set[str] = set()
        offset = 0
        remaining = max(1, limit)
        while remaining > 0:
            params = {
                "query": query,
                "limit": min(remaining, page_size),
                "offset": offset,
                "fields": _SS_FIELDS,
            }
            resp = self._ss_get(_SS_SEARCH_URL, params)
            data = resp.json().get("data", []) or []
            if not data:
                break
            for raw in data:
                paper = self.normalize_paper_record(raw)
                if paper.paper_id and paper.paper_id not in seen:
                    seen.add(paper.paper_id)
                    papers.append(paper)
            got = len(data)
            offset += got
            remaining -= got
            # Be polite between paginated calls (the free endpoint
            # tolerates ~1 req/sec for unauthenticated clients).
            time.sleep(1.0)
            if got < page_size:
                break  # last page
        return papers
