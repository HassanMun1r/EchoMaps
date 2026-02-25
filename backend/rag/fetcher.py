"""
Wikipedia REST API + Wikidata SPARQL fetcher.

Both sources are free and require no API key.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from backend.config import (
    GEOSEARCH_LIMIT,
    GEOSEARCH_RADIUS,
    WIKIDATA_SPARQL,
    WIKIPEDIA_API,
)

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "EchoMaps/1.0 (geospatial RAG; educational)"})


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def get_wikipedia_summary(lat: float, lon: float) -> list[dict[str, Any]]:
    """Return up to GEOSEARCH_LIMIT Wikipedia articles near (lat, lon).

    Each item: {title, extract, url, lat, lon}
    Returns empty list on any error.
    """
    try:
        # Step 1: geosearch — find nearby page IDs
        geo_params = {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": GEOSEARCH_RADIUS,
            "gslimit": GEOSEARCH_LIMIT,
            "format": "json",
        }
        geo_resp = _SESSION.get(WIKIPEDIA_API, params=geo_params, timeout=10)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        hits = geo_data.get("query", {}).get("geosearch", [])
        if not hits:
            log.info("Wikipedia geosearch: no results near (%.4f, %.4f)", lat, lon)
            return []

        page_ids = [str(h["pageid"]) for h in hits]

        # Step 2: fetch extracts for all page IDs in one request
        extract_params = {
            "action": "query",
            "pageids": "|".join(page_ids),
            "prop": "extracts|coordinates",
            "exintro": False,          # full article, not just intro
            "explaintext": True,       # plain text, no HTML
            "exsectionformat": "plain",
            "format": "json",
        }
        ext_resp = _SESSION.get(WIKIPEDIA_API, params=extract_params, timeout=15)
        ext_resp.raise_for_status()
        pages = ext_resp.json().get("query", {}).get("pages", {})

        results: list[dict[str, Any]] = []
        for pid, page in pages.items():
            extract = page.get("extract", "").strip()
            if not extract:
                continue
            coords = page.get("coordinates", [{}])[0]
            results.append(
                {
                    "title": page.get("title", ""),
                    "extract": extract,
                    "url": f"https://en.wikipedia.org/wiki/{page.get('title','').replace(' ','_')}",
                    "lat": coords.get("lat", lat),
                    "lon": coords.get("lon", lon),
                }
            )

        log.info("Wikipedia: fetched %d articles near (%.4f, %.4f)", len(results), lat, lon)
        return results

    except Exception as exc:  # noqa: BLE001
        log.warning("Wikipedia fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------

_WIKIDATA_QUERY = """\
SELECT ?item ?itemLabel ?itemDescription ?typeLabel ?lat ?lon WHERE {{
  SERVICE wikibase:around {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
    bd:serviceParam wikibase:radius "{radius}" .
    bd:serviceParam wikibase:distance ?distance .
  }}
  BIND(geof:latitude(?coord)  AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
  OPTIONAL {{ ?item wdt:P31 ?type }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}
ORDER BY ?distance
LIMIT 20
"""


def get_wikidata_entities(lat: float, lon: float) -> list[dict[str, Any]]:
    """Return Wikidata entities near (lat, lon).

    Each item: {label, description, type, lat, lon}
    Returns empty list on any error.
    """
    radius_km = GEOSEARCH_RADIUS / 1000
    sparql = _WIKIDATA_QUERY.format(lat=lat, lon=lon, radius=radius_km)

    try:
        resp = _SESSION.get(
            WIKIDATA_SPARQL,
            params={"query": sparql, "format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])

        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for b in bindings:
            label = b.get("itemLabel", {}).get("value", "")
            if not label or label.startswith("Q"):  # skip unlabelled items
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "label": label,
                    "description": b.get("itemDescription", {}).get("value", ""),
                    "type": b.get("typeLabel", {}).get("value", ""),
                    "lat": float(b.get("lat", {}).get("value", lat)),
                    "lon": float(b.get("lon", {}).get("value", lon)),
                }
            )

        log.info("Wikidata: fetched %d entities near (%.4f, %.4f)", len(results), lat, lon)
        return results

    except Exception as exc:  # noqa: BLE001
        log.warning("Wikidata fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Budapest, Hungary
    TEST_LAT, TEST_LON = 47.4979, 19.0402

    print("\n=== Wikipedia ===")
    wiki = get_wikipedia_summary(TEST_LAT, TEST_LON)
    for art in wiki:
        print(f"  [{art['title']}] {art['extract'][:120]}...")

    print("\n=== Wikidata ===")
    wd = get_wikidata_entities(TEST_LAT, TEST_LON)
    for ent in wd[:8]:
        print(f"  {ent['label']} ({ent['type']}): {ent['description'][:80]}")
