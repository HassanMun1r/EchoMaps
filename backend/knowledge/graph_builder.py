"""
Knowledge-graph builder using NetworkX DiGraph.

Builds a graph centred on the queried location, with entity nodes
radiating outward. Entities of the same type are also linked to a
shared type-cluster node so the frontend can render coherent clusters.

In-memory cache keyed by location_id avoids re-building on repeated calls.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

log = logging.getLogger(__name__)

# Simple in-memory cache: location_id -> serialised graph dict
_GRAPH_CACHE: dict[str, dict[str, Any]] = {}


def build_knowledge_graph(
    entities: list[dict[str, Any]],
    location_id: str,
    location_name: str,
) -> dict[str, Any]:
    """Build and serialise a knowledge graph for a location.

    Graph structure:
      - One central node for the location itself (type="LOCATION")
      - One cluster node per entity type that appears (type="CLUSTER")
      - One node per unique entity (type = spaCy label)
      - Edges:
          location  -[HAS_ENTITY]->  entity
          entity    -[IS_TYPE]->     cluster-node
          (no duplicate edges)

    Args:
        entities:      Output of extract_entities().
        location_id:   Stable key used for caching.
        location_name: Human-readable name for the central node.

    Returns:
        JSON-serialisable dict:
          {
            nodes: [{id, label, type}],
            edges: [{source, target, relation}],
          }
    """
    if location_id in _GRAPH_CACHE:
        log.debug("Graph cache hit for %s", location_id)
        return _GRAPH_CACHE[location_id]

    G: nx.DiGraph = nx.DiGraph()

    # ── Central location node ─────────────────────────────────────────────────
    loc_id = f"loc::{location_id}"
    G.add_node(loc_id, label=location_name, type="LOCATION")

    # ── Entity and cluster nodes ──────────────────────────────────────────────
    for ent in entities:
        surface = ent["text"]
        label   = ent["label"]   # spaCy label e.g. "PERSON"
        ent_id  = f"{label}::{surface.lower()}"
        cluster_id = f"cluster::{label}"

        # Cluster node (one per label type)
        if not G.has_node(cluster_id):
            G.add_node(cluster_id, label=label, type="CLUSTER")

        # Entity node
        if not G.has_node(ent_id):
            G.add_node(ent_id, label=surface, type=label)

        # location -> entity
        if not G.has_edge(loc_id, ent_id):
            G.add_edge(loc_id, ent_id, relation="HAS_ENTITY")

        # entity -> cluster (for type-grouping on the frontend)
        if not G.has_edge(ent_id, cluster_id):
            G.add_edge(ent_id, cluster_id, relation="IS_TYPE")

    # ── Serialise ─────────────────────────────────────────────────────────────
    nodes = [
        {"id": node_id, "label": data.get("label", node_id), "type": data.get("type", "OTHER")}
        for node_id, data in G.nodes(data=True)
    ]
    edges = [
        {"source": src, "target": tgt, "relation": data.get("relation", "")}
        for src, tgt, data in G.edges(data=True)
    ]

    graph_dict: dict[str, Any] = {"nodes": nodes, "edges": edges}

    _GRAPH_CACHE[location_id] = graph_dict
    log.info("Built knowledge graph for %s: %d nodes, %d edges",
             location_name, len(nodes), len(edges))

    return graph_dict


def invalidate_cache(location_id: str) -> None:
    """Remove a location's graph from the in-memory cache."""
    _GRAPH_CACHE.pop(location_id, None)
