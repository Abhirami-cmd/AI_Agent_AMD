from __future__ import annotations

from typing import Any

import networkx as nx


class TopologyGraphService:
    def build_dependency_graph(self, incident: dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()
        root_service = incident.get("service", "unknown-service")
        graph.add_node(root_service, type="service", label=root_service)

        for dependency in incident.get("dependencies", []):
            source = dependency.get("source")
            target = dependency.get("dependency")
            tower = dependency.get("tower", "unknown")
            if source and target:
                graph.add_node(source, type="service", label=source)
                graph.add_node(target, type="dependency", label=target, tower=tower)
                graph.add_edge(source, target, tower=tower)

        return graph

    def serialize_graph(self, graph: nx.DiGraph) -> dict[str, Any]:
        return {
            "nodes": [
                {**graph.nodes[node], "id": node}
                for node in graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, **attrs}
                for u, v, attrs in graph.edges(data=True)
            ],
        }

    def build_topology_payload(self, incident: dict[str, Any]) -> dict[str, Any]:
        graph = self.build_dependency_graph(incident)
        return {
            "incident_id": incident.get("incident_id"),
            "service": incident.get("service"),
            "graph": self.serialize_graph(graph),
        }
