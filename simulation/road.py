from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import hypot
import heapq
from typing import Dict, Iterable, List, Sequence

from config.constants import INTERSECTION_CONNECTIONS, INTERSECTION_LAYOUT


@dataclass(frozen=True)
class Edge:
    start: int
    end: int
    length_m: float
    lanes: int = 1


class RoadNetwork:
    """In-memory graph model for intersections and directed road segments."""

    def __init__(self, positions: Dict[int, tuple[float, float]], undirected_edges: Sequence[tuple[int, int, int]]):
        self.positions = positions
        self.undirected_edges = list(undirected_edges)
        self.adjacency: Dict[int, list[int]] = defaultdict(list)
        self.directed_edges: Dict[tuple[int, int], Edge] = {}
        self._build_graph()

    @classmethod
    def from_defaults(cls) -> "RoadNetwork":
        return cls(INTERSECTION_LAYOUT, INTERSECTION_CONNECTIONS)

    def _build_graph(self) -> None:
        for a, b, lanes in self.undirected_edges:
            self.adjacency[a].append(b)
            self.adjacency[b].append(a)
            length = self.distance(a, b)
            self.directed_edges[(a, b)] = Edge(start=a, end=b, length_m=length, lanes=lanes)
            self.directed_edges[(b, a)] = Edge(start=b, end=a, length_m=length, lanes=lanes)

    def distance(self, a: int, b: int) -> float:
        ax, ay = self.positions[a]
        bx, by = self.positions[b]
        return hypot(bx - ax, by - ay)

    def neighbors(self, node_id: int) -> List[int]:
        return self.adjacency[node_id]

    def edge(self, start: int, end: int) -> Edge:
        return self.directed_edges[(start, end)]

    def has_edge(self, start: int, end: int) -> bool:
        return (start, end) in self.directed_edges

    def all_nodes(self) -> Iterable[int]:
        return self.positions.keys()

    def shortest_path(self, origin: int, destination: int) -> list[int]:
        if origin == destination:
            return [origin]

        # Prefer shorter local links so simulated traffic follows intersection-by-intersection flow.
        distances: Dict[int, float] = {origin: 0.0}
        parent: Dict[int, int | None] = {origin: None}
        heap: list[tuple[float, int]] = [(0.0, origin)]

        while heap:
            current_cost, current = heapq.heappop(heap)
            if current_cost > distances.get(current, float("inf")):
                continue
            if current == destination:
                break

            for nxt in self.adjacency[current]:
                edge_len = self.edge(current, nxt).length_m
                # Slightly super-linear edge cost discourages skipping intermediate intersections.
                transition_cost = edge_len ** 1.05
                new_cost = current_cost + transition_cost
                if new_cost < distances.get(nxt, float("inf")):
                    distances[nxt] = new_cost
                    parent[nxt] = current
                    heapq.heappush(heap, (new_cost, nxt))

        if destination not in parent:
            return [origin]

        path = [destination]
        node = destination
        while parent[node] is not None:
            node = parent[node]
            path.append(node)
        path.reverse()
        return path

    def incoming_neighbors(self, node_id: int) -> list[int]:
        return list(self.adjacency[node_id])
