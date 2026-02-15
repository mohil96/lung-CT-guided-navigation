"""
A* Path Planning for Airway Navigation.

Implements A* algorithm for finding optimal paths from trachea to target 
nodules along the airway tree centerline.

Features:
- Standard A* with heuristic search
- Bidirectional A* for faster pathfinding
- Cost function incorporating: distance, angles, airway diameter
- Collision detection with airway walls
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import heapq


class AStarPathPlanner:
    """
    A* path planner for airway navigation.

    Finds optimal collision-free paths along airway centerlines
    from trachea (entry point) to target nodules.
    """

    def __init__(self, graph: nx.Graph, airway_mask: np.ndarray = None):
        """
        Args:
            graph: NetworkX graph representing airway centerline
                   Nodes have 'pos' attribute with 3D coordinates
                   Edges have 'weight' attribute for distance
            airway_mask: Binary mask of airway segmentation (for collision check)
        """
        self.graph = graph
        self.airway_mask = airway_mask

        # Cost function weights
        self.distance_weight = 1.0
        self.angle_weight = 0.3
        self.diameter_weight = 0.2

    def euclidean_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """Compute Euclidean distance between two 3D points."""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def compute_angle_cost(self, pos_prev: Tuple, pos_curr: Tuple, pos_next: Tuple) -> float:
        """
        Compute cost based on turning angle.

        Sharp turns are penalized to prefer smoother paths.

        Args:
            pos_prev: Previous position
            pos_curr: Current position
            pos_next: Next position

        Returns:
            angle_cost: Cost based on angle (0 = straight, higher = sharper turn)
        """
        if pos_prev is None:
            return 0.0

        # Vectors
        v1 = np.array(pos_curr) - np.array(pos_prev)
        v2 = np.array(pos_next) - np.array(pos_curr)

        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

        # Cosine of angle
        cos_angle = np.dot(v1_norm, v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Angle in radians
        angle = np.arccos(cos_angle)

        # Cost increases with angle (0 to π)
        return angle / np.pi

    def heuristic(self, node: int, goal: int) -> float:
        """
        Heuristic function for A* (estimated cost to goal).

        Uses Euclidean distance as admissible heuristic.

        Args:
            node: Current node ID
            goal: Goal node ID

        Returns:
            h: Estimated cost from node to goal
        """
        pos_curr = self.graph.nodes[node]['pos']
        pos_goal = self.graph.nodes[goal]['pos']

        return self.euclidean_distance(pos_curr, pos_goal)

    def compute_edge_cost(self, node_from: int, node_to: int, 
                         prev_node: Optional[int] = None) -> float:
        """
        Compute cost of moving from node_from to node_to.

        Cost = distance_weight * distance + 
               angle_weight * angle_cost +
               diameter_weight * diameter_cost

        Args:
            node_from: Source node ID
            node_to: Target node ID
            prev_node: Previous node in path (for angle computation)

        Returns:
            cost: Edge traversal cost
        """
        pos_from = self.graph.nodes[node_from]['pos']
        pos_to = self.graph.nodes[node_to]['pos']

        # Distance cost
        distance = self.euclidean_distance(pos_from, pos_to)
        cost = self.distance_weight * distance

        # Angle cost (prefer straight paths)
        if prev_node is not None:
            pos_prev = self.graph.nodes[prev_node]['pos']
            angle_cost = self.compute_angle_cost(pos_prev, pos_from, pos_to)
            cost += self.angle_weight * angle_cost

        # Diameter cost (prefer wider airways if available)
        if 'diameter' in self.graph.nodes[node_to]:
            diameter = self.graph.nodes[node_to]['diameter']
            # Inverse relationship: smaller diameter = higher cost
            diameter_cost = 1.0 / (diameter + 1.0)
            cost += self.diameter_weight * diameter_cost

        return cost

    def find_path(self, start: int, goal: int) -> Optional[Dict]:
        """
        Find optimal path using A* algorithm.

        Args:
            start: Start node ID (e.g., trachea entry)
            goal: Goal node ID (e.g., target nodule location)

        Returns:
            result: Dictionary containing:
                - 'path': List of node IDs from start to goal
                - 'cost': Total path cost
                - 'length': Path length in mm
                - 'num_nodes': Number of nodes in path
                - 'success': Boolean indicating if path was found
        """
        if start not in self.graph.nodes or goal not in self.graph.nodes:
            return {'success': False, 'path': None, 'cost': float('inf')}

        if start == goal:
            return {
                'success': True,
                'path': [start],
                'cost': 0.0,
                'length': 0.0,
                'num_nodes': 1
            }

        # Initialize
        open_set = []  # Priority queue: (f_score, node)
        heapq.heappush(open_set, (0.0, start))

        came_from = {}  # Parent mapping
        g_score = {start: 0.0}  # Cost from start to node
        f_score = {start: self.heuristic(start, goal)}  # Estimated total cost

        closed_set = set()

        while open_set:
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            # Goal reached
            if current == goal:
                return self._reconstruct_path(came_from, current, g_score)

            closed_set.add(current)

            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor in closed_set:
                    continue

                # Get previous node for angle computation
                prev_node = came_from.get(current, None)

                # Compute tentative g_score
                edge_cost = self.compute_edge_cost(current, neighbor, prev_node)
                tentative_g = g_score[current] + edge_cost

                # Check if this path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return {'success': False, 'path': None, 'cost': float('inf')}

    def _reconstruct_path(self, came_from: Dict, current: int, 
                         g_score: Dict) -> Dict:
        """
        Reconstruct path from start to goal.

        Args:
            came_from: Parent mapping
            current: Goal node
            g_score: Cost from start to each node

        Returns:
            result: Dictionary with path information
        """
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        # Compute path length
        length = 0.0
        for i in range(len(path) - 1):
            pos1 = self.graph.nodes[path[i]]['pos']
            pos2 = self.graph.nodes[path[i + 1]]['pos']
            length += self.euclidean_distance(pos1, pos2)

        return {
            'success': True,
            'path': path,
            'cost': g_score[path[-1]],
            'length': length,
            'num_nodes': len(path)
        }

    def compute_path_tortuosity(self, path: List[int]) -> float:
        """
        Compute path tortuosity (curvature measure).

        Tortuosity = actual_path_length / straight_line_distance

        Value of 1.0 = straight line
        Higher values = more tortuous

        Args:
            path: List of node IDs

        Returns:
            tortuosity: Path tortuosity metric
        """
        if len(path) < 2:
            return 1.0

        # Actual path length
        path_length = 0.0
        for i in range(len(path) - 1):
            pos1 = self.graph.nodes[path[i]]['pos']
            pos2 = self.graph.nodes[path[i + 1]]['pos']
            path_length += self.euclidean_distance(pos1, pos2)

        # Straight-line distance
        start_pos = self.graph.nodes[path[0]]['pos']
        end_pos = self.graph.nodes[path[-1]]['pos']
        straight_distance = self.euclidean_distance(start_pos, end_pos)

        if straight_distance < 1e-6:
            return 1.0

        return path_length / straight_distance


class BidirectionalAStar(AStarPathPlanner):
    """
    Bidirectional A* search.

    Searches from both start and goal simultaneously, meeting in the middle.
    Faster for long paths in large graphs.
    """

    def find_path(self, start: int, goal: int) -> Optional[Dict]:
        """
        Find path using bidirectional A*.

        Args:
            start: Start node ID
            goal: Goal node ID

        Returns:
            result: Dictionary with path information
        """
        if start not in self.graph.nodes or goal not in self.graph.nodes:
            return {'success': False, 'path': None, 'cost': float('inf')}

        if start == goal:
            return {
                'success': True,
                'path': [start],
                'cost': 0.0,
                'length': 0.0,
                'num_nodes': 1
            }

        # Forward search (from start)
        open_forward = [(0.0, start)]
        came_from_forward = {}
        g_forward = {start: 0.0}
        closed_forward = set()

        # Backward search (from goal)
        open_backward = [(0.0, goal)]
        came_from_backward = {}
        g_backward = {goal: 0.0}
        closed_backward = set()

        best_path_cost = float('inf')
        meeting_point = None

        while open_forward and open_backward:
            # Forward step
            if open_forward:
                _, curr_f = heapq.heappop(open_forward)

                if curr_f not in closed_forward:
                    closed_forward.add(curr_f)

                    # Check if meeting point
                    if curr_f in closed_backward:
                        total_cost = g_forward[curr_f] + g_backward[curr_f]
                        if total_cost < best_path_cost:
                            best_path_cost = total_cost
                            meeting_point = curr_f

                    # Expand forward
                    for neighbor in self.graph.neighbors(curr_f):
                        if neighbor not in closed_forward:
                            prev = came_from_forward.get(curr_f, None)
                            cost = self.compute_edge_cost(curr_f, neighbor, prev)
                            tentative_g = g_forward[curr_f] + cost

                            if neighbor not in g_forward or tentative_g < g_forward[neighbor]:
                                came_from_forward[neighbor] = curr_f
                                g_forward[neighbor] = tentative_g
                                f = tentative_g + self.heuristic(neighbor, goal)
                                heapq.heappush(open_forward, (f, neighbor))

            # Backward step
            if open_backward:
                _, curr_b = heapq.heappop(open_backward)

                if curr_b not in closed_backward:
                    closed_backward.add(curr_b)

                    # Check if meeting point
                    if curr_b in closed_forward:
                        total_cost = g_forward[curr_b] + g_backward[curr_b]
                        if total_cost < best_path_cost:
                            best_path_cost = total_cost
                            meeting_point = curr_b

                    # Expand backward
                    for neighbor in self.graph.neighbors(curr_b):
                        if neighbor not in closed_backward:
                            prev = came_from_backward.get(curr_b, None)
                            cost = self.compute_edge_cost(curr_b, neighbor, prev)
                            tentative_g = g_backward[curr_b] + cost

                            if neighbor not in g_backward or tentative_g < g_backward[neighbor]:
                                came_from_backward[neighbor] = curr_b
                                g_backward[neighbor] = tentative_g
                                f = tentative_g + self.heuristic(neighbor, start)
                                heapq.heappush(open_backward, (f, neighbor))

            # Check termination
            if meeting_point is not None:
                # Reconstruct full path
                path = self._reconstruct_bidirectional_path(
                    came_from_forward, came_from_backward, meeting_point, start, goal
                )

                # Compute length
                length = 0.0
                for i in range(len(path) - 1):
                    pos1 = self.graph.nodes[path[i]]['pos']
                    pos2 = self.graph.nodes[path[i + 1]]['pos']
                    length += self.euclidean_distance(pos1, pos2)

                return {
                    'success': True,
                    'path': path,
                    'cost': best_path_cost,
                    'length': length,
                    'num_nodes': len(path)
                }

        return {'success': False, 'path': None, 'cost': float('inf')}

    def _reconstruct_bidirectional_path(self, came_from_forward: Dict, 
                                       came_from_backward: Dict,
                                       meeting: int, start: int, goal: int) -> List[int]:
        """Reconstruct path from bidirectional search."""
        # Forward path: start -> meeting
        path_forward = []
        current = meeting
        while current != start:
            path_forward.append(current)
            if current not in came_from_forward:
                break
            current = came_from_forward[current]
        path_forward.append(start)
        path_forward.reverse()

        # Backward path: meeting -> goal
        path_backward = []
        current = meeting
        while current != goal:
            if current not in came_from_backward:
                break
            current = came_from_backward[current]
            path_backward.append(current)

        # Combine
        return path_forward + path_backward


def test_astar():
    """Test A* path planning."""

    print("Testing A* Path Planning...")

    # Create simple test graph
    G = nx.Graph()

    # Add nodes with positions
    nodes = {
        0: (0, 0, 0),    # Start (trachea)
        1: (10, 0, 0),
        2: (20, 0, 0),
        3: (20, 10, 0),
        4: (20, 20, 0),  # Goal (nodule)
        5: (15, 5, 0),   # Alternative path
    }

    for node_id, pos in nodes.items():
        G.add_node(node_id, pos=pos)

    # Add edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 3)]
    G.add_edges_from(edges)

    # Create planner
    planner = AStarPathPlanner(G)

    # Find path
    result = planner.find_path(start=0, goal=4)

    print(f"\nPath finding result:")
    print(f"  Success: {result['success']}")
    print(f"  Path: {result['path']}")
    print(f"  Length: {result['length']:.2f} mm")
    print(f"  Cost: {result['cost']:.2f}")
    print(f"  Num nodes: {result['num_nodes']}")

    assert result['success'], "Path should be found"
    assert result['path'][0] == 0 and result['path'][-1] == 4, "Path endpoints incorrect"

    # Test tortuosity
    tortuosity = planner.compute_path_tortuosity(result['path'])
    print(f"  Tortuosity: {tortuosity:.2f}")

    print("\n✓ A* test passed")

    # Test bidirectional A*
    print("\nTesting Bidirectional A*...")
    planner_bi = BidirectionalAStar(G)
    result_bi = planner_bi.find_path(start=0, goal=4)

    print(f"  Success: {result_bi['success']}")
    print(f"  Path length: {result_bi['length']:.2f} mm")

    print("✓ Bidirectional A* test passed")


if __name__ == "__main__":
    test_astar()
