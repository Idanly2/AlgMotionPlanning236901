import math

import numpy
from RRTTree import RRTTree
from RRTPlanner import RRTPlanner


class RRTStarPlanner(RRTPlanner):

    def __init__(self, planning_env, step_size=50, k_nearest_neighbors='log'):
        super().__init__(planning_env, step_size=step_size)
        self.k = k_nearest_neighbors

    def add_config(self, x_add, parent_id=None):
        x_add_id = self.tree.AddVertex(x_add)
        if parent_id is not None:
            self.tree.AddEdge(parent_id, x_add_id)  # Edges indicate the parent of each node
            edge_cost = self.planning_env.compute_distance(self.tree.vertices[parent_id], x_add)
            # Cost of added node is the cost of its parent, plus the distance from the parent to it.
            self.tree.SetVertexCost(x_add_id, self.tree.GetVertexCost(parent_id) + edge_cost)
            nearest_neighbors_ids, nearest_neighbors = self.tree.GetKNN(x_add, self.num_nearest_neighbors())
            # Attempt to rewire from all nearest neighbors to added node
            for neighbor_id in nearest_neighbors_ids:
                self.rewire_rrt(neighbor_id, x_add_id)
            # Attempt to rewire from added node to all nearest neighbors
            for neighbor_id in nearest_neighbors_ids:
                self.rewire_rrt(x_add_id, neighbor_id)
        else:
            self.tree.SetVertexCost(x_add_id, 0)

        return x_add_id

    def rewire_rrt(self, x_potential_parent_id, x_child_id):
        x_child = self.tree.vertices[x_child_id]
        x_potential_parent = self.tree.vertices[x_potential_parent_id]

        if self.planning_env.edge_validity_checker(x_potential_parent, x_child):
            c = self.planning_env.compute_distance(x_potential_parent, x_child)
            potential_cost = self.tree.GetVertexCost(x_potential_parent_id) + c
            if potential_cost < self.tree.GetVertexCost(x_child_id):
                self.tree.AddEdge(x_potential_parent_id, x_child_id)
                self.tree.SetVertexCost(x_child_id, potential_cost)

    def num_nearest_neighbors(self):
        n = len(self.tree.vertices) - 1

        if self.k == 'log':
            k = int(math.log(n + 1))
        else:
            k = self.k

        return min(k, n)
