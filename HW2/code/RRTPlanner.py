import numpy
from RRTTree import RRTTree
import operator


class RRTPlanner(object):

    def __init__(self, planning_env):
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

    def Plan(self, start_config, goal_config, step_size=99999999):

        # Initialize an empty plan.
        plan = []

        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)

        # TODO (student): Implement your planner here.

        goal_id = 0
        while True:
            x_random = self.planning_env.sample_biased()
            x_nearest_id, x_nearest = self.tree.GetNearestVertex(x_random)
            x_new = self.extend(x_random, x_nearest, step_size)
            if self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.tree.AddVertex(x_new)
                self.tree.AddEdge(x_nearest_id, x_new_id)  # Edges indicate the parent of each node
                if x_new == goal_config:
                    goal_id = x_new_id  # Currently this will always be the last node added
                    break

        id_to_add = goal_id
        plan.append(self.tree.vertices[id_to_add])
        while id_to_add != self.tree.GetRootID():
            id_to_add = self.tree.edges[id_to_add]
            plan.append(self.tree.vertices[id_to_add])

        plan.reverse()
        return numpy.array(plan)

    def extend(self, x_rand, x_near, eta):
        if self.planning_env.compute_distance(x_rand, x_near) < eta:
            return x_rand
        else:
            dir = numpy.array(tuple(map(operator.sub, x_rand, x_near)))
            norm = numpy.linalg.norm(dir)
            if norm == 0:
                return x_rand
            else:
                dir = (eta / norm) * dir  # Scaling direction to be of size at most eta
                dir = dir.astype(int)  # Removing fractional part to get pixel coordinates
                return [x_near[0] + dir[0], x_near[1] + dir[1]]

    def VisualizeTree(self):
        self.tree.VisualizeTree()

