import numpy
from RRTTree import RRTTree
import operator


class RRTPlanner(object):

    def __init__(self, planning_env, step_size=50):
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)
        self.step_size = step_size

    def Plan(self, start_config, goal_config):
        # Start with adding the start configuration to the tree.
        self.add_config(start_config)

        # TODO (student): Implement your planner here.

        goal_vertex_id = self.build_tree(goal_config=goal_config, step_size=self.step_size)
        return self.plan_to_vertex(goal_vertex_id)

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

    def build_tree(self, goal_config, step_size):
        while True:
            x_random = self.planning_env.sample_biased()
            x_nearest_id, x_nearest = self.tree.GetNearestVertex(x_random)
            x_new = self.extend(x_random, x_nearest, step_size)
            if self.planning_env.state_validity_checker(x_new) and \
                    self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.add_config(x_new, x_nearest_id)
                if x_new == goal_config:
                    return x_new_id  # Technically this will always be len(self.tree.vertices) - 1

    def add_config(self, x_add, parent_id=None):
        x_add_id = self.tree.AddVertex(x_add)
        if parent_id is not None:
            self.tree.AddEdge(parent_id, x_add_id)  # Edges indicate the parent of each node
        return x_add_id

    def plan_to_vertex(self, vertex_id):
        plan = []
        id_to_add = vertex_id

        plan.append(self.tree.vertices[id_to_add])
        while id_to_add != self.tree.GetRootID():
            id_to_add = self.tree.edges[id_to_add]
            plan.append(self.tree.vertices[id_to_add])
        plan.reverse()
        return numpy.array(plan)

    def VisualizeTree(self):
        self.tree.VisualizeTree()
