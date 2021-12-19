import numpy
from RRTTree import RRTTree

class RRTPlanner(object):

    def __init__(self, planning_env):
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

    def Plan(self, start_config, goal_config, step_size=0.001):
        
        # Initialize an empty plan.
        plan = []

        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)

        # TODO (student): Implement your planner here.

        while True:
            # TOOD: think of a stopping condition
            x_random = self.planning_env.sample_biased()
            x_nearest_id, x_nearest = self.planning_env.GetNearestVertex(x_random)
            x_new = self.extend(x_random, x_nearest, step_size)
            if self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.tree.AddVertex(x_new)
                self.tree.AddEdge(x_new_id, x_nearest_id)  # Edges indicate the parent of each node
            if x_new == goal_config:
                # TODO: fix end_condition
                return True

        plan.append(start_config)
        plan.append(goal_config)

        return numpy.array(plan)

    def extend(self, x_rand, x_near, eta):
        pass

