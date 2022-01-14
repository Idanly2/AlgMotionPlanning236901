import numpy as np
from RRTTree import RRTTree
from RRTMotionPlanner import RRTMotionPlanner
import time
import operator


class RRTInspectionPlanner(RRTMotionPlanner):

    def __init__(self, planning_env, ext_mode, goal_prob, coverage):
        super(RRTInspectionPlanner, self).__init__(planning_env, ext_mode, goal_prob)
        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env, task="ip")

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        self.num_converge_success = 0
        self.num_converge_fail = 0

    def get_goal_config(self):
        random_inspect_id = np.random.randint(len(self.planning_env.inspection_points))
        rand_inspect_point = self.planning_env.inspection_points[random_inspect_id]
        closest_id, closest_config = self.tree.get_nearest_config_ee_point(rand_inspect_point)
        converged_config = self.planning_env.robot.converge_to_view(closest_config, rand_inspect_point)
        converge_success = rand_inspect_point in self.planning_env.get_inspected_points(converged_config)
        if converge_success:
            self.num_converge_success += 1
        else:
            self.num_converge_fail += 1
        print("converge ratio: ", str(self.num_converge_success / (self.num_converge_success + self.num_converge_fail)))
        return converged_config

    def add_config(self, x_add, ws_pose, parent_id=None, cost=0):
        inspected_points = self.planning_env.get_inspected_points(x_add)
        x_add_id = self.tree.add_vertex(x_add, inspected_points=inspected_points)
        self.tree.vertices[x_add_id].cost = cost
        self.tree.vertices[x_add_id].ws_pose = ws_pose
        if parent_id is not None:
            self.tree.add_edge(parent_id, x_add_id)  # Edges indicate the parent of each node
            self.tree.vertices[x_add_id].inspected_points = \
                self.planning_env.compute_union_of_points(inspected_points,
                                                          self.tree.vertices[parent_id].inspected_points)
        else:
            self.tree.vertices[x_add_id].inspected_points = inspected_points
        return x_add_id

    def build_coverage_tree(self):
        while True:
            x_random = self.sample_biased()
            x_nearest_id, x_nearest = self.tree.get_nearest_config_approx(x_random)
            x_new = self.extend(x_nearest, x_random)
            if self.planning_env.config_validity_checker(x_new) and \
                    self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.add_config(x_new, ws_pose=self.planning_env.robot.compute_forward_kinematics(x_new),
                                           parent_id=x_nearest_id)
                self.num_added += 1
                if self.tree.max_coverage > self.coverage:
                    return self.tree.max_coverage_id
            else:
                self.num_discarded += 1
            print("ratio added: ", self.num_added / (self.num_added + self.num_discarded))
            print("max_coverage: ", self.tree.max_coverage)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()
        # Start with adding the start configuration to the tree.
        start_config = self.planning_env.start
        self.add_config(start_config, self.planning_env.robot.compute_forward_kinematics(start_config))

        # TODO: Task 2.3
        max_coverage_id = self.build_coverage_tree()
        plan_list_of_trees = self.plan_to_vertex(max_coverage_id)
        end_time = time.time()
        plan = np.vstack([vertex.config for vertex in plan_list_of_trees])
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total planning time: {:.2f}'.format(end_time - start_time))

        return plan
