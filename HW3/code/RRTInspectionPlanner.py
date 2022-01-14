import numpy as np
from RRTTree import RRTTree
from RRTMotionPlanner import RRTMotionPlanner
import time
# import networkx as nx


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
        curr_best_ip = self.tree.vertices[self.tree.max_coverage_id].inspected_points
        inspect_points_left = self.planning_env.compute_diff_of_points(self.planning_env.inspection_points,
                                                                       curr_best_ip)
        random_inspect_id = np.random.randint(len(inspect_points_left))
        rand_inspect_point = inspect_points_left[random_inspect_id]
        closest_id, closest_config = self.tree.get_nearest_config_ee_point(rand_inspect_point)
        converged_config = self.planning_env.robot.converge_to_view(closest_config, rand_inspect_point)
        # converge_success = rand_inspect_point in self.planning_env.get_inspected_points(converged_config)
        # if converge_success:
        #     self.num_converge_success += 1
        # else:
        #     self.num_converge_fail += 1
        # print("converge ratio: ", str(self.num_converge_success / (self.num_converge_success + self.num_converge_fail)))
        return converged_config

    def add_config(self, x_add, ws_pose, parent_id=None, rewire=True):
        inspected_points_at_config = self.planning_env.get_inspected_points(x_add)
        if parent_id is not None:
            inspected_points = self.planning_env.compute_union_of_points(inspected_points_at_config,
                                                                         self.tree.vertices[parent_id].inspected_points)
        else:
            inspected_points = inspected_points_at_config

        x_add_id = self.tree.add_vertex(x_add, inspected_points=inspected_points)
        self.tree.vertices[x_add_id].ws_pose = ws_pose

        if parent_id is not None:
            self.tree.add_edge(parent_id, x_add_id)  # Edges indicate the parent of each node
            if rewire:
                edge_cost = self.planning_env.robot.compute_distance(self.tree.vertices[parent_id].config, x_add)
                # Cost of added node is the cost of its parent, plus the distance from the parent to it.
                self.tree.add_edge(parent_id, x_add_id, edge_cost)  # Edges indicate the parent of each node
                nearest_neighbors_ids = self.tree.get_knn(x_add, self.num_nearest_neighbors())
                # Attempt to rewire from all nearest neighbors to added node
                for neighbor_id in nearest_neighbors_ids:
                    self.rewire_rrt(neighbor_id, x_add_id, inspected_points_at_config)
                # Attempt to rewire from added node to all nearest neighbors
                for neighbor_id in nearest_neighbors_ids:
                    self.rewire_rrt(x_add_id, neighbor_id)

        return x_add_id

    def rewire_rrt(self, x_potential_parent_id, x_child_id, inspected_points_at_child_config=None):
        x_child = self.tree.vertices[x_child_id]
        x_potential_parent = self.tree.vertices[x_potential_parent_id]
        if self.planning_env.edge_validity_checker(x_potential_parent.config, x_child.config):
            edge_cost = self.planning_env.robot.compute_distance(x_potential_parent.config, x_child.config)
            potential_cost = x_potential_parent.cost + edge_cost
            if inspected_points_at_child_config is None:
                inspected_points_at_child_config = self.planning_env.get_inspected_points(x_child.config)
            current_coverage = self.planning_env.compute_coverage(x_child.inspected_points)
            potential_points_union = self.planning_env.compute_union_of_points(x_potential_parent.inspected_points,
                                                                               inspected_points_at_child_config)
            potential_coverage = self.planning_env.compute_coverage(potential_points_union)
            if potential_cost < x_child.cost and potential_coverage >= current_coverage:
                self.tree.add_edge(x_potential_parent_id, x_child_id, edge_cost)
                self.tree.set_inspected_points(x_child_id, potential_points_union)

            # Debugging why loops are formed
            # resulted_graph = nx.DiGraph(self.tree.edges.items())
            # cycles = list(nx.simple_cycles(resulted_graph))
            # if cycles:
            #     print("help")

    def num_nearest_neighbors(self):
        n = len(self.tree.vertices) - 1
        # Reference:
        # https://arxiv.org/pdf/1105.1186.pdf, page 15
        # k = int(2 * np.e * np.log(n + 1))
        k = int(2 * np.log(n + 1))
        return min(k, n)

    def build_coverage_tree(self):
        while True:
            x_random = self.sample_biased()
            x_nearest_id, x_nearest = self.tree.get_nearest_config_approx(x_random)
            x_new = self.extend(x_nearest, x_random)
            if self.planning_env.config_validity_checker(x_new) and \
                    self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.add_config(x_new, ws_pose=self.planning_env.robot.compute_forward_kinematics(x_new),
                                           parent_id=x_nearest_id)
                if self.tree.max_coverage >= self.coverage:
                    return self.tree.max_coverage_id

                # self.num_added += 1
            # else:
            #     self.num_discarded += 1
            # print("ratio added: ", self.num_added / (self.num_added + self.num_discarded))
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
        print("tree built")
        plan_list_of_trees = self.plan_to_vertex(max_coverage_id)
        end_time = time.time()

        plan = np.vstack([vertex.config for vertex in plan_list_of_trees])
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total planning time: {:.2f}'.format(end_time - start_time))

        return plan
