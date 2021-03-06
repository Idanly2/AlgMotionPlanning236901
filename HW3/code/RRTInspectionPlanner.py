import numpy as np
from RRTTree import RRTTree
from RRTMotionPlanner import RRTMotionPlanner
import time


# import networkx as nx


class RRTInspectionPlanner(RRTMotionPlanner):

    def __init__(self, planning_env, ext_mode, goal_prob, coverage, max_time=3600, competition=True):
        super(RRTInspectionPlanner, self).__init__(planning_env, ext_mode, goal_prob)
        # set environment and search tree
        self.planning_env = planning_env

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage

        self.num_converge_success = 0
        self.num_converge_fail = 0

        self.success_history_length = 100  # How many samples to keep in self.success_history
        self.success_history = [0] * self.success_history_length
        self.extend_rate_to_goal_prob = 1.0
        self.extend_success_rate = 0.0

        self.max_time = max_time
        self.max_iter = 5000

        self.history_index = 0
        self.iter_number = 0

        self.competition = competition
        if self.competition:
            self.dynamic_goal_prob = True
            self.deterministic_goal_choice = False
            self.deterministic_goal_choice_start_config = False
            self.rewire = False
            self.coverage = 1.0
        else:
            self.dynamic_goal_prob = False
            self.deterministic_goal_choice = False
            self.deterministic_goal_choice_start_config = False
            self.rewire = False

    def init_tree(self, start_config, target_inspection_points):
        # Start with adding the start configuration to the tree.
        self.tree = RRTTree(self.planning_env, task="ip", target_inspection_points=target_inspection_points)
        self.add_config(start_config, self.planning_env.robot.compute_forward_kinematics(start_config))

    def get_inspected_points(self, config):
        return self.planning_env.compute_intersect_of_points(self.planning_env.get_inspected_points(config),
                                                             self.tree.target_inspection_points)

    def sample_biased(self):
        """
        Sample a random state, with a certain bias to choose the goal state.
        :param bias: Bias parameter to draw the goal state instead of a random state.
        :return: Random state of the map.
        """
        if self.dynamic_goal_prob:
            goal_prob = self.extend_rate_to_goal_prob * self.extend_success_rate
        else:
            goal_prob = self.goal_prob

        # print("goal_prob: ", goal_prob)

        p = np.random.random()
        if p < goal_prob:
            return self.get_goal_config()
        else:
            return self.sample_random_config()

    def get_goal_config(self):
        curr_best_inspected_points = self.tree.vertices[self.tree.max_coverage_id].inspected_points
        inspect_points_left = self.planning_env.compute_diff_of_points(self.tree.target_inspection_points,
                                                                       curr_best_inspected_points)
        if inspect_points_left.size == 0:
            return self.sample_random_config()

        # Choose closest inspection point to best configuration
        if self.deterministic_goal_choice:
            ee_position_best = self.planning_env.robot.compute_ee_position(
                self.tree.vertices[self.tree.max_coverage_id].config)
            diff = inspect_points_left - ee_position_best
            closest_ip_to_best = np.argmin(np.einsum('ij,ij->i', diff, diff))
            chosen_inspect_point = inspect_points_left[closest_ip_to_best]
        # Choose randomly an inspection point
        else:
            random_inspect_id = np.random.randint(len(inspect_points_left))
            chosen_inspect_point = inspect_points_left[random_inspect_id]

        if self.deterministic_goal_choice_start_config:
            closest_id, closest_config = self.tree.get_nearest_config_ee_point(chosen_inspect_point)
        else:
            closest_id = np.random.randint(low=0, high=len(self.tree.vertices))
            closest_config = self.tree.vertices[closest_id].config

        converged_config = self.planning_env.robot.converge_to_view(closest_config, chosen_inspect_point)
        if self.get_inspected_points(converged_config).size == 0 or \
                self.planning_env.compute_intersect_of_points(self.get_inspected_points(converged_config),
                                                              inspect_points_left).size == 0:
            converged_config = self.sample_random_config()
            # print("converge view: ", False)
        # else:
        #     print("converge view: ", True)

        return converged_config

    def add_config(self, x_add, ws_pose, parent_id=None):
        inspected_points_at_config = self.get_inspected_points(x_add)
        if parent_id is not None:
            inspected_points = self.planning_env.compute_union_of_points(inspected_points_at_config,
                                                                         self.tree.vertices[parent_id].inspected_points)
        else:
            inspected_points = inspected_points_at_config

        x_add_id = self.tree.add_vertex(x_add, inspected_points=inspected_points)
        self.tree.vertices[x_add_id].ws_pose = ws_pose

        if parent_id is not None:
            self.tree.add_edge(parent_id, x_add_id)  # Edges indicate the parent of each node
            if self.rewire:
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
                inspected_points_at_child_config = self.get_inspected_points(x_child.config)
            current_coverage = self.planning_env.compute_coverage(x_child.inspected_points,
                                                                  self.tree.target_inspection_points)
            potential_points_union = self.planning_env.compute_union_of_points(x_potential_parent.inspected_points,
                                                                               inspected_points_at_child_config)
            potential_coverage = self.planning_env.compute_coverage(potential_points_union,
                                                                    self.tree.target_inspection_points)

            # This condition sometimes finds a path faster, but the path quality is much worse
            # if self.tree.is_leaf(x_child_id) and (potential_coverage > current_coverage or \
            #     potential_cost < x_child.cost and potential_coverage == current_coverage):
            # This makes the search slower, but path quality significantly better
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

    def check_extend_possible(self, x_new, x_nearest):
        success = 0
        if self.planning_env.config_validity_checker(x_new) and \
                self.planning_env.edge_validity_checker(x_nearest, x_new):
            success = 1

        history_index_wrapped = self.history_index % self.success_history_length
        self.extend_success_rate = self.extend_success_rate + \
                                   (success - self.success_history[history_index_wrapped]) / self.success_history_length
        self.success_history[history_index_wrapped] = success

        self.history_index += 1

        return bool(success)

    def build_coverage_tree(self, start_config, target_inspection_points, min_iter=None, max_iter=None):
        self.init_tree(start_config=start_config, target_inspection_points=target_inspection_points)
        start_time = time.time()
        build_tree_iter_number = 0
        while True:
            if (min_iter is not None and build_tree_iter_number > 0 and build_tree_iter_number % min_iter == 0
                and self.tree.max_coverage > 0) or (max_iter is not None and self.iter_number >= max_iter):
                return self.tree.max_coverage_id

            if build_tree_iter_number % 100 == 0 and time.time() - start_time > self.max_time:
                # Something got stuck in this tree, so throw it away and start all over again
                print("Reset tree")
                self.init_tree(start_config=start_config,
                               target_inspection_points=target_inspection_points)
                start_time = time.time()
            x_random = self.sample_biased()
            x_nearest_id, x_nearest = self.tree.get_nearest_config_approx(x_random)
            x_new = self.extend(x_nearest, x_random)
            if self.check_extend_possible(x_new, x_nearest):
                x_new_id = self.add_config(x_new, ws_pose=self.planning_env.robot.compute_forward_kinematics(x_new),
                                           parent_id=x_nearest_id)
                if self.tree.max_coverage >= self.coverage:
                    return self.tree.max_coverage_id
                # self.num_added += 1
            # else:
            #     self.num_discarded += 1
            # print("ratio added: ", self.num_added / (self.num_added + self.num_discarded))
            # print("max_coverage: ", self.tree.max_coverage)

            build_tree_iter_number += 1
            self.iter_number += 1
            # print("iter_number: ", self.iter_number)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()

        start_config = self.planning_env.start
        target_inspection_points = self.planning_env.inspection_points

        agg_plan = []
        agg_plan.append(start_config)

        while target_inspection_points.size > 0 and self.iter_number < self.max_iter:
            max_coverage_id = self.build_coverage_tree(start_config=start_config,
                                                       target_inspection_points=target_inspection_points,
                                                       min_iter=200, max_iter=self.max_iter)
            plan_list_of_trees = self.plan_to_vertex(max_coverage_id)

            start_config = self.tree.vertices[max_coverage_id].config
            target_inspection_points = self.planning_env.compute_diff_of_points(
                target_inspection_points, self.tree.vertices[max_coverage_id].inspected_points)
            plan = [vertex.config for vertex in plan_list_of_trees]
            agg_plan.extend(plan[1:])

        plan_np = np.vstack(agg_plan)
        end_time = time.time()

        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan_np)))
        print('Total planning time: {:.2f}'.format(end_time - start_time))

        final_coverage = self.planning_env.compute_coverage(
            self.planning_env.compute_diff_of_points(self.planning_env.inspection_points, target_inspection_points),
            self.planning_env.inspection_points)

        print("final coverage: ", final_coverage)
        print("final iteration number: ", self.iter_number)

        return plan_np
