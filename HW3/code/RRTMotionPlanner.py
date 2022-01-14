import numpy as np
from RRTTree import RRTTree
import time
import operator
from Robot import Robot


class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

        # Parameters
        self.step_size = 0.2
        self.goal_reach_dist_threshold = 2.0

        self.num_added = 0
        self.num_discarded = 0

    def sample_biased(self):
        """
        Sample a random state, with a certain bias to choose the goal state.
        :param bias: Bias parameter to draw the goal state instead of a random state.
        :return: Random state of the map.
        """
        p = np.random.random()
        if p < self.goal_prob:
            return self.get_goal_config()
        else:
            theta_1 = np.random.uniform(0, np.pi / 2)
            while True:
                theta_2 = np.random.uniform(-np.pi, np.pi)
                is_validate_robot = self.planning_env.robot.validate_robot(
                    np.vstack((self.planning_env.robot.base_position,
                               self.planning_env.robot.compute_forward_kinematics(
                                   [theta_1, theta_2]))))
                if np.cos(theta_1) + np.cos(theta_1 + theta_2) >= 0 and np.sin(theta_1) + np.sin(
                        theta_1 + theta_2) >= 0 and is_validate_robot:
                    break
            while True:
                theta_3 = np.random.uniform(-np.pi, np.pi)
                is_validate_robot = self.planning_env.robot.validate_robot(
                    np.vstack((self.planning_env.robot.base_position,
                               self.planning_env.robot.compute_forward_kinematics(
                                   [theta_1, theta_2, theta_3]))))
                if np.cos(theta_1) + np.cos(theta_1 + theta_2) + np.cos(theta_1 + theta_2 + theta_3) >= 0 and np.sin(
                        theta_1) + np.sin(
                    theta_1 + theta_2) + np.sin(theta_1 + theta_2 + theta_3) >= 0 and is_validate_robot:
                    break
            while True:
                theta_4 = np.random.uniform(-np.pi, np.pi)
                is_validate_robot = self.planning_env.robot.validate_robot(
                    np.vstack((self.planning_env.robot.base_position,
                               self.planning_env.robot.compute_forward_kinematics(
                                   [theta_1, theta_2, theta_3, theta_4]))))
                if np.cos(theta_1) + np.cos(theta_1 + theta_2) + np.cos(theta_1 + theta_2 + theta_3) \
                        + np.cos(theta_1 + theta_2 + theta_3 + theta_4) >= 0 and np.sin(theta_1) \
                        + np.sin(theta_1 + theta_2) + np.sin(theta_1 + theta_2 + theta_3) \
                        + np.sin(theta_1 + theta_2 + theta_3 + theta_4) >= 0 and is_validate_robot:
                    break
            return np.array([theta_1, theta_2, theta_3, theta_4])

    def get_goal_config(self):
        return self.planning_env.goal

    def add_config(self, x_add, ws_pose, parent_id=None, cost=0):
        x_add_id = self.tree.add_vertex(x_add)
        self.tree.vertices[x_add_id].cost = cost
        self.tree.vertices[x_add_id].ws_pose = ws_pose
        if parent_id is not None:
            self.tree.add_edge(parent_id, x_add_id)  # Edges indicate the parent of each node
        return x_add_id

    def build_tree(self, goal_config):
        while True:
            x_random = self.sample_biased()
            x_nearest_id, x_nearest = self.tree.get_nearest_config_approx(x_random)
            x_new = self.extend(x_nearest, x_random)
            if self.planning_env.config_validity_checker(x_new) and \
                    self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.add_config(x_new, ws_pose=self.planning_env.robot.compute_forward_kinematics(x_new),
                                           parent_id=x_nearest_id)
                # self.num_added += 1
                if self.planning_env.robot.compute_ee_distance(x_new, goal_config) < self.goal_reach_dist_threshold:
                    return x_new_id  # Technically this will always be len(self.tree.vertices) - 1
            # else:
            # self.num_discarded += 1
            # print("ratio added: ", self.num_added / (self.num_added + self.num_discarded))

    def plan_to_vertex(self, vertex_id):
        plan = []
        id_to_add = vertex_id

        plan.append(self.tree.vertices[id_to_add])
        while id_to_add != 0:  # self.tree.GetRootID():
            id_to_add = self.tree.edges[id_to_add]
            plan.append(self.tree.vertices[id_to_add])
        plan.reverse()
        return np.array(plan)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.time()
        # Start with adding the start configuration to the tree.
        start_config = self.planning_env.start
        self.add_config(start_config, self.planning_env.robot.compute_forward_kinematics(start_config))

        # TODO: Task 2.3
        goal_config = self.planning_env.goal
        goal_vertex_id = self.build_tree(goal_config=goal_config)
        plan_list_of_trees = self.plan_to_vertex(goal_vertex_id)
        end_time = time.time()
        plan = np.vstack([vertex.config for vertex in plan_list_of_trees])
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total planning time: {:.2f}'.format(end_time - start_time))

        return plan

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # TODO: Task 2.3
        distance = 0.0
        for i in range(len(plan) - 1):
            distance += self.planning_env.robot.compute_distance(plan[i], plan[i + 1])
        return distance

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # TODO: Task 2.3
        if self.ext_mode == "E1":
            return rand_config
        else:
            # Need to determine the direction of extension by wrapping angle diff
            dir = self.planning_env.robot.wrap_to_pi(rand_config - near_config)
            norm = np.linalg.norm(dir)
            if norm == 0:
                return rand_config
            else:
                scale = np.minimum(norm, self.step_size / norm)
                # Normalize the step vector to the minimal of step_size, or until rand_config is reached
                dir = scale * dir  # Scaling direction to be of size at most step_size
                return self.planning_env.robot.wrap_to_pi(near_config + dir)
