import numpy as np
from RRTTree import RRTTree
import time
import operator
from Robot import Robot


class RRTMotionPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob, step_size=50):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

        self.step_size = step_size

    def sample_biased(self):
        """
        Sample a random state, with a certain bias to choose the goal state.
        :param bias: Bias parameter to draw the goal state instead of a random state.
        :return: Random state of the map.
        """
        p = np.random.random()
        if p < self.goal_prob:
            return self.planning_env.goal
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

    def add_config(self, x_add, parent_id=None):
        x_add_id = self.tree.add_vertex(x_add)
        if parent_id is not None:
            self.tree.add_edge(parent_id, x_add_id)  # Edges indicate the parent of each node
        return x_add_id

    def build_tree(self, goal_config, step_size):
        while True:
            x_random = self.sample_biased()
            x_nearest_id, x_nearest = self.tree.get_nearest_config(x_random)
            x_new = self.extend(x_nearest, x_random)
            if self.planning_env.config_validity_checker(x_new) and \
                    self.planning_env.edge_validity_checker(x_nearest, x_new):
                x_new_id = self.add_config(x_new, x_nearest_id)
                if self.planning_env.robot.compute_distance(x_new, goal_config) < 0.1:
                    return x_new_id  # Technically this will always be len(self.tree.vertices) - 1

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
        self.add_config(start_config)
        # Initialize an empty plan.
        plan = []

        # TODO: Task 2.3
        goal_config = self.planning_env.goal
        goal_vertex_id = self.build_tree(goal_config=goal_config, step_size=self.step_size)
        plan_list_of_trees = self.plan_to_vertex(goal_vertex_id)
        plan = plan_list_of_trees[0].config
        for i in range(1,len(plan_list_of_trees)):
            plan = np.vstack((plan, plan_list_of_trees[i].config))

        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time() - start_time))

        return np.array(plan)

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
        eta = 1000
        if self.ext_mode == 'E1':
            eta = 1000

        if self.planning_env.robot.compute_distance(rand_config, near_config) < eta:
            return rand_config
        else:
            dir = np.array(tuple(map(operator.sub, rand_config, near_config)))
            norm = np.linalg.norm(dir)
            if norm == 0:
                return rand_config
            else:
                dir = (eta / norm) * dir  # Scaling direction to be of size at most eta
                dir = dir.astype(int)  # Removing fractional part to get pixel coordinates
                return [near_config[0] + dir[0], near_config[1] + dir[1]]
