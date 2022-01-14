import itertools
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from shapely.geometry import Point, LineString


class Robot(object):

    def __init__(self):

        # define robot properties
        self.links = np.array([80.0, 70.0, 40.0, 40.0])
        self.dim = len(self.links)

        # Robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # The position of the base - currently constant.
        self.base_position = np.array([0.0, 0.0])

        # Visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

        # Parameters regarding converging to view
        self.cost_angle_weight = 2 / self.ee_fov
        self.cost_distance_weight = 1 / self.vis_dist

    def compute_distance(self, prev_config, next_config):
        """
        Compute the euclidean distance between two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        """
        # TODO: Task 2.2

        forward_kinematics_prev = self.compute_forward_kinematics(prev_config)
        forward_kinematics_next = self.compute_forward_kinematics(next_config)
        # Sum of distances between each corre
        return np.sum(np.linalg.norm(forward_kinematics_prev - forward_kinematics_next, axis=1))

    def compute_distance_squared_approx(self, target_ws_pose, ws_poses):
        """
        Fast calculation of
        @param target_ws_pose np array: 8
        @param ws_poses np array Nx8
        """
        # TODO: Task 2.2

        diff = ws_poses - target_ws_pose
        if len(ws_poses.shape) > 1 and ws_poses.shape[0] > 1:
            return np.einsum('ij,ij->i', diff, diff)
        else:
            return np.inner(diff, diff)

    def compute_ee_distance(self, prev_config, next_config):
        forward_kinematics_prev = self.compute_forward_kinematics(prev_config)
        forward_kinematics_next = self.compute_forward_kinematics(next_config)
        # Distance between the end effector in each configuration
        return np.linalg.norm(forward_kinematics_prev[-1, :] - forward_kinematics_next[-1, :])

    def compute_forward_kinematics(self, given_config):
        """
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        """
        # TODO: Task 2.2

        # For each link - compute its angle wrt. global coordinates. Then compute the difference in x, y of its position
        # relative to the last link. Add this difference to the last link position to compute the next link.

        current_link_angle = 0
        current_link_position = self.base_position

        # if len(given_config) != len(self.links):
        #     raise Exception("Given config is not valid.")

        links_positions = np.zeros((len(given_config), 2))

        for i in range(len(given_config)):
            link_orientation = given_config[i]
            link_length = self.links[i]
            current_link_angle = self.compute_link_angle(current_link_angle, link_orientation)
            current_link_position = current_link_position + \
                                    link_length * np.array([np.cos(current_link_angle), np.sin(current_link_angle)])
            links_positions[i, :] = current_link_position

        return links_positions

    def ee_pose_jacobian(self, given_config):
        # ee pose:
        # ( x ) = r1*cos(theta1) + r2*cos(theta1+theta2) + ...
        # ( y ) = r1*sin(theta1) + r2*sin(theta1+theta2) + ...
        # ( theta ) = theta1 + theta2 + theta3 + theta4
        # Therefore:
        # J ((x,y,theta)^T) = ( -r1*sin(theta1) -r2*sin(theta1+theta2) ... )
        #                     (  r1*cos(theta1)  r2*cos(theta1+theta2) ... )
        #                     ( 1 1 1 1 )
        j = np.zeros(shape=(3, 4))
        j[0, 0] = -self.links[0] * np.sin(given_config[0])
        j[0, 1] = -self.links[1] * np.sin(given_config[0] + given_config[1])
        j[0, 2] = -self.links[2] * np.sin(given_config[0] + given_config[1] + given_config[2])
        j[0, 3] = -self.links[3] * np.sin(given_config[0] + given_config[1] + given_config[2] + given_config[3])
        j[1, 0] = self.links[0] * np.cos(given_config[0])
        j[1, 1] = self.links[1] * np.cos(given_config[0] + given_config[1])
        j[1, 2] = self.links[2] * np.cos(given_config[0] + given_config[1] + given_config[2])
        j[1, 3] = self.links[3] * np.cos(given_config[0] + given_config[1] + given_config[2] + given_config[3])
        j[2, :] = 1.0
        return j

    def view_cost_gradient(self, ee_pose, target_point_to_view):
        # Computed using wolfram alpha symbolic calculator
        # Link:
        # https://www.wolframalpha.com/input/?i=partial+%28a*sqrt%28%28f-x%29%5E2+%2B+%28g-y%29%5E2%29%29%5E2+%2B+%28b+*+abs%28arctan%28%28g-y%29%2F%28f-x%29%29+-+z%29%29%5E2
        j = np.zeros(shape=(3,))
        x_diff = target_point_to_view[0] - ee_pose[0]
        y_diff = target_point_to_view[1] - ee_pose[1]
        ang_diff = np.arctan((target_point_to_view[1] - ee_pose[1]) / (target_point_to_view[0] - ee_pose[0])) - \
                   ee_pose[2]
        pos_diff_sq = x_diff * x_diff + y_diff * y_diff

        j[0] = 2 * self.cost_angle_weight * self.cost_angle_weight * y_diff * ang_diff / pos_diff_sq \
               - 2 * self.cost_distance_weight * self.cost_distance_weight * x_diff
        j[1] = -2 * self.cost_angle_weight * self.cost_angle_weight * x_diff * ang_diff / pos_diff_sq \
               - 2 * self.cost_distance_weight * self.cost_distance_weight * y_diff
        j[2] = -2 * self.cost_angle_weight * self.cost_angle_weight * ang_diff

        return j

    def view_cost_function(self, ee_pose, target_point_to_view):

        angle_ee_target = np.arctan2(target_point_to_view[1] - ee_pose[1],
                                     target_point_to_view[0] - ee_pose[0])
        angle_diff = np.abs(self.wrap_to_pi(angle_ee_target - ee_pose[2]))
        position_diff = np.linalg.norm(target_point_to_view - ee_pose[:2])
        return (self.cost_distance_weight * position_diff) ** 2 + (self.cost_angle_weight * angle_diff) ** 2

    def converge_to_view(self, start_config, target_point_to_view, start_step_size=0.1, max_iter=100):
        current_config = start_config.copy()
        step_size = start_step_size
        current_ee_pose = self.compute_ee_pose(current_config)

        num_iter = 0
        last_cost = self.view_cost_function(current_ee_pose, target_point_to_view)
        # While convergence threhold is not reached and max_iter is not exceeded:
        while True:
            if num_iter > max_iter or last_cost < 1 or step_size < 1e-15:
                break
            # Compute gradient at current point
            curr_grad = np.dot(self.ee_pose_jacobian(current_config).T,
                               self.view_cost_gradient(current_ee_pose, target_point_to_view))
            # Get closer to target current via gradient descent
            current_config = self.wrap_to_pi(current_config - step_size * curr_grad)
            current_ee_pose = self.compute_ee_pose(current_config)

            # If cost is diverging, reduce the step size
            current_cost = self.view_cost_function(current_ee_pose, target_point_to_view)
            if current_cost > last_cost:
                step_size /= 2
            last_cost = current_cost
            num_iter += 1

        return current_config

    def flattened_forward_kinematics(self, given_config):
        return self.compute_forward_kinematics(given_config).flatten()

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        return self.wrap_to_pi(np.sum(given_config))

    def compute_ee_position(self, config):
        ee_position = np.zeros(shape=(2,))
        ee_position[0] = self.links[0] * np.cos(config[0]) + self.links[1] * np.cos(config[0] + config[1]) + self.links[
            2] * np.cos(config[0] + config[1] + config[2]) + self.links[3] * np.cos(
            config[0] + config[1] + config[2] + config[3])
        ee_position[1] = self.links[0] * np.sin(config[0]) + self.links[1] * np.sin(config[0] + config[1]) + self.links[
            2] * np.sin(config[0] + config[1] + config[2]) + self.links[3] * np.sin(
            config[0] + config[1] + config[2] + config[3])
        return ee_position

    def compute_ee_pose(self, config):
        ee_pose = np.zeros(shape=(3,))
        ee_pose[:2] = self.compute_ee_position(config)
        ee_pose[2] = self.compute_ee_angle(config)
        return ee_pose

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2 * np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2 * np.pi
        else:
            return link_angle + given_angle

    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        # TODO: Task 2.2

        # Assuming that the links are infinitely thin, intersections only occur when one link is crossing another one
        # (as opposed to 2 links being too close to each other).
        # base_with_links = np.vstack([self.base_position, robot_positions])
        manipulator_lines = LineString([robot_positions[i, :] for i in range(len(robot_positions))])
        # is_simple is True when the LineString does not self-intersect.
        return manipulator_lines.is_simple

    def wrap_to_pi(self, angles):
        return (angles + np.pi) % (2 * np.pi) - np.pi
