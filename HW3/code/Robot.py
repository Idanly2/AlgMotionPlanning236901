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

    def compute_distance(self, prev_config, next_config, trivial=False):
        """
        Compute the euclidean distance between two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        @param trivial:
                        True - only compare end effector distance.
                        False - sum of distances between corresponding links' positions.
        """
        # TODO: Task 2.2

        # Trivial implementation - only compare end effector distance
        # Non-trivial - sum of distances between corresponding links' positions

        distance = 0.0

        forward_kinematics_prev = self.compute_forward_kinematics(prev_config)
        forward_kinematics_next = self.compute_forward_kinematics(next_config)
        if trivial:
            distance += np.linalg.norm(forward_kinematics_prev[-1, :] - forward_kinematics_next[-1, :])
        else:
            distance += np.sum(np.linalg.norm(forward_kinematics_prev - forward_kinematics_next, axis=1))

        return distance

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

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1, len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

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
