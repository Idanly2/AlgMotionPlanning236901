import numpy
from IPython import embed
from matplotlib import pyplot as plt

from bresenham import bresenhamline


class MapEnvironment(object):
    output_counter = 0

    def __init__(self, mapfile, start, goal, goal_bias_sample=0.05, output_plot_desc=None):

        # Obtain the boundary limits.
        # Check if file exists.
        self.map = numpy.loadtxt(mapfile)
        self.xlimit = [0, self.map.shape[1]]
        self.ylimit = [0, self.map.shape[0]]

        # Check if start and goal are within limits and collision free
        if not self.state_validity_checker(start) or not self.state_validity_checker(goal):
            raise ValueError('Start and Goal state must be within the map limits')
            exit(0)

        self.start_config = start
        self.goal_config = goal
        self.output_plot_desc = output_plot_desc
        self.goal_bias_sample = goal_bias_sample

    def sample_biased(self):
        """
        Sample a random state, with a certain bias to choose the goal state.
        :param bias: Bias parameter to draw the goal state instead of a random state.
        :return: Random state of the map.
        """
        p = numpy.random.random()
        if p < self.goal_bias_sample:
            return self.goal_config
        else:
            return [numpy.random.randint(self.xlimit[0], self.xlimit[1]),
                    numpy.random.randint(self.ylimit[0], self.ylimit[1])]

    def compute_distance(self, start_config, end_config):
        #
        # TODO: Implement a function which computes the distance between
        # two configurations.
        #
        return numpy.sqrt((start_config[0] - end_config[0]) ** 2 + (start_config[1] - end_config[1]) ** 2)

    def state_validity_checker(self, config):
        #
        # TODO: Implement a state validity checker
        # Return true if valid.
        #
        if config[0] < self.xlimit[0] or config[0] > self.xlimit[1] or config[1] < self.ylimit[0] or config[1] > \
                self.ylimit[1] or self.map[config[1], config[0]] == 1:
            return False
        return True

    def edge_validity_checker(self, config1, config2):

        #
        # TODO: Implement an edge validity checker
        #
        #
        start = numpy.array([[config1[1], config1[0]]])
        end = numpy.array([[config2[1], config2[0]]])
        line = bresenhamline(start, end)
        return numpy.all(self.map[line[:, 0], line[:, 1]] == 0)

    def compute_heuristic(self, config):
        #
        # TODO: Implement a function to compute heuristic.
        #
        return numpy.sqrt((config[0] - self.goal_config[0]) ** 2 + (config[1] - self.goal_config[1]) ** 2)

    def visualize_plan(self, plan):
        '''
        Visualize the final path
        @param plan Sequence of states defining the plan.
        input plan should be in [x, y] convention.
        '''
        plt.imshow(self.map, interpolation='nearest')
        for i in range(numpy.shape(plan)[0] - 1):
            x = [plan[i, 0], plan[i + 1, 0]]
            y = [plan[i, 1], plan[i + 1, 1]]
            plt.plot(x, y, 'k')

        graph_title = self.output_plot_desc + '_plan'
        plt.title(graph_title)

        if self.output_plot_desc:
            plt.savefig(graph_title + '_' + str(MapEnvironment.output_counter) + '.png', bbox_inches='tight')
            plt.close()
            MapEnvironment.output_counter += 1
        else:
            plt.show()

    def visualize_tree(self, lines):
        '''
        Visualize an iterable of lines
        input lines should be in [x1, y1, x2, y2] convention.
        '''
        plt.imshow(self.map, interpolation='nearest')
        for i in range(len(lines)):
            x = [lines[i, 0], lines[i, 2]]
            y = [lines[i, 1], lines[i, 3]]
            plt.plot(x, y, 'k')

        graph_title = self.output_plot_desc + '_tree'
        plt.title(graph_title)

        if self.output_plot_desc:
            plt.savefig(graph_title + '_' + str(MapEnvironment.output_counter) + '.png', bbox_inches='tight')
            plt.close()
            MapEnvironment.output_counter += 1
        else:
            plt.show()
