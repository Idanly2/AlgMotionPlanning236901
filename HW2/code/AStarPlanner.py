import numpy
from heapdict import heapdict
import operator

class Node:
    def __init__(self, config, parent: "Node", g_value=None):
        self.config = config
        self.parent = parent
        self.g_value = g_value


class NodesCollection:
    def __init__(self):
        self._collection = dict()

    def add(self, node: Node):
        assert isinstance(node, Node)
        assert node.config not in self._collection
        self._collection[node.config] = node

    def remove_node(self, node):
        assert node.state in self._collection
        del self._collection[node.config]

    def __contains__(self, state):
        return state in self._collection

    def get_node(self, state):
        assert state in self._collection
        return self._collection[state]


class NodesPriorityQueue:
    def __init__(self):
        self.nodes_queue = heapdict()
        self.state_to_node = dict()

    def add(self, node, priority):
        assert node.config not in self.state_to_node
        self.nodes_queue[node] = priority
        self.state_to_node[node.config] = node

    def pop(self):
        if len(self.nodes_queue) > 0:
            node, priority = self.nodes_queue.popitem()
            del self.state_to_node[node.config]
            return node
        else:
            return None

    def __contains__(self, state):
        return state in self.state_to_node

    def get_node(self, state):
        assert state in self.state_to_node
        return self.state_to_node[state]

    def remove_node(self, node):
        assert node in self.nodes_queue
        del self.nodes_queue[node]
        assert node.config in self.state_to_node
        del self.state_to_node[node.config]

    def __len__(self):
        return len(self.nodes_queue)


class AStarPlanner(object):
    def __init__(self, planning_env, eps=1):
        self.eps = eps
        self.planning_env = planning_env
        self.vertices = []
        self.edges = dict()
        self.open = None
        self.close = None
        self.start_config = None
        self.goal_config = None

    def AddVertex(self, config):
        '''
        Add a state to the tree.
        @param config Configuration to add to the tree.
        '''
        vid = len(self.vertices)
        self.vertices.append(config)
        return vid

    def heuristic(self, config):
        return numpy.sqrt((config[0] - self.goal_config[0]) ** 2 + (config[1] - self.goal_config[1]) ** 2)

    def _calc_node_priority(self, node):
        return node.g_value + self.eps * self.heuristic(node.config)

    def expand_state(self, config):
        directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
        expands = []
        for d in directions:
            new_config = tuple(map(operator.add, config, d))
            if new_config[0] < self.planning_env.xlimit[0] or new_config[0] > self.planning_env.xlimit[1]:
                continue
            elif new_config[1] < self.planning_env.ylimit[0] or new_config[1] > self.planning_env.ylimit[1]:
                continue
            elif self.planning_env.map[new_config] == 0:
                cost = numpy.sqrt(( new_config[0] - config[0] ) ** 2 + ( new_config[1] - config[1] ) ** 2)
                expands.append((new_config, cost))
        return expands

    def show_map(self, config):
        _map = self.planning_env.map.copy()
        _map[config] = 4
        _map[self.start_config] = 2
        _map[self.goal_config] = 3
        print(_map)

    def Plan(self, start_config, goal_config, step_size=0.001):
        # Initialize an empty plan.
        plan = []

        self.start_config = tuple(start_config)
        self.goal_config = tuple(goal_config)

        # Start with adding the start configuration to the tree.
        self.AddVertex(self.start_config)

        # TODO (student): Implement your planner here.
        self.open = NodesPriorityQueue()
        self.close = NodesCollection()

        initial_node = Node(self.start_config, None, g_value=0)
        initial_node_priority = self._calc_node_priority(initial_node)
        self.open.add(initial_node, initial_node_priority)

        n_node_expanded = 0  # count the number of nodes expanded during the algorithm run.

        while True:
            next_node = self.open.pop()
            self.close.add(next_node)
            if next_node.config == self.goal_config:
                cost = next_node.g_value
                print("Done!")
                c_node = next_node
                while c_node.parent is not None:
                    plan.insert(0, c_node.config)
                    c_node = c_node.parent
                plan.insert(0, c_node.config)
                break
                # return cost, n_node_expanded
            n_node_expanded += 1
            for config, cost in self.expand_state(next_node.config):
                self.show_map(config)
                successor_node = Node(config=config, parent=next_node, g_value=next_node.g_value + cost)
                successor_node_priority = self._calc_node_priority(successor_node)
                if config not in self.open and config not in self.close:
                    self.open.add(successor_node, successor_node_priority)
                elif config in self.open:
                    node_in_open = self.open.get_node(config)
                    if successor_node.g_value < node_in_open.g_value:
                        self.open.remove_node(node_in_open)
                        self.open.add(successor_node, successor_node_priority)
                else:  # s is in close
                    node_in_close = self.close.get_node(config)
                    if successor_node.g_value < node_in_close.g_value:
                        self.close.remove_node(node_in_close)
                        self.open.add(successor_node, successor_node_priority)
                # plan.append(config)

        # plan.append(start_config)
        # plan.append(goal_config)
        return numpy.array(plan)

# class BestFirstSearchRobot:
#     def __init__(self):
#         super(BestFirstSearchRobot, self).__init__()
#         self.open = None
#         self.close = None
#         self.name = "abstract best first search robot"
#
#     def solve(self, maze_problem: MazeProblem, time_limit=float("inf"), compute_all_dists=False):
#         start_time = curr_time()
#
#         self.open = NodesPriorityQueue()
#         self.close = NodesCollection()
#
#         if hasattr(self, "_init_heuristic"):  # some heuristics need to be initialized with the maze problem
#             init_heuristic_start_time = curr_time()
#             self._init_heuristic(maze_problem)
#             init_heuristic_time = curr_time() - init_heuristic_start_time
#         else:
#             init_heuristic_time = None
#
#         initial_node = Node(maze_problem.initial_state, None, g_value=0)
#         initial_node_priority = self._calc_node_priority(initial_node)
#         self.open.add(initial_node, initial_node_priority)
#
#         n_node_expanded = 0  # count the number of nodes expanded during the algorithm run.
#
#         while True:
#             if curr_time() - start_time >= time_limit:
#                 no_solution_found = True
#                 no_solution_reason = "time limit exceeded"
#                 break
#
#             next_node = self.open.pop()
#             if next_node is None:
#                 no_solution_found = True
#                 no_solution_reason = "no solution exists"
#                 break
#
#             self.close.add(next_node)
#             if maze_problem.is_goal(next_node.state):
#                 if not compute_all_dists:  # we will use this later, don't change
#                     return GraphSearchSolution(next_node, solve_time=curr_time() - start_time,
#                                                n_node_expanded=n_node_expanded, init_heuristic_time=init_heuristic_time)
#             ############################################################################################################
#             # TODO (EX. 5.1): complete code here
#             n_node_expanded += 1
#             for s, cost in maze_problem.expand_state(next_node.state):
#                 successor_node = Node(state=s, parent=next_node, g_value=next_node.g_value + cost)
#                 successor_node_priority = self._calc_node_priority(successor_node)
#                 if s not in self.open and s not in self.close:
#                     self.open.add(successor_node, successor_node_priority)
#                 elif s in self.open:
#                     node_in_open = self.open.get_node(s)
#                     if successor_node.g_value < node_in_open.g_value:
#                         self.open.remove_node(node_in_open)
#                         self.open.add(successor_node, successor_node_priority)
#                 else:  # s is in close
#                     node_in_close = self.close.get_node(s)
#                     if successor_node.g_value < node_in_close.g_value:
#                         self.close.remove_node(node_in_close)
#                         self.open.add(successor_node, successor_node_priority)
#             ############################################################################################################
#
#         if compute_all_dists:
#             return self.close
#         else:
#             assert no_solution_found
#             return GraphSearchSolution(final_node=None, solve_time=curr_time() - start_time,
#                                        n_node_expanded=n_node_expanded, no_solution_reason=no_solution_reason,
#                                        init_heuristic_time=init_heuristic_time)
#
#
# class AStartPlanner(BestFirstSearchRobot):
#     def __init__(self, heuristic, eps=1, **h_params):
#         super().__init__()
#         assert 0 <= w <= 1
#         self.heuristic = heuristic
#         self.orig_heuristic = heuristic  # in case heuristic is an object function, we need to keep the class
#         self.w = w
#         self.name = f"wA* [{self.w}, {heuristic.__name__}]" if len(h_params) == 0 else \
#             f"wA* [{self.w}, {heuristic.__name__}, {h_params}]"
#         self.h_params = h_params
#
#     def _init_heuristic(self, maze_problem):
#         if isinstance(self.orig_heuristic, type):
#             self.heuristic = self.orig_heuristic(maze_problem, **self.h_params)
#
#     def _calc_node_priority(self, node):
#         # TODO (Ex. 7.1): complete code here
#         return (1 - self.w) * node.g_value + self.w * self.heuristic(node.state)
