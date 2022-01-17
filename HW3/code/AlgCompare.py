import time

import MapEnvironment


class AlgCompare(object):
    planning_env: MapEnvironment

    def __init__(self, planning_env: MapEnvironment):
        self.planning_env = planning_env
        self.start_time = 0
        self.end_time = 0

        self.times = list()
        self.costs = list()

    def register_plan_cost(self, plan):
        plan_cost = sum([self.planning_env.compute_distance(plan[i - 1], plan[i]) for i in range(1, len(plan))])
        self.costs.append(plan_cost)

    def stopwatch_start(self):
        self.start_time = time.time()

    def stopwatch_end(self):
        self.end_time = time.time()
        self.times.append(self.end_time - self.start_time)
