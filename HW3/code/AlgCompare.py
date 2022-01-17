import time


class AlgCompare(object):

    def __init__(self):
        self.start_time = 0
        self.end_time = 0

        self.times = list()
        self.costs = list()

    def register_plan_cost(self, plan_cost):
        self.costs.append(plan_cost)

    def stopwatch_start(self):
        self.start_time = time.time()

    def stopwatch_end(self):
        self.end_time = time.time()
        self.times.append(self.end_time - self.start_time)
