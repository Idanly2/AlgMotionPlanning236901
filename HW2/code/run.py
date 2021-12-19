#!/usr/bin/env python

import argparse, numpy, time

from MapEnvironment import MapEnvironment
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from AStarPlanner import AStarPlanner
from AlgCompare import AlgCompare

from IPython import embed
import matplotlib.pyplot as plt


# p2 = {1: (251.45079348883246, 17805), 10: (256.7228714274748, 2103), 20: (256.72287142747473,2894)}
# p1 = {1: (13.242640687119286, 40), 10: (13.242640687119286, 44), 20: (13.242640687119286,46)}
# dates_2 = [1, 10, 20]
# values_2 = [251.45079348883246, 256.7228714274748, 256.72287142747473]
# values_22 = [17805, 2103, 2894]
# values_1 = [13.242640687119286, 13.242640687119286, 13.242640687119286]
# values_12 = [40, 44, 46]
# plt.plot(dates_2, values_1, '-o')
# plt.xlabel("Epsilon")
# plt.ylabel("Cost")
# # plt.title("AStarPlanner - map2 - start:(321, 148) goal:(106, 202)")
# plt.title("AStarPlanner - map1 - start:(0, 8) goal:(5, 4)")
# plt.grid()
# plt.show()

def main(planning_env, planner, start, goal):
    # Notify.
    # input('Press any key to begin planning')

    alg_compare = AlgCompare(planning_env=planning_env)

    # Plan.
    alg_compare.stopwatch_start()
    plan = planner.Plan(start, goal)
    alg_compare.stopwatch_end()
    # TODO (student): Do not shortcut when comparing the performance of algorithms.
    print("Cost: {} , Time(secs): {}".format(alg_compare.plan_cost(plan), alg_compare.plan_time()))

    if isinstance(planner, RRTPlanner):
        planner.VisualizeTree()
        
    # Visualize the final path.
    planning_env.visualize_plan(plan)

    embed()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')
    parser.add_argument('-p', '--planner', type=str, default='rrt',
                        help='The planner to run (astar, rrt, rrtstar)')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)

    args = parser.parse_args()

    # First setup the environment and the robot.
    planning_env = MapEnvironment(args.map, args.start, args.goal)

    # Next setup the planner
    if args.planner == 'astar':
        planner = AStarPlanner(planning_env)
    elif args.planner == 'rrt':
        planner = RRTPlanner(planning_env)
    elif args.planner == 'rrtstar':
        planner = RRTStarPlanner(planning_env)
    else:
        print('Unknown planner option: %s' % args.planner)
        exit(0)

    main(planning_env, planner, args.start, args.goal)
