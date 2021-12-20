#!/usr/bin/env python

import argparse, numpy, time
from statistics import mean
from sys import maxsize

import numpy as np

from MapEnvironment import MapEnvironment
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from AStarPlanner import AStarPlanner
from AlgCompare import AlgCompare

from IPython import embed
import matplotlib.pyplot as plt


def test_planner(alg_compare, planner, start, goal):
    # Notify.
    # input('Press any key to begin planning')

    alg_compare.stopwatch_start()
    plan = planner.Plan(start, goal)
    alg_compare.stopwatch_end()
    alg_compare.register_plan_cost(plan)

    # Visualize the tree if it's a tree planner
    if isinstance(planner, RRTPlanner) or isinstance(planner, RRTStarPlanner):
        planner.VisualizeTree()

    # Visualize the final path.
    planning_env.visualize_plan(plan)


def print_statistics(costs, times, desc_str, file_handle):
    mean_cost = mean(costs)
    mean_time = mean(times)
    print(desc_str + " Cost: {} , Time(secs): {}".format(mean_cost, mean_time), file=file_handle)


def print_success_graphs(costs, times, graph_title):
    costs_np = numpy.array(costs)
    times_np = numpy.array(times)

    times_sorted_indices = numpy.argsort(times_np)
    costs_np = costs_np[times_sorted_indices]
    times_np = times_np[times_sorted_indices]

    successes = numpy.arange(start=1, stop=len(times_np) + 1) / len(times_np)

    plt.plot(times_np, successes, '-o')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Success rate")
    plt.title(graph_title)
    plt.grid()
    plt.savefig(graph_title + '_success_rate_vs_time.png', bbox_inches='tight')
    plt.close()

    plt.plot(times_np, costs_np, '-o')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Plan Cost (pixels)")
    plt.title(graph_title)
    plt.grid()
    plt.savefig(graph_title + '_plan_cost_vs_time.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')
    parser.add_argument('-p', '--planner', type=str, default='rrt',
                        help='The planner to run (astar, rrt, rrtstar)')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)

    args = parser.parse_args()
    env_desc = str(args.map) + " - start: " + str(args.start) + ", goal: " + str(args.goal)

    # Next setup the planner
    if args.planner == 'astar':
        epsilons = [1, 10, 20]
        for eps in epsilons:
            description_str = args.planner + "_eps_" + str(eps)
            planning_env = MapEnvironment(args.map, args.start, args.goal, output_plot_desc=description_str)

            alg_compare = AlgCompare(planning_env=planning_env)
            planner = AStarPlanner(planning_env, eps=eps)
            test_planner(alg_compare=alg_compare, planner=planner, start=args.start, goal=args.goal)

            costs, times = alg_compare.costs, alg_compare.times

            with open(description_str + '_summary.txt', 'w') as f:
                print_statistics(costs, times, description_str, f)

    elif args.planner == 'rrt':
        step_sizes = [5, 5e1, 5e9]
        goal_biases = [0.05, 0.2]
        for step_size in step_sizes:
            for bias in goal_biases:
                description_str = args.planner + "_step_" + str(step_size) + "_bias_" + str(bias)
                planning_env = MapEnvironment(args.map, args.start, args.goal, goal_bias_sample=bias,
                                              output_plot_desc=description_str)

                alg_compare = AlgCompare(planning_env=planning_env)
                for i in range(10):
                    planner = RRTPlanner(planning_env, step_size=step_size)
                    test_planner(alg_compare=alg_compare, planner=planner, start=args.start, goal=args.goal)

                costs, times = alg_compare.costs, alg_compare.times

                with open(description_str + '_summary.txt', 'w') as f:
                    print_statistics(costs, times, description_str, f)
                print_success_graphs(costs, times, description_str + " - " + env_desc)

    elif args.planner == 'rrtstar':
        step_sizes = [5, 5e1]
        ks = ['log', 3, 10, 30, 70]
        for step_size in step_sizes:
            for k in ks:
                description_str = args.planner + "_step_" + str(step_size) + '_k_' + str(k)
                planning_env = MapEnvironment(args.map, args.start, args.goal, output_plot_desc=description_str)

                alg_compare = AlgCompare(planning_env=planning_env)
                for i in range(10):
                    planner = RRTStarPlanner(planning_env, step_size=step_size, k_nearest_neighbors=k)
                    test_planner(alg_compare=alg_compare, planner=planner, start=args.start, goal=args.goal)

                costs, times = alg_compare.costs, alg_compare.times

                with open(description_str + '_summary.txt', 'w') as f:
                    print_statistics(costs, times, description_str, f)
                print_success_graphs(costs, times, description_str + " - " + env_desc)
    else:
        print('Unknown planner option: %s' % args.planner)
        exit(0)
