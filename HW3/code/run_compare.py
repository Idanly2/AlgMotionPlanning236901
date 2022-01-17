#!/usr/bin/env python

import argparse, time
import numpy as np
from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from AlgCompare import AlgCompare
import matplotlib.pyplot as plt

import imageio


def test_planner(alg_compare, planner):
    alg_compare.stopwatch_start()
    plan = planner.plan()
    alg_compare.stopwatch_end()

    cost = planner.compute_cost(plan)
    alg_compare.register_plan_cost(cost)

    # print total path cost and time
    print('Total cost of path: {:.2f}'.format(cost))
    print('Total planning time: {:.2f}'.format(alg_compare.end_time - alg_compare.start_time))

    # Visualize the tree
    planner.tree.visualize_tree()

    # Visualize the final path.
    planning_env.visualize_plan(plan)


def print_statistics(costs, times, desc_str, file_handle):
    mean_cost = np.mean(costs)
    mean_time = np.mean(times)
    print(desc_str + " Cost: {} , Time(secs): {}".format(mean_cost, mean_time), file=file_handle)


def print_success_graphs(costs, times, graph_title):
    costs_np = np.array(costs)
    times_np = np.array(times)

    times_sorted_indices = np.argsort(times_np)
    costs_np = costs_np[times_sorted_indices]
    times_np = times_np[times_sorted_indices]

    successes = np.arange(start=1, stop=len(times_np) + 1) / len(times_np)

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
    parser.add_argument('-map', '--map', type=str, default='map_mp.json',
                        help='Json file name containing all map information')
    parser.add_argument('-task', '--task', type=str, default='mp',
                        help='choose from mp (motion planning) and ip (inspection planning)')
    parser.add_argument('-ext_mode', '--ext_mode', type=str, default='E2', help='edge extension mode')
    parser.add_argument('-goal_prob', '--goal_prob', type=float, default=0.05, help='probability to draw goal vertex')
    parser.add_argument('-coverage', '--coverage', type=float, default=0.5,
                        help='percentage of points to inspect (inspection planning)')
    args = parser.parse_args()

    # setup the planner
    if args.task == 'mp':
        goal_biases = [0.05, 0.2]
        for goal_bias in goal_biases:
            # prepare the map
            description_str = args.task + "_" + args.map + "_bias_" + str(goal_bias)
            alg_compare = AlgCompare()
            for i in range(10):
                planning_env = MapEnvironment(json_file=args.map, task=args.task, output_plot_desc=description_str)
                planner = RRTMotionPlanner(planning_env=planning_env, ext_mode=args.ext_mode, goal_prob=goal_bias)
                test_planner(alg_compare=alg_compare, planner=planner)
                del planner, planning_env

            costs, times = alg_compare.costs, alg_compare.times

            with open(description_str + '_summary.txt', 'w') as f:
                print_statistics(costs, times, description_str, f)
            print_success_graphs(costs, times, description_str)

    elif args.task == 'ip':
        coverages = [0.5, 0.75]
        for coverage in coverages:
            # prepare the map
            description_str = args.task + "_" + args.map + "_coverage_" + str(coverage)
            alg_compare = AlgCompare()
            for i in range(10):
                planning_env = MapEnvironment(json_file=args.map, task=args.task, output_plot_desc=description_str)
                planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=args.ext_mode,
                                               goal_prob=args.goal_prob, coverage=coverage, competition=False)
                test_planner(alg_compare=alg_compare, planner=planner)
                del planner, planning_env

            costs, times = alg_compare.costs, alg_compare.times

            with open(description_str + '_summary.txt', 'w') as f:
                print_statistics(costs, times, description_str, f)
            print_success_graphs(costs, times, description_str)
    else:
        print('Unknown task option: %s' % args.task)
        exit(1)
