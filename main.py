# -*- coding:utf-8 -*-
# @FileName  :main.py
# @Time      :2024/8/1 下午12:55
# @Author    :ZMFY
# Description:

import argparse
import os
import time
import random
import numpy as np

from instance import Instance
from ecbs import ECBS
from conflict import ConflictSelection
from nodes import NodeSelection
from cbs import HighLevelSolverType
from cbs_heuristic import HeuristicType


def main():
    parser = argparse.ArgumentParser(description="Allowed options")
    parser.add_argument("--map", "-m", help="input file for map",
                        default='./instances/random-32-32-20.map')
    parser.add_argument("--agents", "-a", help="input file for agents",
                        default='./instances/random-32-32-20-random-1.scen')
    parser.add_argument('--output_dir', '-o', help="output directory", default='./output')
    parser.add_argument("--outputRes", help="output file for statistics",
                        default='./random-32-32-20.csv')
    parser.add_argument("--outputPaths", help="output file for paths",
                        default='./random-32-32-20_path.txt')
    parser.add_argument("--agentNum", "-k", type=int, default=50, help="number of agents")
    parser.add_argument("--cutoffTime", "-t", type=float, default=60, help="cutoff time (seconds)")
    parser.add_argument("--screen", "-s", type=int, default=2, help="screen option (0: none; 1: results; 2:all)")
    parser.add_argument("--stats", type=bool, default=False, help="write to files some detailed statistics")
    parser.add_argument("--highLevelSolver", default="EES", help="the high-level solver (A*, A*eps, EES, NEW)")
    parser.add_argument("--lowLevelSolver", type=bool, default=True, help="using suboptimal solver in the low level")
    parser.add_argument("--inadmissibleH", default="Global",
                        help="inadmissible heuristics (Zero, Global, Path, Local, Conflict)")
    parser.add_argument("--suboptimality", type=float, default=1.3, help="sub optimality bound")
    parser.add_argument("--heuristics", default="Zero",
                        help="admissible heuristics for the high-level search (Zero, CG,DG, WDG)")
    parser.add_argument("--prioritizingConflicts", type=bool, default=True, help="conflict prioritization")
    parser.add_argument("--bypass", type=bool, default=True, help="Bypass1")
    parser.add_argument("--disjointSplitting", type=bool, default=False, help="disjoint splitting")
    parser.add_argument("--rectangleReasoning", type=bool, default=True, help="rectangle reasoning")
    parser.add_argument("--corridorReasoning", type=bool, default=True, help="corridor reasoning")
    parser.add_argument("--targetReasoning", type=bool, default=True, help="target reasoning")
    parser.add_argument("--sipp", type=bool, default=True, help="using SIPPS as the low-level solver")
    parser.add_argument("--restart", type=int, default=0, help="rapid random restart times")

    args = parser.parse_args()
    random.seed(520)
    np.random.seed(520)

    if args.suboptimality < 1:
        print("Suboptimal bound should be at least 1!")
        return -1

    high_level_solver = args.highLevelSolver
    high_level_solver_dict = {
        'A*': HighLevelSolverType.ASTAR, "A*eps": HighLevelSolverType.ASTAREPS,
        'EES': HighLevelSolverType.EES, 'NEW': HighLevelSolverType.NEW,
    }
    if high_level_solver not in ["A*", "A*eps", "EES", "NEW"]:
        print("WRONG high level solver!")
        return -1
    solver_type = high_level_solver_dict[high_level_solver]

    if high_level_solver == "A*" and args.suboptimality > 1:
        print("A* cannot perform suboptimal search!")
        return -1

    heuristics = args.heuristics
    heuristics_dict = {
        "Zero": HeuristicType.ZERO, 'CG': HeuristicType.CG,
        'DG': HeuristicType.DG, 'WDG': HeuristicType.WDG,
        'Global': HeuristicType.GLOBAL, 'Path': HeuristicType.PATH,
        'Local': HeuristicType.LOCAL, 'Conflict': HeuristicType.CONFLICT,
    }
    if heuristics not in ["Zero", "CG", "DG", "WDG"]:
        print("WRONG heuristics strategy!")
        return -1
    h = heuristics_dict[heuristics]

    if heuristics in ["CG", "DG"] and args.lowLevelSolver:
        print("CG or DG heuristics do not work with low level of suboptimal search!")
        return -1

    inadmissible_heuristics = args.inadmissibleH
    if high_level_solver in ["A*", "A*eps"] or inadmissible_heuristics == "Zero":
        h_hat = heuristics_dict["Zero"]
    elif inadmissible_heuristics in ["Global", "Path", "Local", "Conflict"]:
        h_hat = heuristics_dict[inadmissible_heuristics]
    else:
        print("WRONG inadmissible heuristics strategy!")
        return -1

    conflict = ConflictSelection.EARLIEST
    node_selection = NodeSelection.NODE_CONFLICTPAIRS

    random.seed(int(time.time()))

    # Load the instance
    instance = Instance(args.map, args.agents, args.agentNum)

    random.seed(0)
    runs = 1 + args.restart

    # Initialize the solver
    if args.lowLevelSolver:
        solver = ECBS(instance, args.sipp, args.screen)
    else:
        raise RuntimeError
        # solver = CBS(instance, args.sipp, args.screen)

    solver.set_prioritize_conflicts(args.prioritizingConflicts)
    solver.set_disjoint_splitting(args.disjointSplitting)
    solver.set_bypass(args.bypass)
    solver.set_rectangle_reasoning(args.rectangleReasoning)
    solver.set_corridor_reasoning(args.corridorReasoning)
    solver.set_heuristic_type(h, h_hat)
    solver.set_target_reasoning(args.targetReasoning)
    solver.set_mutex_reasoning(False)
    solver.set_conflict_selection_rule(conflict)
    solver.set_node_selection_rule(node_selection)
    solver.set_saving_stats(args.stats)
    solver.set_high_level_solver(solver_type, args.suboptimality)

    # Run the solver
    runtime = 0
    lowerbound = 0
    for i in range(runs):
        solver.clear()
        solver.solve(args.cutoffTime / runs, lowerbound)
        runtime += solver.runtime
        if solver.solution_found:
            break
        lowerbound = solver.cost_lowerbound
        solver.random_root = True
        print(f"Failed to find solutions in Run {i}")

    solver.runtime = runtime
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        fname = os.path.join(args.output_dir, args.outputRes)
        print(f'save results into {fname}')
        solver.save_results(fname, args.agents)
    if solver.solution_found and args.outputPaths:
        fname = os.path.join(args.output_dir, args.outputPaths)
        print(f'save paths into {fname}')
        solver.save_paths(fname)
    if args.stats:
        solver.saving_stats(args.output, args.agents)


if __name__ == "__main__":
    main()