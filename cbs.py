# -*- coding:utf-8 -*-
# @FileName  :cbs.py
# @Time      :2024/7/30 下午5:41
# @Author    :ZMFY
# Description:
import os
import random
import sys
import time
from itertools import combinations
import numpy as np
from enum import Enum
from typing import List, Tuple, Union, Dict

from multipledispatch import dispatch

from cbs_heuristic import CSBHeuristic, HeuristicType
from nodes import CBSNode, HLNode, NodeSelection, ECBSNode
from corridor_reasoning import CorridorReasoning
from mutex_reasoning import MutexReasoning
from rectangle_reasoning import RectangleReasoning
from instance import Instance
from single_agent_solver import SingleAgentSolver
from constraint_table import ConstraintTable
import common as cm
from mdd import MDDTable
from space_time_astar import SpaceTimeAstar
from conflict import Conflict, ConflictSelection, ConstraintType, Constraint, ConflictType
from sipp import SIPP


class HighLevelSolverType(Enum):
    ASTAR = 0
    ASTAREPS = 1
    NEW = 2
    EES = 3


class CBS:
    def __init__(self, instance: Instance, sipp: bool = False, screen: int = 2):
        self.screen = screen
        self.sipp = sipp
        self.instance = instance
        self.num_of_agents = instance.num_of_agents
        self.random_root = False

        self.rectangle_reasoning = False  # using rectangle reasoning
        self.corridor_reasoning = False  # using corridor reasoning
        self.target_reasoning = False  # using target reasoning
        self.mutex_reasoning = False  # using mutex reasoning
        self.disjoint_splitting = False  # disjoint splitting
        self.bypass = False  # using Bypass1
        self.pc = False  # prioritize conflicts
        self.saving_stats = False

        self.solver_type = HighLevelSolverType.ASTAR  # the solver for the high-level search
        self.conflict_selection_rule = ConflictSelection.RANDOM
        self.node_selection_rule = NodeSelection.NODE_RANDOM

        # this is ued for both ECBS and EES
        self.all_nodes_table: List[Union[HLNode, CBSNode, ECBSNode]] = []

        st = time.perf_counter()
        self.initial_constraints = [ConstraintTable(instance.num_of_cols, instance.map_size)
                                    for _ in range(self.num_of_agents)]
        if sipp:
            self.search_engines = [SIPP(instance, i) for i in range(self.num_of_agents)]
        else:
            self.search_engines = [SpaceTimeAstar(instance, i) for i in range(self.num_of_agents)]

        self.paths: List[cm.Path] = []
        self.paths_found_initially: List[cm.Path] = []  # contain initial paths found

        self.mutex_helper = MutexReasoning(instance, self.initial_constraints)
        self.rectangle_helper = RectangleReasoning(instance)
        self.mdd_helper = MDDTable(self.initial_constraints, self.search_engines)
        self.corridor_helper = CorridorReasoning(self.search_engines, self.initial_constraints)
        self.heuristic_helper = CSBHeuristic(
            instance.num_of_agents, self.paths, self.search_engines,
            self.initial_constraints, self.mdd_helper
        )

        self.mutex_helper.search_engines = self.search_engines
        if screen >= 2:
            instance.print_agents()

        self.runtime_preprocessing = time.perf_counter() - st  # runtime of building heuristic table for the low level
        self.runtime_generate_child = 0  # runtime of generating child nodes
        self.runtime_build_ct = 0  # runtime of building constraint table
        self.runtime_build_cat = 0  # runtime of building conflict avoidance table
        self.runtime_path_finding = 0  # runtime of finding paths for single agents
        self.runtime_detect_conflicts = 0
        self.runtime = 0
        self.time_limit = np.inf
        self.start_time = 0

        self.sub_optimality = 1.0
        self.cost_lowerbound = 0
        self.inadmissible_cost_lowerbound = 0
        self.node_limit = cm.MAX_NODES
        self.cost_upperbound = cm.MAX_COST

        self.num_cardinal_conflicts = 0
        self.num_cardinal_conflicts = 0
        self.num_corridor_conflicts = 0
        self.num_rectangle_conflicts = 0
        self.num_target_conflicts = 0
        self.num_mutex_conflicts = 0
        self.num_standard_conflicts = 0
        self.num_adopt_bypass = 0  # number of times when adopting bypasses

        self.num_HL_expanded = 0
        self.num_HL_generated = 0
        self.num_LL_expanded = 0
        self.num_LL_generated = 0
        self.num_cleanup = 0  # number of expanded nodes chosen from cleanup list
        self.num_open = 0  # number of expanded nodes chosen from open list
        self.num_focal = 0  # number of expanded nodes chosen from focal list

        self.dummy_start: HLNode | CBSNode | None = None
        self.goal_node: HLNode | CBSNode | None = None

        self.solution_found = False
        self.solution_cost = -2

        self.open_list = cm.PrioritySet()  # it is called open list in ECBS
        self.cleanup_list = cm.PrioritySet()  # this is used for EES
        self.focal_list = cm.PrioritySet()  # this is ued for both ECBS and EES

    @classmethod
    def build_cbs(cls, search_engines: List[SingleAgentSolver], initial_constraints: List[ConstraintTable],
                  paths_found_initially: List[cm.Path], screen=1):
        pass

    def clear(self):
        self.mdd_helper.clear()
        self.heuristic_helper.clear()
        self.release_nodes()
        self.paths.clear()
        self.paths_found_initially.clear()
        self.dummy_start = None
        self.goal_node = None
        self.solution_found = None
        self.solution_cost = -2

    def release_nodes(self):
        self.open_list.clear()
        self.cleanup_list.clear()
        self.focal_list.clear()
        self.all_nodes_table.clear()

    def clear_search_engines(self):
        self.search_engines.clear()

    def get_solver_name(self):
        name = ""
        if self.disjoint_splitting:
            name += "Disjoint "

        if self.heuristic_helper.h_type == HeuristicType.ZERO:
            if self.pc:
                name += "ICBS"
            else:
                name += "CBS"
        elif self.heuristic_helper.h_type == HeuristicType.CG:
            name += "CG"
        elif self.heuristic_helper.h_type == HeuristicType.DG:
            name += "DG"
        elif self.heuristic_helper.h_type == HeuristicType.WDG:
            name += "WDG"

        if self.rectangle_reasoning:
            name += "+R"
        if self.corridor_reasoning:
            name += "+C"
        if self.target_reasoning:
            name += "+T"
        if self.mutex_reasoning:
            name += "+MP"
        if self.bypass:
            name += "+BP"

        name += " with " + self.search_engines[0].get_name()
        return name

    def add_constraints(self, curr: HLNode, child1: HLNode, child2: HLNode):
        if self.disjoint_splitting and curr.conflict.con_type == ConflictType.STANDARD:
            first = bool(random.randint(0, 1))
            if first:  # disjoint splitting on the first agent
                child1.constraints = curr.conflict.constraint1
                a, x, y, t, flag = curr.conflict.constraint1[-1]
                if flag == ConstraintType.VERTEX:
                    child2.constraints.append(Constraint(a, x, y, t, ConstraintType.POSITIVE_VERTEX))
                else:
                    assert flag == ConstraintType.EDGE
                    child2.constraints.append(Constraint(a, x, y, t, ConstraintType.POSITIVE_EDGE))
            else:  # disjoint splitting on the second agent
                child2.constraints = curr.conflict.constraint2
                a, x, y, t, flag = curr.conflict.constraint2[-1]
                if flag == ConstraintType.VERTEX:
                    child1.constraints.append(Constraint(a, x, y, t, ConstraintType.POSITIVE_VERTEX))
                else:
                    assert flag == ConstraintType.EDGE
                    child1.constraints.append(Constraint(a, x, y, t, ConstraintType.POSITIVE_EDGE))
        else:
            child1.constraints = curr.conflict.constraint1
            child2.constraints = curr.conflict.constraint2

    def _find_conflicts(self, curr: HLNode):
        st = time.perf_counter()
        if curr.parent is not None:
            # Copy from parent
            new_agents = curr.get_replanned_agents()
            self._copy_conflicts(curr.parent.conflicts, curr.conflicts, new_agents)
            self._copy_conflicts(curr.parent.unknown_conf, curr.unknown_conf, new_agents)

            # detect new conflicts
            new_agents_set = set(new_agents)  # 转换为集合以加速查找
            for a1 in new_agents:
                for a2 in range(self.num_of_agents):
                    if a1 == a2 or a2 in new_agents_set:
                        continue
                    self._find_conflicts_for_pair(curr, a1, a2)
        else:
            now_agents = [i for i in range(self.num_of_agents)]
            for a1, a2 in combinations(now_agents, 2):
                self._find_conflicts_for_pair(curr, a1, a2)

        self.runtime_detect_conflicts += time.perf_counter() - st

    def _find_conflicts_for_pair(self, curr: HLNode, a1, a2):
        min_path_length = min(len(self.paths[a1]), len(self.paths[a2]))
        for timestep in range(min_path_length):
            loc1 = self.paths[a1][timestep].location
            loc2 = self.paths[a2][timestep].location
            if loc1 == loc2:
                conflict = Conflict()
                if self.target_reasoning and len(self.paths[a1]) == timestep + 1:
                    conflict.target_conflict(a1, a2, loc1, timestep)
                elif self.target_reasoning and len(self.paths[a2]) == timestep + 1:
                    conflict.target_conflict(a2, a1, loc1, timestep)
                else:
                    conflict.vertex_conflict(a1, a2, loc1, timestep)
                assert conflict.constraint1
                assert conflict.constraint2
                curr.unknown_conf.append(conflict)
            elif (timestep < min_path_length - 1 and
                  loc1 == self.paths[a2][timestep + 1].location and
                  loc2 == self.paths[a1][timestep + 1].location):
                conflict = Conflict()
                conflict.edge_conflict(a1, a2, loc1, loc2, timestep + 1)
                assert conflict.constraint1
                assert conflict.constraint2
                curr.unknown_conf.append(conflict)

        if len(self.paths[a1]) != len(self.paths[a2]):
            a1_, a2_ = (a1, a2) if len(self.paths[a1]) < len(self.paths[a2]) else (a2, a1)
            loc1 = self.paths[a1_][-1].location
            for timestep in range(min_path_length, len(self.paths[a2_])):
                loc2 = self.paths[a2_][timestep].location
                if loc1 == loc2:
                    conflict = Conflict()
                    if self.target_reasoning:
                        conflict.target_conflict(a1_, a2_, loc1, timestep)
                    else:
                        conflict.vertex_conflict(a1_, a2_, loc1, timestep)
                    assert conflict.constraint1
                    assert conflict.constraint2
                    curr.unknown_conf = [conflict] + curr.unknown_conf

    def _choose_conflict(self, node: HLNode):
        if self.screen >= 3:
            self.print_conflicts(node)

        if len(node.conflicts) == 0 and len(node.unknown_conf) == 0:
            return None
        if len(node.conflicts) != 0:
            choose = min(node.conflicts)  # min function works because of the __lt__() of Conflict class
        else:
            choose = min(node.unknown_conf)
        return choose

    @staticmethod
    def _copy_conflicts(conflicts: List[Conflict], cpy: List[Conflict], excluded_agent: List[int]):
        for conflict in conflicts:
            # found = False
            # for a in excluded_agent:
            #     if conflict.a1 == a or conflict.a2 == a:
            #         found = True
            #         break
            if conflict.a1 not in excluded_agent and conflict.a2 not in excluded_agent:
                assert not len(conflict.constraint1) == 0
                assert not len(conflict.constraint2) == 0
                cpy.append(conflict)

    def _remove_low_priority_conflicts(self, conflicts: List[Conflict]):
        if not conflicts:
            return

        keep: Dict[int, Conflict] = dict()
        to_delete: List[Conflict] = []
        for conflict in conflicts:
            a1, a2 = sorted([conflict.a1, conflict.a2])
            key = a1 * self.num_of_agents + a2
            p = keep.get(key, None)
            if p is None:
                keep[key] = conflict
            elif p < conflict:
                to_delete.append(p)
                keep[key] = conflict
            else:
                to_delete.append(conflict)

        for conflict in to_delete:
            conflicts.remove(conflict)

    def _get_invalid_agents(self, constraints: List[Constraint]):
        """return agents that violate the constraints"""
        assert len(constraints) != 0
        agent, x, y, t, flag = constraints[0]
        agents = set()

        if flag == ConstraintType.LEQLENGTH:
            assert len(constraints) == 1
            for ag in range(self.num_of_agents):
                if ag == agent:
                    continue
                for i in range(t, len(self.paths[ag])):
                    if self.paths[ag][i].location == x:
                        agents.add(ag)
                        break
        elif flag == ConstraintType.POSITIVE_VERTEX:
            assert len(constraints) == 1
            for ag in range(self.num_of_agents):
                if ag == agent:
                    continue
                if self._get_agent_location(ag, t) == x:
                    agents.add(ag)
        elif flag == ConstraintType.POSITIVE_EDGE:
            assert len(constraints) == 1
            for ag in range(self.num_of_agents):
                if ag == agent:
                    continue
                prev = self._get_agent_location(ag, t - 1)
                curr = self._get_agent_location(ag, t)
                if prev == x or curr == y or (prev == y and curr == x):
                    agents.add(ag)
        else:
            agents.add(agent)

        return agents

    def _compute_second_priority_for_conflict(self, conflict: Conflict, node: HLNode):
        count = [0, 0]
        if self.conflict_selection_rule == ConflictSelection.RANDOM:
            conflict.secondary_priority = 0
        elif self.conflict_selection_rule == ConflictSelection.EARLIEST:
            c = conflict.con_type
            if c == ConflictType.STANDARD or c == ConflictType.RECTANGLE or \
                    c == ConflictType.TARGET or c == ConflictType.MUTEX:
                conflict.secondary_priority = conflict.constraint1[0].t
            elif c == ConflictType.CORRIDOR:
                conflict.secondary_priority = min(conflict.constraint1[0].loc2, conflict.constraint2[0].t)
        elif self.conflict_selection_rule == ConflictSelection.CONFLICTS:
            for c in node.conflicts:
                if c.a1 == conflict.a1 or c.a2 == conflict.a2:
                    count[0] += 1
                if c.a1 == conflict.a2 or c.a2 == conflict.a1:
                    count[1] += 1
            conflict.secondary_priority = sum(count)

    def print_results(self):
        txt = ""
        if self.solution_cost >= 0:
            txt += "Succeed\t"
        elif self.solution_cost == -1:
            txt += "Timeout\t"
        elif self.solution_cost == -2:
            txt += "No solutions\t"
        elif self.solution_cost == -3:
            txt += "Nodes out\t"

        metrics = [self.solution_cost, self.runtime, self.num_HL_expanded, self.num_LL_expanded,
                   self.cost_lowerbound, self.dummy_start.g_val, self.dummy_start.f_val]
        txt += ', '.join((str(m) for m in metrics))
        print(txt)

    @staticmethod
    def print_conflicts(curr: HLNode):
        for conflict in curr.conflicts:
            print(conflict)
        for conflict in curr.unknown_conf:
            print(conflict)

    def save_results(self, fname, instance_name):
        if not os.path.isfile(fname):
            with open(fname, 'w') as f:
                f.write(
                    "runtime,#high-level expanded,#high-level generated,#low-level expanded,#low-level generated," +
                    "solution cost,min f value,root g value, root f value," +
                    "#adopt bypasses," +
                    "cardinal conflicts," +
                    "standard conflicts,rectangle conflicts,corridor conflicts,target conflicts,mutex conflicts," +
                    "chosen from cleanup,chosen from open,chosen from focal," +
                    "#solve MVCs,#merge MDDs,#solve 2 agents,#memoization," +
                    "cost error,distance error," +
                    "runtime of building heuristic graph,runtime of solving MVC," +
                    "runtime of detecting conflicts," +
                    "runtime of rectangle conflicts,runtime of corridor conflicts,runtime of mutex conflicts," +
                    "runtime of building MDDs,runtime of building constraint tables,runtime of building CATs," +
                    "runtime of path finding,runtime of generating child nodes," +
                    "preprocessing runtime,solver name,instance name\n"
                )

                f.write(
                    f"{self.runtime},{self.num_HL_expanded},{self.num_HL_generated},{self.num_LL_expanded},"
                    f"{self.num_LL_generated},{self.solution_cost},{self.cost_lowerbound},{self.dummy_start.g_val},"
                    f"{self.dummy_start.g_val + self.dummy_start.h_val},{self.num_adopt_bypass},"
                    f"{self.num_cardinal_conflicts},{self.num_standard_conflicts},{self.num_rectangle_conflicts},"
                    f"{self.num_corridor_conflicts},{self.num_target_conflicts},{self.num_mutex_conflicts},"
                    f"{self.num_cleanup},{self.num_open},{self.num_focal},{self.heuristic_helper.num_solve_mvc},"
                    f"{self.heuristic_helper.num_merge_mdds},{self.heuristic_helper.num_solve_2agent_problems},"
                    f"{self.heuristic_helper.num_memoization},{self.heuristic_helper.get_cost_error()},"
                    f"{self.heuristic_helper.get_distance_error()},{self.heuristic_helper.runtime_build_dependency_graph},"
                    f"{self.heuristic_helper.runtime_solve_mvc},{self.runtime_detect_conflicts},"
                    f"{self.rectangle_helper.accumulated_runtime},{self.corridor_helper.accumulated_runtime},"
                    f"{self.mutex_helper.accumulated_runtime},{self.mdd_helper.accumulated_runtime},"
                    f"{self.runtime_build_ct},{self.runtime_build_cat},{self.runtime_path_finding},"
                    f"{self.runtime_generate_child},{self.runtime_preprocessing},"
                    f"{self.get_solver_name()},{instance_name}\n"
                )

    def _valid_solution(self) -> bool:
        # Check whether the solution cost is within the bound
        if self.solution_cost > self.cost_lowerbound * self.sub_optimality:
            print("Solution cost exceeds the sub-optimality bound!")
            return False

        # Check whether the paths are feasible
        soc = 0
        for a1 in range(self.num_of_agents):
            soc += len(self.paths[a1]) - 1
            for a2 in range(a1 + 1, self.num_of_agents):
                min_path_length = min(len(self.paths[a1]), len(self.paths[a2]))
                for timestep in range(min_path_length):
                    loc1 = self.paths[a1][timestep].location
                    loc2 = self.paths[a2][timestep].location
                    if loc1 == loc2:
                        print(f"Agents {a1} and {a2} collide at {loc1} at timestep {timestep}")
                        return False
                    elif timestep < min_path_length - 1 \
                            and loc1 == self.paths[a2][timestep + 1].location \
                            and loc2 == self.paths[a1][timestep + 1].location:
                        print(f"Agents {a1} and {a2} collide at ({loc1}-->{loc2}) at timestep {timestep}")
                        return False

                if len(self.paths[a1]) != len(self.paths[a2]):
                    a1_ = a1 if len(self.paths[a1]) < len(self.paths[a2]) else a2
                    a2_ = a2 if len(self.paths[a1]) < len(self.paths[a2]) else a1
                    loc1 = self.paths[a1_][-1].location
                    for timestep in range(min_path_length, len(self.paths[a2_])):
                        loc2 = self.paths[a2_][timestep].location
                        if loc1 == loc2:
                            print(f"Agents {a1} and {a2} collide at {loc1} at timestep {timestep}")
                            return False  # It's at least a semi conflict

        if soc != self.solution_cost:
            print("The solution cost is wrong!")
            return False

        return True

    def _get_agent_location(self, agent_id, timestep) -> int:
        t = max(min(timestep, len(self.paths[agent_id]) - 1), 0)
        return self.paths[agent_id][t].location

    def _terminate(self, curr: HLNode) -> bool:
        """check the stop condition and return true if it meets"""
        if self.cost_lowerbound >= self.cost_upperbound:
            self.solution_cost = self.cost_lowerbound
            self.solution_found = False
            if self.screen > 0:
                self.print_results()
            return True
        self.runtime = time.perf_counter() - self.start_time

        if len(curr.conflicts) == 0 and len(curr.unknown_conf) == 0:
            self.solution_found = True
            self.goal_node = curr
            self.solution_cost = self.goal_node.f_hat_val - self.goal_node.cost_to_go
            if not self._valid_solution():
                print("solution invalid")
                self.print_results()
                self._print_paths()
                sys.exit(-1)
            if self.screen > 0:
                self.print_results()
            return True
        if self.runtime > self.time_limit or self.num_HL_expanded > self.node_limit:
            self.solution_cost = -1
            self.solution_found = False
            if self.screen > 0:
                self.print_results()
            return True

        return False

    def _shuffle_agents(self) -> List[int]:
        """generate random permutation of agent indices"""
        agents = list(range(self.num_of_agents))
        if self.random_root:
            random.shuffle(agents)
        return agents

    def _compute_conflict_priority(self, conflict: Conflict, node: CBSNode):
        """check the conflict is cardinal, semi-cardinal or non-cardinal"""
        raise NotImplemented
        # a1, a2 = conflict.a1, conflict.a2
        # timestep, flag = conflict.constraint1[-1].t, conflict.constraint1[-1].flag
        # cardinal1, cardinal2 = False, False
        # if timestep >= len(self.paths[a1]):
        #     cardinal1 = True
        #     mdd1 = None
        # else:
        #     mdd1 = self.mdd_helper.get_mdd(node, a1, len(self.paths[a1]))
        # if timestep >= len(self.paths[a2]):
        #     cardinal2 = True
        #     mdd2 = None
        # else:
        #     mdd2 = self.mdd_helper.get_mdd(node, a2, len(self.paths[a2]))
        #
        # if flag == ConstraintType.EDGE:  # Edge conflict
        #     cardinal1 = len(mdd1.levels[timestep]) == 1 and len(mdd1.levels[timestep - 1]) == 1
        #     cardinal2 = len(mdd2.levels[timestep]) == 1 and len(mdd2.levels[timestep - 1]) == 1
        # else:  # vertex conflict or target conflict
        #     if not cardinal1:
        #         cardinal1 = len(mdd1.levels[timestep]) == 1
        #     if not cardinal2:
        #         cardinal2 = len(mdd2.levels[timestep]) == 1
        #
        # if cardinal1 and cardinal2:
        #     conflict.priority = ConflictPriority.CARDINAL
        # elif cardinal1 or cardinal2:
        #     conflict.priority = ConflictPriority.SEMI
        # else:
        #     conflict.priority = ConflictPriority.NON

    def set_heuristic_type(self, h: HeuristicType, h_hat: HeuristicType):
        self.heuristic_helper.h_type = h
        self.heuristic_helper.set_inadmissible_heuristics(h_hat)

    def set_prioritize_conflicts(self, p: bool):
        self.pc = p
        self.heuristic_helper.pc = p

    def set_rectangle_reasoning(self, r: bool):
        self.rectangle_reasoning = r
        self.heuristic_helper.rectangle_reasoning = r

    def set_corridor_reasoning(self, c: bool):
        self.corridor_reasoning = c
        self.heuristic_helper.corridor_reasoning = c

    def set_target_reasoning(self, t: bool):
        self.target_reasoning = t
        self.heuristic_helper.target_reasoning = t

    def set_mutex_reasoning(self, m: bool):
        self.mutex_reasoning = m
        self.heuristic_helper.mutex_reasoning = m

    def set_disjoint_splitting(self, d: bool):
        self.disjoint_splitting = d
        self.heuristic_helper.disjoint_splitting = d

    def set_bypass(self, b: bool):
        self.bypass = b

    def set_conflict_selection_rule(self, c: ConflictSelection):
        self.conflict_selection_rule = c
        self.heuristic_helper.conflict_selection_rule = c

    def set_node_selection_rule(self, n: NodeSelection):
        self.node_selection_rule = n
        self.heuristic_helper.node_selection_rule = n

    def set_saving_stats(self, s: bool):
        self.saving_stats = s
        self.heuristic_helper.saving_stats = s

    def set_high_level_solver(self, s: HighLevelSolverType, w: float):
        self.solver_type = s
        self.sub_optimality = w

    def solve(self, time_limit, cost_lowerbond=0, cost_upperbond=cm.MAX_COST) -> bool:
        pass

    def save_stats(self, fname, instance_name):
        pass

    def save_ct(self, fname):
        pass

    def save_paths(self, fname):
        with open(fname, 'w') as f:
            line = ""
            for i in range(self.num_of_agents):
                line += f"Agent {i}"
                for t in self.paths[i]:
                    line += (f"({self.search_engines[0].instance.get_row_coordinate(t.location)}, "
                             f"{self.search_engines[0].instance.get_col_coordinate(t.location)}) -> ")
            f.write(line + '\n')

    def _push_node(self, node: CBSNode | ECBSNode):
        raise NotImplemented

    def _select_node(self):
        raise NotImplemented

    def _reinsert_node(self, node: CBSNode | ECBSNode) -> bool:
        raise NotImplemented

    def _generate_child(self, child: CBSNode | ECBSNode, curr: CBSNode | ECBSNode) -> bool:
        raise NotImplemented

    def _generate_root(self) -> bool:
        raise NotImplemented

    def _find_path_for_single_agent(self, node: CBSNode | ECBSNode, agent, lower_bound=0) -> bool:
        raise NotImplemented

    def _classify_conflicts(self, parent: CBSNode | ECBSNode):
        raise NotImplemented

    def _update_paths(self, curr: CBSNode | ECBSNode):
        raise NotImplemented

    def _print_paths(self):
        raise NotImplemented


if __name__ == "__main__":
    pass
