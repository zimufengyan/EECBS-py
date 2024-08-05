# -*- coding:utf-8 -*-
# @FileName  :cbs_heuristic.py
# @Time      :2024/7/29 下午4:05
# @Author    :ZMFY
# Description:
import sys
import time
from collections import deque
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from enum import Enum
from multipledispatch import dispatch
import numpy as np
import numba

from rectangle_reasoning import RectangleReasoning
from corridor_reasoning import CorridorReasoning
from mdd import MDD
from single_agent_solver import SingleAgentSolver
import common as cm
from nodes import HLNode, NodeSelection, CBSNode, ECBSNode, SyncMDDNode
from conflict import ConstraintType, ConflictSelection, ConflictPriority, ConflictType, Constraint
from constraint_table import ConstraintTable
from mdd import MDDTable, SyncMDD


class HeuristicType(Enum):
    ZERO = 0
    CG = 1
    DG = 2
    WDG = 3
    GLOBAL = 4
    PATH = 5
    LOCAL = 6
    CONFLICT = 7
    STRATEGY_COUNT = 8


class HTableEntry:
    def __init__(self, a1: int, a2: int, n: HLNode):
        self.a1 = a1
        self.a2 = a2
        self.n = n

    @staticmethod
    def _get_constrains(entry) -> Tuple[set, set]:
        cons1, cons2 = set(), set()
        curr = entry.n
        while curr.parent is not None:
            if curr.constraints[0].flag == ConstraintType.LEQLENGTH or \
                    curr.constraints[1].flag == ConstraintType.POSITIVE_VERTEX or \
                    curr.constraints[0].flag == ConstraintType.POSITIVE_EDGE:
                for con in curr.constraints:
                    cons1.add(con)
                    cons2.add(con)
            else:
                if curr.constraints[0].agent == entry.a1:
                    for con in curr.constraints:
                        cons1.add(con)
                elif curr.constraints[0].agent == entry.a2:
                    for con in curr.constraints:
                        cons2.add(con)
            curr = curr.parent
        return cons1, cons2

    def __eq__(self, other):
        cons1 = [self._get_constrains(self), self._get_constrains(other)]
        cons2 = [self._get_constrains(self), self._get_constrains(other)]
        if len(cons1[0]) != len(cons1[1]) or len(cons2[0]) != len(cons2[1]):
            return False

        if not all([a == b for a, b in zip(cons1[0], cons1[1])]):
            return False
        return all([a == b for a, b in zip(cons2[0], cons2[1])])

    def __hash__(self):
        return hash(self.get_hash_key())

    def get_hash_key(self):
        curr = self.n
        cons1_hash, cons2_hash = 0, 0
        while curr.parent is not None:
            if curr.constraints[0].agent == self.a2 or \
                    curr.constraints[0].flag == ConstraintType.LEQLENGTH or \
                    curr.constraints[1].flag == ConstraintType.POSITIVE_VERTEX or \
                    curr.constraints[0].flag == ConstraintType.POSITIVE_EDGE:
                for con in curr.constraints:
                    cons1_hash += (3 * con.agent + 5 * con.loc1 + 7 * con.loc2 + 11 * con.t)
            elif curr.constraints[0].agent == self.a2 or \
                    curr.constraints[0].flag == ConstraintType.LEQLENGTH or \
                    curr.constraints[1].flag == ConstraintType.POSITIVE_VERTEX or \
                    curr.constraints[0].flag == ConstraintType.POSITIVE_EDGE:
                for con in curr.constraints:
                    cons2_hash += (3 * con.agent + 5 * con.loc1 + 7 * con.loc2 + 11 * con.t)

            curr = curr.parent

        return cons1_hash ^ (cons2_hash << 1)


HTable = Dict[HTableEntry, Tuple[int, int, int]]


class CSBHeuristic:
    def __init__(self, num_of_agents, paths: List[cm.Path], search_engines: List[SingleAgentSolver],
                 initial_constraints: List[ConstraintTable], mdd_helper: MDDTable):
        self.num_of_agents = num_of_agents
        self.paths = paths
        self.search_engines = search_engines
        self.initial_constraints = initial_constraints
        self.mdd_helper = mdd_helper

        # list for <agent 1, agent 2, node, number of expanded CT nodes, h value>
        self.sub_instances: List[Tuple[int, int, HLNode, int, int]] = []
        self.h_type: HeuristicType = HeuristicType.GLOBAL
        self.rectangle_reasoning = False  # using rectangle reasoning
        self.corridor_reasoning = False  # using corridor reasoning
        self.target_reasoning = False  # using target reasoning
        self.mutex_reasoning = False  # using mutex reasoning
        self.disjoint_splitting = False  # disjoint splitting
        self.conflicts_prioritize = False  # prioritize conflicts

        self.save_stats = False
        self.conflict_selection_rule: ConflictSelection = ConflictSelection.RANDOM
        self.node_selection_rule: NodeSelection = NodeSelection.NODE_RANDOM

        self.runtime_build_dependency_graph = 0
        self.runtime_solve_mvc = 0
        self.num_solve_mvc = 0
        self.num_merge_mdds = 0
        self.num_solve_2agent_problems = 0
        self.num_memoization = 0  # number of times when memeorization helps

        self.inadmissible_heuristic: HeuristicType = HeuristicType.GLOBAL
        self.screen = 0
        self.lookup_table: List[List[HTable]] = []
        self.sum_distance_errors: np.ndarray | None = None
        self.sum_cost_errors: np.ndarray | None = None
        self.num_of_errors: np.ndarray | None = None

        self.time_limit = np.inf
        self.node_limit = 4  # terminate the sub CBS solver if the number of its expanded nodes exceeds the node limit.
        self.start_time = 0
        self.ilp_node_threshold = 5  # when #nodes >= ilp_node_threshold, use ILP solver; otherwise, use DP solver
        self.ilp_edge_threshold = 10  # when #edges >= ilp_edge_threshold, use ILP solver; otherwise, use DP solver
        self.ilp_value_threshold = 32  # when value >= ilp_value_threshold, use ILP solver; otherwise, use DP solver

    def init(self):
        if self.h_type == HeuristicType.DG or self.h_type == HeuristicType.WDG:
            self.lookup_table = [[dict() for _ in range(self.num_of_agents)] for _ in range(self.num_of_agents)]

    def clear(self):
        self.lookup_table.clear()

    def set_inadmissible_heuristics(self, h: HeuristicType):
        self.inadmissible_heuristic = h
        if h == HeuristicType.CONFLICT:
            self.sum_distance_errors = np.zeros(ConflictType.TYPE_COUNT.value)
            self.sum_cost_errors = np.zeros(ConflictType.TYPE_COUNT.value)
            self.num_of_errors = np.ones(ConflictType.TYPE_COUNT.value)
        else:
            self.sum_distance_errors = np.zeros(1)
            self.sum_cost_errors = np.zeros(1)
            self.num_of_errors = np.zeros(1)

    def compute_cbs_informed_heuristics(self, curr: CBSNode, time_limit) -> bool:
        curr.h_computed = True
        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        hg = np.zeros(int(self.num_of_agents * self.num_of_agents), dtype=bool)  # heuristic graph
        h, num_of_cg_edges = -1, 0

        if self.h_type == HeuristicType.ZERO:
            h = 0
        elif self.h_type == HeuristicType.CG:
            num_of_cg_edges = self._build_cardinal_conflict_graph(curr, hg)
            if curr.parent is None or num_of_cg_edges > self.ilp_edge_threshold or \
                    self.target_reasoning or self.disjoint_splitting:
                # root node of CBS tree or the graph is too large
                # when we are allowed to replan for multiple agents, the incremental method is not correct any longer.
                h = self._minimum_vertex_cover(hg)
            else:
                h = self._minimum_vertex_cover(hg, curr.parent.h_val, self.num_of_agents, num_of_cg_edges)
        elif self.h_type == HeuristicType.DG:
            num_of_cg_edges, succ = self._build_dependence_graph(curr, hg)
            if not succ:
                return False
            # minimum vertex cover
            if curr.parent is None or num_of_cg_edges > self.ilp_edge_threshold or \
                    self.target_reasoning or self.disjoint_splitting:
                h = self._minimum_vertex_cover(hg)
            else:
                h = self._minimum_vertex_cover(hg, curr.parent.h_val, self.num_of_agents, num_of_cg_edges)
        elif self.h_type == HeuristicType.WDG:
            raise NotImplementedError
            # if not self._build_weighted_dependency_graph(curr, hg):
            #     return False
            # h = self._minimum_weighted_vertex_cover(hg)

        if h < 0:
            return False

        curr.h_val = max(h, curr.h_val)
        return True

    def compute_ecbs_informed_heuristics(self, curr: ECBSNode, min_f_vals: List[int], time_limit):
        curr.h_computed = True
        self.start_time = time.perf_counter()
        self.time_limit = time_limit
        h, num_of_cg_edges = -1, 0
        hg = np.zeros(int(self.num_of_agents * self.num_of_agents), dtype=int)  # heuristic graph

        if self.h_type == HeuristicType.ZERO:
            h = 0
        elif self.h_type == HeuristicType.WDG:
            raise NotImplementedError
            # delta_g, succ = self._build_ecbs_weighted_dependency_graph(curr, min_f_vals, hg)
            # if not succ:
            #     return False
            # assert delta_g >= 0
            # h = self._minimum_weighted_vertex_cover(hg) + delta_g
        else:
            print("ERROR in computing informed heurisctis")

        if h < 0:
            return False
        curr.h_val = max(h, curr.h_val)
        curr.cost_to_go = max(curr.cost_to_go, curr.f_val - curr.sum_of_costs)  # ensure that f <= f^
        return True

    def compute_quick_heuristics(self, node: HLNode):
        if node.parent is not None:
            node.h_val = max(0, node.parent.f_val - node.g_val)

        node.update_distance_to_go()
        if node.parent is not None:
            self.update_inadmissible_heuristics(node)  # compute inadmissible heuristics

    def _update_online_heuristics(self, node: CBSNode | ECBSNode):
        if self.inadmissible_heuristic == HeuristicType.GLOBAL:
            self.sum_distance_errors[0] += node.distance_error
            self.sum_cost_errors[0] += node.cost_error
            self.num_of_errors[0] += 1
        elif self.inadmissible_heuristic == HeuristicType.LOCAL:
            self.num_of_errors[0] += 1
            learning_rate = 1e-3
            if self.num_of_errors[0] * learning_rate < 1:
                learning_rate = 1. / self.num_of_errors[0]
            self.sum_distance_errors[0] = (self.sum_cost_errors[0] * (1 - learning_rate) +
                                           node.distance_error * learning_rate)
            self.sum_cost_errors[0] = (self.sum_cost_errors[0] * (1 - learning_rate) +
                                       node.cost_error * learning_rate)
        elif self.inadmissible_heuristic == HeuristicType.CONFLICT:
            idx = node.conflict.get_conflict_id()
            self.sum_distance_errors[idx] += node.distance_error
            self.sum_cost_errors[idx] += node.cost_error
            self.num_of_errors[idx] += 1

    def update_cbs_online_heuristic_errors(self, curr: CBSNode):
        if (self.inadmissible_heuristic == HeuristicType.GLOBAL or
            self.inadmissible_heuristic == HeuristicType.PATH or
            self.inadmissible_heuristic == HeuristicType.LOCAL or
            self.inadmissible_heuristic == HeuristicType.CONFLICT) \
                and curr.parent is not None:
            curr.parent.fully_expanded = True
            best = curr
            for child in curr.parent.children:
                if not child.h_computed:
                    curr.parent.fully_expanded = False
                    break
                if best.f_val > child.f_val or \
                        (best.f_val == child.f_val and best.distance_to_go > child.distance_to_go):
                    best = child
            if curr.parent.fully_expanded:  # update error
                curr.parent.distance_error = 1 + best.distance_to_go - curr.parent.distance_to_go
                curr.parent.cost_error = best.f_val - curr.f_val
                self._update_online_heuristics(curr.parent)

    def update_ecbs_online_heuristic_errors(self, parent: ECBSNode):
        if not self.inadmissible_heuristic == HeuristicType.ZERO:
            return
        best_child: ECBSNode = parent.children[0]
        assert len(parent.children) <= 2
        if len(parent.children) == 2:
            other = parent.children[-1]
            if best_child.f_hat_val > other.f_hat_val or \
                    (best_child.f_hat_val == other.f_hat_val and best_child.distance_to_go > other.distance_to_go):
                best_child = other

        # update the errors
        parent.distance_error = 1 + best_child.distance_to_go - parent.distance_to_go
        parent.cost_error = best_child.f_hat_val - best_child.cost_to_go - parent.sum_of_costs
        self._update_online_heuristics(parent)

    def update_inadmissible_heuristics(self, curr: HLNode | CBSNode):
        h = curr.h_val if curr.get_name() == "CBS Node" else 0
        cost_error = 0
        distance_error = 0
        if self.inadmissible_heuristic == HeuristicType.PATH:
            # update errors along the path to the node
            self.num_of_errors[0] = 0
            self.sum_distance_errors[0] = 0
            self.sum_cost_errors[0] = 0
            ptr = curr.parent
            while ptr is not None:
                if ptr.fully_expanded:
                    self.num_of_errors[0] += 1
                    self.sum_distance_errors[0] += ptr.distance_error
                    self.sum_cost_errors[0] += ptr.cost_error
                ptr = ptr.parent
        if self.inadmissible_heuristic == HeuristicType.PATH or self.inadmissible_heuristic == HeuristicType.GLOBAL:
            # Note: here we compute heuristics for both GLOBAL and PATH
            if self.num_of_errors[0] < 1:
                curr.cost_to_go = max(0, curr.f_val - curr.f_hat_val)
                return

            if self.num_of_errors[0] <= self.sum_distance_errors[0]:
                c = self.sum_cost_errors[0] / self.num_of_errors[0] * 10
            else:
                c = self.sum_cost_errors[0] / (self.num_of_errors[0] - self.sum_distance_errors[0])

            curr.cost_to_go = h + int(curr.distance_to_go * c)
        if self.inadmissible_heuristic == HeuristicType.LOCAL:
            if abs(1 - self.sum_distance_errors[0]) < 0.001:
                curr.cost_to_go = h + max(0, int(curr.distance_to_go) * 1000)
            else:
                curr.cost_to_go = h + max(0, int(curr.distance_to_go * self.sum_cost_errors[0] / (
                            1 - self.sum_distance_errors[0])))
        if self.inadmissible_heuristic == HeuristicType.CONFLICT:
            if not curr.conflicts and not curr.unknown_conf:
                return

            for conflict in curr.conflicts:
                conflict_id = conflict.get_conflict_id()
                cost_error += self.get_cost_error(conflict_id)
                distance_error += self.get_distance_error(conflict_id)

            cost_error /= len(curr.conflicts)
            distance_error /= len(curr.conflicts)

            if distance_error >= 1:
                curr.cost_to_go = int(curr.distance_to_go * cost_error)
            else:
                curr.cost_to_go = int(curr.distance_to_go * cost_error / (1 - distance_error))

            curr.cost_to_go = max(min(cm.MAX_COST, curr.cost_to_go), 0)
            curr.cost_to_go += h

        if curr.f_val > curr.f_hat_val:
            curr.cost_to_go += curr.f_val - curr.f_hat_val

    def get_cost_error(self, i=0) -> float:
        return self.sum_cost_errors[i] / self.num_of_errors[i] if self.num_of_errors[i] != 0 else 0

    def get_distance_error(self, i=0) -> float:
        return self.sum_distance_errors[i] / self.num_of_errors[i] if self.num_of_errors[i] != 0 else 0

    def _build_conflict_graph(self, hg: List[bool], curr: HLNode):
        pass

    def _build_cardinal_conflict_graph(self, curr: CBSNode | ECBSNode, cg: np.ndarray) -> int:
        num_of_cg_edges = 0
        for con in curr.conflicts:
            if con.priority == ConflictPriority.CARDINAL:
                a1, a2 = con.a1, con.a2
                if not cg[a1 * self.num_of_agents + a2]:
                    cg[int(a1 * self.num_of_agents + a2)] = True
                    cg[int(a2 * self.num_of_agents + a1)] = True
                    num_of_cg_edges += 1

        self.runtime_build_dependency_graph += time.perf_counter() - self.start_time
        return num_of_cg_edges

    def _build_dependence_graph(self, node: CBSNode, cg: np.ndarray) -> Tuple[int, bool]:
        num_of_cg_edges = 0
        for i, con in enumerate(node.conflicts):
            a1 = min(con.a1, con.a2)
            a2 = max(con.a1, con.a2)
            idx = int(a1 * self.num_of_agents + a2)
            if cg[idx]:
                continue
            if con.priority == ConflictPriority.CARDINAL:
                cg[int(a1 * self.num_of_agents + a2)] = True
                cg[int(a2 * self.num_of_agents + a1)] = True
                num_of_cg_edges += 1
                continue
            got = self.lookup_table[a1][a2].get(HTableEntry(a1, a2, node), None)
            if got is not None:  # check the lookup table first
                cg[a2 * self.num_of_agents + a1] = cg[idx] = 1 if got[0] > 0 else 0
                num_of_cg_edges += 1
            else:
                cg[a2 * self.num_of_agents + a1] = cg[idx] = self._dependent(a1, a2, node)
                num_of_cg_edges += 1
            if cg[idx]:
                num_of_cg_edges += 1
                # the two agents are dependent, although resolving this conflict might not increase the cost
                node.conflicts[i].priority = ConflictPriority.PSEUDO_CARDINAL

        self.runtime_build_dependency_graph += time.perf_counter() - self.start_time
        return num_of_cg_edges, True

    def _build_cbs_weighted_dependency_graph(self, node: CBSNode, cg: np.ndarray) -> bool:
        for i, con in enumerate(node.conflicts):
            a1 = min(con.a1, con.a2)
            a2 = max(con.a1, con.a2)
            idx1 = int(a1 * self.num_of_agents + a2)
            idx2 = int(a2 * self.num_of_agents + a1)
            got = self.lookup_table[a1][a2].get(HTableEntry(a1, a2, node), None)
            if got is not None:
                self.num_memoization += 1
                cg[idx2] = cg[idx1] = got[0]
            elif self.rectangle_reasoning:
                rst = self._solve_2agents(a1, a2, node, False)
                assert rst[0] >= 0
                self.lookup_table[a1][a2][HTableEntry(a1, a2, node)] = (rst[0], rst[1], 1)
                cg[idx2] = cg[idx1] = rst[0]
            else:
                cardinal = con.priority == ConflictPriority.CARDINAL
                if not cardinal and not self.mutex_reasoning:
                    # using merging MDD methods before runing 2-agent instance
                    cardinal = self._dependent(a1, a2, node)
                if cardinal:
                    # run 2-agent solver only for dependent agents
                    rst = self._solve_2agents(a1, a2, node, cardinal)
                    assert rst[0] >= 1
                    self.lookup_table[a1][a2][HTableEntry(a1, a2, node)] = (rst[0], rst[1], 1)
                else:
                    self.lookup_table[a1][a2][HTableEntry(a1, a2, node)] = (0, 1, 0)
                    cg[idx2] = cg[idx1] = 0
            if time.perf_counter() - self.start_time > self.time_limit:  # run out of time
                self.runtime_build_dependency_graph += time.perf_counter() - self.start_time
                return False
            if cg[idx1] == cm.MAX_COST:  # no solution
                return False
            if con.priority == ConflictPriority.CARDINAL and cg[idx1] > 0:
                # the two agents are dependent, although resolving this conflict might not increase the cost
                node.conflicts[i].priority = ConflictPriority.PSEUDO_CARDINAL

        self.runtime_build_dependency_graph += time.perf_counter() - self.start_time
        return True

    def _build_ecbs_weighted_dependency_graph(self, node: ECBSNode, min_f_vals: List[int],
                                              cg: np.ndarray) -> Tuple[int, bool]:

        def check_conflicts(conflicts):
            delta_g = 0
            counted = np.zeros(self.num_of_agents, dtype=bool)
            for i, con in enumerate(conflicts):
                a1 = min(con.a1, con.a2)
                a2 = max(con.a1, con.a2)
                idx1 = int(a1 * self.num_of_agents + a2)
                idx2 = int(a2 * self.num_of_agents + a1)
                got = self.lookup_table[a1][a2].get(HTableEntry(a1, a2, node), None)
                if got is not None:
                    self.num_memoization += 1
                    cg[idx2] = cg[idx1] = got[0]
                    if not counted[a1]:
                        assert got[1] >= min_f_vals[a1]
                        delta_g += got[1] - min_f_vals[a1]
                        counted[a1] = True
                    if not counted[a2]:
                        assert got[2] >= min_f_vals[a2]
                        delta_g += got[2] - min_f_vals[a2]
                        counted[a2] = True
                else:
                    rst = self._solve_2agents(a1, a2, node)
                    self.lookup_table[a1][a2][HTableEntry(a1, a2, node)] = rst
                    if time.perf_counter() - self.start_time > self.time_limit:
                        self.runtime_build_dependency_graph += time.perf_counter() - self.start_time
                        return delta_g, False, idx1
                    cg[idx2] = cg[idx1] = rst[0]
                    if not counted[a1]:
                        assert rst[1] >= min_f_vals[a1]
                        delta_g += rst[1] - min_f_vals[a1]
                        counted[a1] = True
                    if not counted[a2]:
                        assert rst[2] >= min_f_vals[a2]
                        delta_g += rst[2] - min_f_vals[a2]
                        counted[a2] = True
                return delta_g, True, idx1

        delta_g, success, idx = check_conflicts(node.conflicts)

        if cg[idx] == cm.MAX_COST:  # no solution
            return delta_g, False

        delta_g, success, idx = check_conflicts(node.unknown_conf)

        if cg[idx] == cm.MAX_COST:  # no solution
            return delta_g, False

        self.runtime_build_dependency_graph += time.perf_counter() - self.start_time
        return delta_g, success

    def _dependent(self, a1, a2, node: HLNode) -> bool:
        """return true if the two agents are dependent"""
        mdd1 = self.mdd_helper.get_mdd(node, a1, len(self.paths[a1]))
        mdd2 = self.mdd_helper.get_mdd(node, a2, len(self.paths[a2]))
        if len(mdd1.levels) > len(mdd2.levels):  # swap
            mdd1, mdd2 = mdd2, mdd1
        self.num_merge_mdds += 1
        return not self._sync_mdds(mdd1, mdd2)

    @dispatch(int, int, CBSNode, bool)
    def _solve_2agents(self, a1, a2, node: CBSNode, cradinal: bool) -> Tuple[int, int]:
        """return h value and num of CT nodes"""
        pass

    @dispatch(int, int, ECBSNode)
    def _solve_2agents(self, a1, a2, node: ECBSNode) -> Tuple[int, int, int]:
        """return h value and num of CT nodes"""
        # TODO need to written in CBS rather than here
        pass

    @staticmethod
    def _sync_mdds(mdd: MDD, other: MDD) -> bool:
        """Match and prune MDD according to another MDD."""
        if len(other.levels) <= 1:  # Either of the MDDs was already completely pruned already
            return False

        cpy = SyncMDD(mdd)
        if len(cpy.levels) < len(other.levels):
            i = len(cpy.levels)
            cpy.levels += [[] for _ in range(len(other.levels) - i)]
            while i < len(cpy.levels):
                parent = cpy.levels[i - 1][0]
                node = SyncMDDNode(parent.location, parent)
                parent.children.append(node)
                cpy.levels[i].append(node)

        # Cheaply find the coexisting nodes on level zero - all nodes coexist
        # because agent starting points never collide
        cpy.levels[0][0].coexisting_nodes_from_other_mdds.append(other.levels[0][0])

        for i in range(1, len(cpy.levels)):
            node_iter = iter(cpy.levels[i])
            while True:
                try:
                    node = next(node_iter)
                except StopIteration:
                    break

                for parent in node.parents:
                    for parent_coexisting_node in parent.coexisting_nodes_from_other_mdds:
                        for child_of_parent_coexisting_node in parent_coexisting_node.children:
                            if (node.location == child_of_parent_coexisting_node.location or
                                    (node.location == parent_coexisting_node.location and
                                     parent.location == child_of_parent_coexisting_node.location)):
                                continue

                            if child_of_parent_coexisting_node not in node.coexisting_nodes_from_other_mdds:
                                node.coexisting_nodes_from_other_mdds.append(child_of_parent_coexisting_node)

                if not node.coexisting_nodes_from_other_mdds:
                    cpy.delete_node(node, i)
                else:
                    node_iter.__next__()

            if not cpy.levels[i]:
                cpy.clear()
                return False

        cpy.clear()
        return True

    @dispatch(list)
    def _minimum_vertex_cover(self, cg: np.ndarray) -> int:
        """mvc on disjoint components"""
        start_time = time.perf_counter()
        rst = 0
        done = np.zeros(self.num_of_agents, dtype=bool)

        for i in range(self.num_of_agents):
            if done[i]:
                continue

            indices = []
            q = deque()
            q.append(i)
            done[i] = True

            while not q:
                j = q.popleft()
                indices.append(j)
                for k in range(self.num_of_agents):
                    if cg[j * self.num_of_agents + k] or cg[k * self.num_of_agents + j]:
                        if not done[k]:
                            q.append(k)
                            done[k] = True

            if len(indices) == 1:
                continue
            elif len(indices) == 2:
                rst += 1
                continue

            subgraph = np.zeros((len(indices), len(indices)), dtype=int)
            num_edges = 0

            for j in range(len(indices)):
                for k in range(j + 1, len(indices)):
                    subgraph[j, k] = int(cg[indices[j] * self.num_of_agents + indices[k]])
                    subgraph[k, j] = int(cg[indices[k] * self.num_of_agents + indices[j]])
                    if subgraph[j, k] > 0:
                        num_edges += 1

            if num_edges > self.ilp_edge_threshold:
                rst += self._greedy_matching(subgraph, len(indices))
                if time.perf_counter() - start_time > self.time_limit:
                    return -1  # run out of time
            else:
                for k in range(1, len(indices)):
                    if self._k_vertex_cover(subgraph, len(indices), num_edges, k, len(indices)):
                        rst += k
                        break
                    if time.perf_counter() - start_time > self.time_limit:
                        return -1  # run out of time

        return rst

    @dispatch(np.ndarray, int, int, int)
    def _minimum_vertex_cover(self, cg: np.ndarray, old_mvc: int, cols: int, num_of_edges: int) -> int:
        st = time.perf_counter()
        rst = 0
        if num_of_edges < 2:
            return num_of_edges

        # Compute #CG nodes that have edges
        rows = len(cg) // cols
        triu_mask = np.triu(np.ones((rows, cols), dtype=bool), k=1)  # 上三角掩码(不含对角线)
        num_of_cg_nodes = np.sum(((cg.reshape(-1, cols).astype(int) > 0) & triu_mask).flatten())

        if old_mvc == -1:
            for i in range(1, num_of_cg_nodes):
                if self._k_vertex_cover(cg, num_of_cg_nodes, num_of_edges, i, cols):
                    rst = i
                    break
        else:
            if self._k_vertex_cover(cg, num_of_cg_nodes, num_of_edges, old_mvc - 1, cols):
                rst = old_mvc - 1
            elif self._k_vertex_cover(cg, num_of_cg_nodes, num_of_edges, old_mvc, cols):
                rst = old_mvc
            else:
                rst = old_mvc + 1
        self.num_solve_mvc += 1
        self.runtime_solve_mvc += time.perf_counter() - st
        return rst

    def _k_vertex_cover(self, cg: np.ndarray, num_of_cg_nodes, num_of_cg_edges, k, cols) -> bool:
        """Whether there exists a k-vertex cover solution"""
        runtime = time.perf_counter() - self.start_time
        if runtime > self.time_limit:
            return True
        if num_of_cg_edges == 0:
            return True
        elif num_of_cg_edges > k * num_of_cg_nodes - k:
            return True

        rows = len(cg) // cols
        triu_mask = np.triu(np.ones((rows, cols), dtype=bool), k=1)  # 上三角掩码(不含对角线)
        indices = np.where((cg.reshape(-1, cols) > 0) & triu_mask)
        node = [indices[0][0], indices[0][1]] if indices[0].size > 0 else [0, 0]

        for i in range(2):
            cg_copy = cg.copy()
            num_of_cg_edges_copy = num_of_cg_edges
            for j in range(cols):
                if cg_copy[node[i] * cols + j] > 0:
                    cg_copy[node[i] * cols + j] = 0
                    cg_copy[j * cols + node[i]] = 0
                    num_of_cg_edges_copy -= 1
            if self._k_vertex_cover(cg_copy, num_of_cg_nodes - 1, num_of_cg_edges_copy, k - 1, cols):
                return True

        return False

    @staticmethod
    @numba.jit(nopython=True)
    def _greedy_matching(cg: np.ndarray, cols) -> int:
        rst = 0
        used = np.zeros(cols, dtype=bool)
        for i in range(cols):
            if used[i]:
                continue
            for j in range(i + 1, cols):
                if used[j]:
                    continue
                if cg[i * cols + j] > 0:
                    rst += 1
                    used[i] = used[j] = True
                    break

        return rst

    @staticmethod
    @numba.jit(nopython=True)
    def _greedy_weighted_matching(cg: np.ndarray, cols) -> int:
        rst = 0
        used = np.zeros(cols, dtype=bool)
        while True:
            max_weight, ep1, ep2 = 0, 0, 0
            for i in range(cols):
                if used[i]:
                    continue
                for j in range(i + 1, cols):
                    if used[j]:
                        continue
                    elif max_weight < cg[i * cols + j]:
                        max_weight = cg[i * cols + j]
                        ep1, ep2 = i, j
            if max_weight == 0:
                return rst
            rst += max_weight
            used[ep1] = True
            used[ep2] = True

    def _minimum_weighted_vertex_cover(self, hg: np.ndarray) -> int:
        st = time.perf_counter()
        rst = self._weighted_vertex_cover(hg)
        self.num_solve_mvc += 1
        self.runtime_solve_mvc += time.perf_counter() - st
        return rst

    def _weighted_vertex_cover(self, cg: np.ndarray) -> int:
        start_time = time.perf_counter()
        rst = 0
        done = np.zeros(self.num_of_agents, dtype=bool)

        for i in range(self.num_of_agents):
            if done[i]:
                continue

            range_list = []
            indices = []
            q = deque()
            q.append(i)
            done[i] = True

            while not q:
                j = q.popleft()
                range_list.append(0)
                indices.append(j)
                for k in range(self.num_of_agents):
                    if cg[j * self.num_of_agents + k] > 0:
                        range_list[-1] = max(range_list[-1], cg[j * self.num_of_agents + k].item())
                        if not done[k]:
                            q.append(k)
                            done[k] = True
                    elif cg[k * self.num_of_agents + j] > 0:
                        range_list[-1] = max(range_list[-1], cg[k * self.num_of_agents + j].item())
                        if not done[k]:
                            q.append(k)
                            done[k] = True

            num = len(indices)

            if num == 1:
                continue
            elif num == 2:
                rst += max(cg[indices[0] * self.num_of_agents + indices[1]],
                           cg[indices[1] * self.num_of_agents + indices[0]])
                continue

            g = np.zeros((num, num), dtype=int)
            for j in range(num):
                for k in range(j + 1, num):
                    g[j, k] = max(cg[indices[j] * self.num_of_agents + indices[k]],
                                  cg[indices[k] * self.num_of_agents + indices[j]])

            if num > self.ilp_node_threshold:
                rst += self._greedy_weighted_matching(g, num)
            else:
                x = np.zeros(num, dtype=int)
                ret, _ = self._dp_for_wmvc(x, 0, 0, g, range_list, cm.MAX_COST)
                rst += ret

            runtime = time.time() - start_time
            if runtime > self.time_limit:
                return -1  # run out of time

        return rst

    def _dp_for_wmvc(self, x: np.ndarray, i, total: int, cg: np.ndarray, rang: List[int],
                     best_so_far: int) -> Tuple[int, int]:
        if total >= best_so_far:
            return cm.MAX_COST, best_so_far
        runtime = time.perf_counter() - self.start_time
        if runtime > self.time_limit:
            return -1, best_so_far
        elif i == len(x):
            best_so_far = total
            return best_so_far, best_so_far
        elif rang[i] == 0:  # vertex i does not have any edges.
            rst, best_so_far = self._dp_for_wmvc(x, i + 1, total, cg, rang, best_so_far)
            best_so_far = min(best_so_far, rst)
            return best_so_far, best_so_far

        cols = len(x)
        # find minimum cost for this vertex
        min_cost = np.max(cg.reshape(-1, cols)[:i, i] - x[i])
        # for j in range(i):
        #     if min_cost + x[j] < cg[j * cols + i]:  # infeasible assignment
        #         min_cost = cg[j * cols + i] - x[j]  # cost should be at least CG[i][j] - x[j];

        best_cost = -1
        for cost in range(min_cost, rang[i] + 1):
            x[i] = cost
            rst, best_so_far = self._dp_for_wmvc(x, i + 1, total + x[i].item(), cg, rang, best_so_far)
            if rst < best_so_far:
                best_so_far = rst
                best_cost = cost
        if best_cost >= 0:
            x[i] = best_cost

        return best_so_far, best_so_far

    def _ilp_for_constrained_wmvc(self, cg: np.ndarray, rang: List[int]) -> int:
        pass

    def _dp_for_constrained_wmvc(self, x: List[int], i, total: int, cg: np.ndarray,
                                 rang: List[int], best_so_far: int) -> int:
        if total >= best_so_far:
            return sys.maxsize
        runtime = time.perf_counter() - self.start_time
        if runtime > self.time_limit:
            return -1
        elif i == len(x):
            best_so_far = total
            return best_so_far
        elif rang[i] == 0:
            rst = self._dp_for_constrained_wmvc(x, i + 1, total, cg, rang, best_so_far)
            best_so_far = min(best_so_far, rst)
            return best_so_far

        cols = len(x)
        min_cost = np.max(cg.reshape(-1, cols)[:i, i] - x[i])
        if min_cost == 0:
            x[i] = 0
            rst = self._dp_for_constrained_wmvc(x, i + 1, total, cg, rang, best_so_far)
            best_so_far = min(best_so_far, rst)
        if min_cost < rang[i]:
            x[i] = 1
            rst = self._dp_for_constrained_wmvc(x, i + 1, total + x[i] * rang[i], cg, rang, best_so_far)
            best_so_far = min(best_so_far, rst)

        return best_so_far


if __name__ == "__main__":
    pass
