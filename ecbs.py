# -*- coding:utf-8 -*-
# @FileName  :ecbs.py
# @Time      :2024/7/22 下午1:05
# @Author    :ZMFY
# Description:
import heapq as hpq
import sys
import time
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np

import common as cm
from cbs import CBS, HighLevelSolverType
from conflict import ConstraintType, ConflictType, Conflict, ConflictPriority
from constraint_table import ConstraintTable
from instance import Instance
from nodes import ECBSNode


class ECBS(CBS):
    def __init__(self, instance: Instance, sipp: bool = False, screen=2):
        super().__init__(instance, sipp, screen)
        # lower bounds of the cost of the shortest path
        self.min_f_vals: np.ndarray = np.zeros(self.num_of_agents, dtype=int)
        self.paths_found_initially: List[Tuple[cm.Path, int]] = []  # contain initial paths found

    def solve(self, time_limit, cost_lowerbound=0, cost_upperbond=cm.MAX_COST):
        self.cost_lowerbound = cost_lowerbound
        self.inadmissible_cost_lowerbound = 0
        self.time_limit = time_limit

        if self.screen > 0:
            print(self.get_solver_name() + " : ")

        self.start_time = time.perf_counter()
        self._generate_root()

        while not self.cleanup_list.empty() and not self.solution_found:
            curr = self._select_node()
            if self._terminate(curr):
                return self.solution_found

            if (curr == self.dummy_start or curr.chosen_form == 'cleanup') and not curr.h_computed:
                self.runtime = time.perf_counter() - self.start_time
                succ = self.heuristic_helper.compute_ecbs_informed_heuristics(
                    curr, self.min_f_vals, time_limit - self.runtime)
                self.runtime = time.perf_counter() - self.start_time
                if not succ:
                    if self.screen > 1:
                        print(f"	Prune {curr}")
                    curr.clear()
                    continue

                if self._reinsert_node(curr):
                    continue
            self._classify_conflicts(curr)

            # Expand the node
            self.num_HL_expanded += 1
            curr.time_expanded = self.num_HL_expanded
            if self.bypass and curr.chosen_form != 'cleanup':
                found_bypass = True
                while found_bypass:
                    if self._terminate(curr):
                        return self.solution_found
                    found_bypass = False
                    children = [ECBSNode(), ECBSNode()]
                    curr.conflict = self._choose_conflict(curr)
                    self.add_constraints(curr, children[0], children[1])
                    if self.screen > 1:
                        print(f"	Expand {curr}\n	on {curr.conflict}")
                    solved = [False, False]
                    path_copy = deepcopy(self.paths)
                    fmin_cpy = self.min_f_vals.copy()
                    for i in range(2):
                        if i > 0:
                            self.paths = path_copy
                            self.min_f_vals = fmin_cpy
                        solved[i] = self._generate_child(children[i], curr)
                        if not solved[i]:
                            print('no child generated.')
                            # del children[i]
                            continue
                        elif i == 1 and not solved[0]:
                            continue
                        elif self.bypass and children[i].sum_of_costs <= self.sub_optimality * self.cost_lowerbound \
                                and children[i].distance_to_go < curr.distance_to_go:
                            found_bypass = True
                            for path in children[i].paths:
                                if len(path[1][0]) - 1 > self.sub_optimality * fmin_cpy[path[0]]:
                                    found_bypass = False
                                    break
                            if found_bypass:
                                self._adopt_bypass(curr, children[i], fmin_cpy)
                                if self.screen > 1:
                                    print(f"	Update {curr}")
                                break
                    if found_bypass:
                        self._classify_conflicts(curr)  # classify the new-detected conflicts
                        # children.clear()
                    else:
                        for i in range(2):
                            if solved[i]:
                                self._push_node(children[i])
                                curr.children.append(children[i])
                                if self.screen > 1:
                                    print(f"		Generate {children[i]}")
            else:   # no bypass
                children = [ECBSNode(), ECBSNode()]
                curr.conflict = self._choose_conflict(curr)
                self.add_constraints(curr, children[0], children[1])
                if self.screen > 1:
                    print(f"	Expand {curr}\n	on {curr.conflict}")
                solved = [False, False]
                path_copy = deepcopy(self.paths)
                fmin_cpy = self.min_f_vals.copy()
                for i in range(2):
                    if i > 0:
                        self.paths = path_copy
                        self.min_f_vals = fmin_cpy
                    solved[i] = self._generate_child(children[i], curr)
                    if not solved[i]:
                        del children[i]
                        continue
                    self._push_node(children[i])
                    curr.children.append(children[i])
                    if self.screen > 1:
                        print(f"		Generate {children[i]}")

            if curr.conflict.con_type == ConflictType.RECTANGLE:
                self.num_rectangle_conflicts += 1
            elif curr.conflict.con_type == ConflictType.CORRIDOR:
                self.num_corridor_conflicts += 1
            elif curr.conflict.con_type == ConflictType.TARGET:
                self.num_target_conflicts += 1
            elif curr.conflict.con_type == ConflictType.STANDARD:
                self.num_standard_conflicts += 1
            elif curr.conflict.con_type == ConflictType.MUTEX:
                self.num_mutex_conflicts += 1

            if curr.chosen_form == 'cleanup':
                self.num_cleanup += 1
            elif curr.chosen_form == 'open':
                self.num_open += 1
            elif curr.chosen_form == 'focal':
                self.num_focal += 1

            if curr.conflict.priority == ConflictPriority.CARDINAL:
                self.num_cardinal_conflicts += 1
            if not curr.children:
                self.heuristic_helper.update_ecbs_online_heuristic_errors(curr)
            # curr.clear()

        return self.solution_found

    @staticmethod
    def solve_two_agents(a1, a2, node: ECBSNode, engines, initial_constraints, screen=1) -> Tuple[int, int]:
        """return optimal f and a1_shortest path * #agents + a2_shortest path"""
        pass

    def clear(self):
        """used for rapid random  restart"""
        self.min_f_vals.fill(0)
        super().clear()

    def _adopt_bypass(self, curr: ECBSNode, child: ECBSNode, fmin_copy: List[int]):
        self.num_adopt_bypass += 1
        curr.sum_of_costs = child.sum_of_costs
        curr.distance_to_go = child.distance_to_go
        curr.conflicts = child.conflicts
        curr.unknown_conf = child.unknown_conf
        curr.conflict = None
        curr.makespan = child.makespan

        for path in child.paths:  # update paths
            found = False
            for i, p in enumerate(curr.paths):
                if path[0] == p[0]:
                    curr.paths[i] = (p[0], (path[1][0], p[1][1]))
                    self.paths[p[0]] = path[1][0]
                    self.min_f_vals[p[0]] = p[1][1]
                    found = True
                    break
            if not found:
                curr.paths.append((path[0], (path[1][0], fmin_copy[path[0]])))
                self.paths[path[0]] = curr.paths[-1][1][0]
                self.min_f_vals[path[0]] = fmin_copy[path[0]]

    def _push_node(self, node: ECBSNode):
        self.num_HL_generated += 1
        node.time_generated = self.num_HL_generated

        # hpq.heappush(self.cleanup_list, node.get_sort_tuple_for_cleanup())
        self.cleanup_list.add(node, *node.get_sort_tuple_for_open())
        if self.solver_type == HighLevelSolverType.ASTAREPS:
            # cleanup_list is called open_list in ECBS
            if node.sum_of_costs <= self.sub_optimality * self.cost_lowerbound:
                # hpq.heappush(self.focal_list, node.get_sort_tuple_for_focal())
                self.focal_list.add(node, *node.get_sort_tuple_for_focal())
        elif self.solver_type == HighLevelSolverType.NEW:
            if node.f_hat_val <= self.sub_optimality * self.cost_lowerbound:
                # hpq.heappush(self.focal_list, node.get_sort_tuple_for_focal())
                self.focal_list.add(node, *node.get_sort_tuple_for_focal())
        elif self.solver_type == HighLevelSolverType.EES:
            # hpq.heappush(self.open_list, node)
            self.open_list.add(node, *node.get_sort_tuple_for_open())
            if node.f_hat_val <= self.sub_optimality * self.inadmissible_cost_lowerbound:
                # hpq.heappush(self.focal_list, node.get_sort_tuple_for_focal())
                self.focal_list.add(node, *node.get_sort_tuple_for_focal())

        if self.screen > 1:
            print(f"Push {node}")

        self.all_nodes_table.append(node)

    def _select_node(self) -> Union[ECBSNode, None]:
        assert self.solver_type != HighLevelSolverType.ASTAR
        curr = None
        if self.solver_type == HighLevelSolverType.EES:
            # update the focal list if necessary
            # top = hpq.nsmallest(1, self.open_list)[0]
            open_top = self.open_list.top()
            if open_top.f_hat_val != self.inadmissible_cost_lowerbound:
                self.inadmissible_cost_lowerbound = open_top.f_hat_val
                focal_list_threshold = self.sub_optimality * self.inadmissible_cost_lowerbound
                self.focal_list.clear()
                for n in self.open_list:
                    if n.f_hat_val <= focal_list_threshold:
                        self.focal_list.add(n, *n.get_sort_tuple_for_focal())
                # self.focal_list += [n.get_sort_tuple_for_focal() for n in self.open_list
                #                     if n.f_hat_val <= focal_list_threshold]
                # hpq.heapify(self.focal_list)

            # choose the best node
            # cleanup_top = hpq.nsmallest(1, self.cleanup_list)[0]
            cleanup_top = self.cleanup_list.top()
            if self.screen > 1 and cleanup_top.f_val > self.cost_lowerbound:
                print(f"Lower bound increases from {self.cost_lowerbound} to {cleanup_top.f_val}")
            self.cost_lowerbound = max(cleanup_top.f_hat_val, self.cost_lowerbound)
            # focal_top = hpq.nsmallest(1, self.focal_list)[0]
            focal_top = self.focal_list.top()
            if focal_top.sum_of_costs <= self.sub_optimality * self.cost_lowerbound:
                # curr: ECBSNode = hpq.heappop(self.focal_list)[-1]
                curr: ECBSNode = self.focal_list.pop()
                curr.chosen_form = 'focal'
                # self.cleanup_list.remove(curr.get_sort_tuple_for_cleanup())
                self.cleanup_list.remove(curr)
                self.open_list.remove(curr)
            elif open_top.sum_of_costs <= self.sub_optimality * self.cost_lowerbound:
                # curr: ECBSNode = hpq.heappop(self.open_list)
                curr: ECBSNode = self.open_list.pop()
                curr.chosen_form = 'open'
                # self.cleanup_list.remove(curr.get_sort_tuple_for_cleanup())
                # self.focal_list.remove(curr.get_sort_tuple_for_focal())
                self.focal_list.remove(curr)
                self.cleanup_list.remove(curr)
            else:
                # curr: ECBSNode = hpq.heappop(self.cleanup_list)[-1]
                curr: ECBSNode = self.cleanup_list.pop()
                curr.chosen_form = 'cleanup'
                self.open_list.remove(curr)
                if curr.f_hat_val <= self.sub_optimality * self.inadmissible_cost_lowerbound:
                    # self.focal_list.remove(curr.get_sort_tuple_for_focal())
                    self.focal_list.remove(curr)
        elif self.solver_type == HighLevelSolverType.ASTAREPS:
            # cleanup_top = hpq.nsmallest(1, self.cleanup_list)[0]
            cleanup_top = self.cleanup_list.top()
            if cleanup_top.f_val > self.cost_lowerbound:
                old_focal_list_threshold = self.sub_optimality * self.cost_lowerbound
                self.cost_lowerbound = max(self.cost_lowerbound, cleanup_top.f_val)
                new_focal_list_threshold = self.sub_optimality * self.cost_lowerbound
                for n in self.cleanup_list:
                    if old_focal_list_threshold < n.sum_of_costs < new_focal_list_threshold:
                        self.focal_list.add(n, *n.get_sort_tuple_for_focal())
                # self.focal_list += [n[-1].get_sort_tuple_for_focal() for n in self.cleanup_list
                #                     if old_focal_list_threshold < n[-1].sum_of_costs < new_focal_list_threshold]
                # hpq.heapify(self.focal_list)
                if self.screen == 3:
                    print(f"Note -- FOCAL UPDATE!! from |FOCAL|={len(self.focal_list)} "
                          f"with |OPEN|={len(self.cleanup_list)} to |FOCAL|={len(self.focal_list)}")

            # curr = hpq.heappop(self.focal_list)[-1]
            curr = self.focal_list.pop()
            curr.chosen_form = 'focal'
            # self.cleanup_list.remove(curr.get_sort_tuple_for_cleanup())
            self.cleanup_list.remove(curr)
        elif self.solver_type == HighLevelSolverType.NEW:
            # cleanup_top = hpq.nsmallest(1, self.cleanup_list)[0]
            cleanup_top = self.cleanup_list.top()
            if cleanup_top.f_val > self.cost_lowerbound:
                old_focal_list_threshold = self.sub_optimality * self.cost_lowerbound
                self.cost_lowerbound = max(self.cost_lowerbound, cleanup_top.f_val)
                new_focal_list_threshold = self.sub_optimality * self.cost_lowerbound
                self.focal_list.clear()
                for n in self.cleanup_list:
                    self.heuristic_helper.update_inadmissible_heuristics(n)
                    if n.f_hat_val <= new_focal_list_threshold:
                        # hpq.heappush(self.focal_list, n[-1].get_sort_tuple_for_focal())
                        self.focal_list.add(n, *n.get_sort_tuple_for_focal())
                if self.screen == 3:
                    print(f"Note -- FOCAL UPDATE!! from |FOCAL|={len(self.focal_list)} "
                          f"with |OPEN|={len(self.cleanup_list)} to |FOCAL|={len(self.focal_list)}")
            if not self.focal_list:
                # curr = hpq.heappop(self.cleanup_list)[-1]
                curr = self.cleanup_list.pop()
                curr.chosen_form = 'cleanup'
            else:
                # curr: ECBSNode = hpq.heappop(self.focal_list)[-1]
                curr: ECBSNode = self.focal_list.pop()
                curr.chosen_form = 'focal'
                # self.cleanup_list.remove(curr.get_sort_tuple_for_cleanup())
                self.cleanup_list.remove(curr)

        self._update_paths(curr)

        if self.screen > 1:
            print(f"Pop {curr}. \n\tnow |OPEN|={len(self.focal_list)}, |FOCAL|={len(self.cleanup_list)}, "
                  f"|CLEANUP|={len(self.cleanup_list)}")

        return curr

    def _reinsert_node(self, node: ECBSNode):
        if self.solver_type == HighLevelSolverType.ASTAREPS:
            if node.sum_of_costs <= self.sub_optimality * self.cost_lowerbound:
                return False
            # hpq.heappush(self.cleanup_list, node.get_sort_tuple_for_cleanup())
            self.cleanup_list.add(node, *node.get_sort_tuple_for_cleanup())
        elif self.solver_type == HighLevelSolverType.NEW:
            if node.f_hat_val <= self.sub_optimality * self.cost_lowerbound:
                return False
            # hpq.heappush(self.cleanup_list, node.get_sort_tuple_for_cleanup())
            self.cleanup_list.add(node, *node.get_sort_tuple_for_cleanup())
        elif self.solver_type == HighLevelSolverType.EES:
            # hpq.heappush(self.cleanup_list, node.get_sort_tuple_for_cleanup())
            self.cleanup_list.add(node, *node.get_sort_tuple_for_cleanup())
            # hpq.heappush(self.open_list, node)
            self.open_list.add(node, *node.get_sort_tuple_for_open())
            if node.f_hat_val <= self.sub_optimality * self.inadmissible_cost_lowerbound:
                # hpq.heappush(self.focal_list, node.get_sort_tuple_for_focal())
                self.focal_list.add(node, *node.get_sort_tuple_for_focal())

        if self.screen >= 2:
            print(f"	Reinsert {node}")

    def _generate_child(self, node: ECBSNode, parent: ECBSNode) -> bool:
        st = time.perf_counter()
        node.parent = parent
        node.g_val = parent.g_val
        node.sum_of_costs = parent.sum_of_costs
        node.makespan = parent.makespan
        node.depth = parent.depth + 1
        agents = self._get_invalid_agents(node.constraints)
        assert len(agents) != 0
        for agent in agents:
            if not self._find_path_for_single_agent(node, agent):
                if self.screen > 1:
                    print(f"	No paths for agent {agent}. Node pruned.")
                self.runtime_generate_child += time.perf_counter() - st
                return False

        self._find_conflicts(node)
        self.heuristic_helper.compute_quick_heuristics(node)
        self.runtime_generate_child += time.perf_counter() - st
        return True

    def _generate_root(self) -> bool:
        root = ECBSNode()
        root.g_val = 0
        root.sum_of_costs = 0
        self.paths = [[] for _ in range(self.num_of_agents)]
        self.min_f_vals = np.zeros(self.num_of_agents, dtype=int)
        self.mdd_helper.init(self.num_of_agents)
        self.heuristic_helper.init()

        # initialize paths_found_initially
        assert len(self.paths_found_initially) == 0
        # generate random permutation of agent indices
        agents = self._shuffle_agents()

        for i in agents:
            individual_path = self.search_engines[i].find_suboptimal_path(
                root, self.initial_constraints[i], self.paths, i, 0, self.sub_optimality
            )
            self.paths_found_initially.append(individual_path)
            # print(f"Path for Agent {i}: {self.paths_found_initially[i][0]}")
            if len(individual_path[0]) == 0:
                print(f"The start-goal locations of agent {i} are not connected")
                sys.exit(-1)
            self.paths[i] = individual_path[0]
            self.min_f_vals[i] = individual_path[1]
            root.makespan = max(root.makespan, len(self.paths[i]) - 1)
            root.g_val += self.min_f_vals[i]
            root.sum_of_costs += len(self.paths[i]) - 1
            self.num_LL_expanded += self.search_engines[i].num_expanded
            self.num_LL_generated += self.search_engines[i].num_generated

        root.h_val = 0
        root.depth = 0
        self._find_conflicts(root)
        self.heuristic_helper.compute_quick_heuristics(root)
        self._push_node(root)
        self.dummy_start = root

        if self.screen >= 2:
            self._print_paths()
            print(f'root generated: {root}')

        return True

    def _find_path_for_single_agent(self, node: ECBSNode, agent, lower_bound=0) -> bool:
        st = time.perf_counter()
        new_path = self.search_engines[agent].find_suboptimal_path(
            node, self.initial_constraints[agent], self.paths, agent, self.min_f_vals[agent], self.sub_optimality
        )
        self.num_LL_expanded += self.search_engines[agent].num_expanded
        self.num_LL_generated += self.search_engines[agent].num_generated
        self.runtime_build_ct += self.search_engines[agent].runtime_build_ct
        self.runtime_build_cat += self.search_engines[agent].runtime_build_cat
        self.runtime_path_finding += time.perf_counter() - st
        if len(new_path[0]) == 0:
            return False

        assert not cm.is_same_path(self.paths[agent], new_path[0])
        node.paths.append((agent, new_path))
        node.g_val = node.g_val - self.min_f_vals[agent] + new_path[1]
        node.sum_of_costs = node.sum_of_costs - len(self.paths[agent]) + len(new_path[0])
        self.paths[agent] = node.paths[-1][1][0]
        self.min_f_vals[agent] = new_path[1]
        node.makespan = max(node.makespan, len(new_path[0]) - 1)
        return True

    def _classify_conflicts(self, node: ECBSNode):
        if not node.unknown_conf:
            return

        node.unknown_conf.reverse()
        # Classify all conflicts in unknownConf
        while len(node.unknown_conf) != 0:
            conflict = node.unknown_conf.pop()
            a1, a2 = conflict.a1, conflict.a2
            timestep, flag = conflict.constraint1[-1].t, conflict.constraint2[-1].flag

            if self.pc:
                if node.chosen_form == 'cleanup' or \
                        len(self.paths[a1]) - 1 == self.min_f_vals[a1] or \
                        len(self.paths[a2]) - 1 == self.min_f_vals[a2]:
                    # the path of at least one agent is its shortest path
                    self._compute_conflict_priority(conflict, node)

            # Target Reasoning
            if conflict.con_type == ConflictType.TARGET:
                self._compute_second_priority_for_conflict(conflict, node)
                node.conflicts.append(conflict)
                continue

            # Corridor reasoning
            if self.corridor_reasoning:
                corridor = self.corridor_helper.run(conflict, self.paths, node)
                if corridor is not None:
                    corridor.priority = conflict.priority
                    self._compute_second_priority_for_conflict(corridor, node)
                    node.conflicts.append(corridor)
                    continue

            # Rectangle reasoning
            if self.rectangle_reasoning \
                    and len(self.paths[a1]) - 1 == self.min_f_vals[a1] \
                    and len(self.paths[a2]) - 1 == self.min_f_vals[a2] \
                    and self.min_f_vals[a1] > timestep and self.min_f_vals[a2] > timestep \
                    and flag == ConstraintType.VERTEX:
                mdd1 = self.mdd_helper.get_mdd(node, a1, len(self.paths[a1]))
                mdd2 = self.mdd_helper.get_mdd(node, a2, len(self.paths[a2]))
                rectangle = self.rectangle_helper.run(self.paths, timestep, a1, a2, mdd1, mdd2)
                if rectangle is not None and not self.pc:
                    rectangle.priority = ConflictPriority.UNKNOWN
                    self._compute_second_priority_for_conflict(rectangle, node)
                    node.constraints.append(rectangle)
                    continue

            self._compute_second_priority_for_conflict(conflict, node)
            node.conflicts.append(conflict)

        self._remove_low_priority_conflicts(node.conflicts)

    def _compute_conflict_priority(self, conflict: Conflict, node: ECBSNode):
        a1, a2 = conflict.a1, conflict.a2
        timestep, flag = conflict.constraint1[-1].t, conflict.constraint1[-1].flag
        cardinal1, cardinal2 = False, False
        if timestep >= len(self.paths[a1]):
            cardinal1 = True
            mdd1 = None
        else:
            mdd1 = self.mdd_helper.get_mdd(node, a1)
        if timestep >= len(self.paths[a2]):
            cardinal2 = True
            mdd2 = None
        else:
            mdd2 = self.mdd_helper.get_mdd(node, a2)

        if flag == ConstraintType.EDGE:  # Edge conflict
            if timestep < len(mdd1.levels):
                cardinal1 = len(mdd1.levels[timestep]) == 1 and len(mdd1.levels[timestep - 1]) == 1 \
                            and mdd1.levels[timestep][0].location == self.paths[a1][timestep].location and \
                            mdd1.levels[timestep - 1][0].location == self.paths[a1][timestep - 1].location
            if timestep < len(mdd2.levels):
                cardinal2 = len(mdd2.levels[timestep]) == 1 and len(mdd2.levels[timestep - 1]) == 1 \
                            and mdd2.levels[timestep][0].location == self.paths[a2][timestep].location and \
                            mdd2.levels[timestep - 1][0].location == self.paths[a2][timestep - 1].location
        else:  # vertex conflict or target conflict
            if not cardinal1 and timestep < len(mdd1.levels):
                cardinal1 = len(mdd1.levels[timestep]) == 1 and \
                            mdd1.levels[timestep][0].location == self.paths[a1][timestep].location
            if not cardinal2 and timestep < len(mdd2.levels):
                cardinal2 = len(mdd2.levels[timestep]) == 1 and \
                            mdd2.levels[timestep][0].location == self.paths[a2][timestep].location

        if cardinal1 and cardinal2:
            conflict.priority = ConflictPriority.CARDINAL
        elif cardinal1 or cardinal2:
            conflict.priority = ConflictPriority.SEMI
        else:
            conflict.priority = ConflictPriority.NON

    def _update_paths(self, curr: ECBSNode):
        """
        takes the paths_found_initially and UPDATE all (constrained) paths found for agents from curr to start
        also, do the same for ll_min_f_vals and paths_costs (since its already "on the way")
        """
        self.paths = [p[0] for p in self.paths_found_initially]
        self.min_f_vals = np.array([p[1] for p in self.paths_found_initially], dtype=int)
        updated = np.zeros(self.num_of_agents, dtype=bool)
        while curr is not None:
            for path in curr.paths:
                agent = path[0]
                if not updated[agent]:
                    self.paths[agent] = path[1][0]
                    self.min_f_vals[agent] = path[1][1]
                    updated[agent] = True
            curr = curr.parent

    def _print_paths(self):
        for i in range(self.num_of_agents):
            txt = f"Agent {i} ({len(self.paths_found_initially[i][0]) - 1} --> {len(self.paths[i]) - 1}): "
            txt += '->'.join([str(t.location) for t in self.paths[i]])
            print(txt)


if __name__ == "__main__":
    pass
