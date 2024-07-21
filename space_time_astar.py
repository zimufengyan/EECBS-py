# -*- coding:utf-8 -*-
# @FileName  :space_time_astar.py
# @Time      :2024/7/21 下午3:53
# @Author    :ZMFY
# Description:
import heapq as hpq
import time
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np

import common as cm
from contraint_table import ConstraintTable
from instance import Instance
from nodes import AstarNode, LLNode
from single_agent_solver import SingleAgentSolver
from nodes import HLNode


class SpaceTimeSolver(SingleAgentSolver):
    def __init__(self, instance: Instance, agent: int):
        super().__init__(instance, agent)
        self.open_list = []
        self.focal_list = []  # <num_of_conflicts, f_val, h_val, random, node>
        self.all_nodes_table: Dict[int, AstarNode] = dict()

    def get_name(self):
        return "Astar"

    def find_optimal_path(self, node: HLNode, initial_constraint: ConstraintTable,
                          paths: List[cm.Path], agent: int, lower_bound: int):
        return self.find_optimal_path(node, initial_constraint, paths, agent, lower_bound)[0]

    def find_suboptimal_path(self, node: HLNode, initial_constraint: ConstraintTable,
                             paths: List[cm.Path], agent: int, lower_bound: int, w: float) -> Tuple[cm.Path, int]:
        """
        find path by time-space A* search
        Returns a bounded-suboptimal path that satisfies the constraints of the give node  while
        minimizing the number of internal conflicts (that is conflicts with known_paths for other agents found so far).
        lowerbound is an underestimation of the length of the path in order to speed up the search.
        """
        self.w = w
        path = cm.Path()
        self.num_expanded = 0
        self.num_generated = 0

        # build constraint table
        st = time.perf_counter()
        constraint_table = deepcopy(initial_constraint)
        constraint_table.insert_node_to_ct(node, agent)
        self.runtime_build_ct = time.perf_counter() - st
        if constraint_table.constrained(self.start_location, 0):
            return path, 0

        st = time.perf_counter()
        constraint_table.insert_paths_to_cat(agent, paths)
        self.runtime_build_cat = time.perf_counter() - st

        # the earliest timestep that the agent can hold its goal location. The length_min is considered here.
        holding_time = constraint_table.get_holding_time(self.goal_location, constraint_table.length_min)
        static_timestep = constraint_table.get_max_timestep() + 1
        lower_bound = max(lower_bound, holding_time)

        # generate start and add it to the OPEN & FOCAL list
        start = AstarNode(self.start_location, 0, max(lower_bound, self.my_heuristic[self.start_location]),
                          None, 0, 0)
        self.num_generated += 1
        hpq.heappush(self.open_list, start)
        start.in_openlist = True
        hpq.heappush(self.focal_list, (
            start.num_of_conflicts, start.f_val, start.h_val, np.random.random(), start
        ))
        key = start.get_hash_key()
        self.all_nodes_table[key] = start
        self.min_f_val = start.f_val

        while len(self.open_list) > 0:
            self._update_focal_list()  # update FOCAL if min f-val increased
            curr = self.pop_node()
            assert curr.location >= 0

            # check if the popped node is a goal
            if (curr.location == self.goal_location  # arrive at the goal location
                    and not curr.wait_at_goal  # not wait at the goal location
                    and curr.timestep >= holding_time):  # the agent can hold the goal location afterward
                path = self._update_path(curr, path)
                break
            if curr.timestep >= constraint_table.length_max:
                continue

            next_locations = self.get_next_locations(curr.location)
            for next_location in next_locations:
                next_timestep = curr.timestep + 1
                if static_timestep < next_timestep:
                    # now everything is static, so switch to space A* where we always use the same timestep
                    if next_location == curr.location:
                        continue
                    next_timestep -= 1
                if (constraint_table.constrained(next_location, next_timestep) or
                        constraint_table.constrained(curr.location, next_location, next_timestep)):
                    continue

                # compute cost to next_id via curr node
                next_g_val = curr.g_val + 1
                next_h_val = max(lower_bound - next_g_val, self.my_heuristic[next_location])
                if next_g_val + next_h_val > constraint_table.length_max:
                    continue
                next_internal_conflicts = (curr.num_of_conflicts +
                                           constraint_table.get_num_of_conflicts_for_step(
                                               curr.location, next_location, next_timestep))

                # generate (maybe temporary) node
                nxt = AstarNode(next_location, next_g_val, next_h_val, curr, next_timestep, next_internal_conflicts)
                if next_location == self.goal_location and curr.location == self.goal_location:
                    nxt.wait_at_goal = True

                # try to retrieve it from the hash table
                nxt_key = nxt.get_hash_key()
                existing_node = self.all_nodes_table.get(nxt_key, None)
                if existing_node is None:
                    self.push_node(nxt)
                    self.all_nodes_table[nxt_key] = nxt
                    continue
                else:
                    if (existing_node.f_val > nxt.f_val  # if f-val decreased through this new path
                            or (existing_node.f_val == nxt.f_val  # or it remains the same but there's fewer conflicts
                                and existing_node.num_of_conflicts > nxt.num_of_conflicts)):
                        if existing_node.in_openlist:
                            existing_node.init_from_other(nxt)
                            self.push_node(existing_node)
                        else:
                            # check if it was above the focal bound before and now below (thus need to be inserted)
                            add_to_focal = False
                            # check if it was inside the focal and needs to be updated (because f-val changed)
                            update_in_focal = False
                            update_open = False
                            if next_g_val + next_h_val <= self.w * self.min_f_val:
                                # if the new f-val qualify to be in FOCAL
                                if existing_node.f_val > self.w * self.min_f_val:
                                    # and the previous f-val did not qualify to be in FOCAL then add
                                    add_to_focal = True
                                else:
                                    # and the previous f-val did qualify to be in FOCAL then update
                                    update_in_focal = True
                            if existing_node.f_val > next_g_val + next_h_val:
                                update_open = True

                            existing_node.init_from_other(nxt)  # update existing node
                            if update_open:
                                hpq.heapify(self.open_list)
                            if add_to_focal:
                                hpq.heappush(self.focal_list, (
                                    existing_node.num_of_conflicts, existing_node.f_val, existing_node.h_val,
                                    np.random.random(), existing_node
                                ))
                            if update_in_focal:
                                hpq.heapify(self.focal_list)
        self.release_nodes()
        return path, self.min_f_val

    def get_travel_time(self, start: int, end: int, constraint_table: ConstraintTable, upper_bound: int):
        length = cm.MAX_TIMESTEP
        static_timestep = constraint_table.get_max_timestep() + 1  # everything is static after this timestep
        root = AstarNode(start, 0, self.compute_heuristic(start, end), None, 0, 0)
        key = root.get_hash_key()
        self.all_nodes_table[key] = root
        hpq.heappush(self.open_list, root)

        while len(self.open_list) > 0:
            curr: AstarNode = hpq.heappop(self.open_list)
            if curr.location == end:
                length = curr.g_val
                break
            next_locations = self.instance.get_neighbors(curr.location)
            next_locations.append(curr.location)
            for next_location in next_locations:
                next_timestep = curr.timestep + 1
                next_g_val = curr.g_val + 1
                if static_timestep < next_timestep:
                    if curr.location == next_location:
                        continue
                    next_timestep -= 1
                if (not constraint_table.constrained(next_location, next_timestep)
                        and not constraint_table.constrained(curr.location, next_location, next_timestep)):
                    # if that grid is not blocked
                    next_h_val = self.compute_heuristic(next_location, end)
                    if next_g_val + next_h_val >= upper_bound:
                        # the cost of the path is larger than the upper bound
                        continue
                    nxt = AstarNode(next_location, next_g_val, next_h_val, None, next_timestep, 0)
                    nxt_key = nxt.get_hash_key()
                    existing_node = self.all_nodes_table.get(nxt_key, None)
                    if existing_node is None:
                        # add the newly generated node to heap and hash table
                        hpq.heappush(self.open_list, nxt)
                        self.all_nodes_table[nxt_key] = nxt
                    else:
                        if existing_node.g_val > next_g_val:
                            # update existing node's g_val (only in the heap)
                            existing_node.g_val = next_g_val
                            existing_node.timestep = next_timestep
                            hpq.heapify(self.open_list)  # update open-list

        self.release_nodes()
        return length

    @staticmethod
    def _update_path(goal: LLNode, path: List[cm.PathEntry]) -> List[cm.PathEntry]:
        curr = goal
        if curr.is_goal:
            curr = curr.parent
        new_path = deepcopy(path)
        new_path = new_path[:curr.g_val + 1][::-1] + new_path[curr.g_val + 1:]

        while curr is not None:
            new_path.append(cm.PathEntry(curr.location))
            curr = curr.parent

        new_path.reverse()
        return new_path

    def _update_focal_list(self):
        open_head: AstarNode = self.open_list[0]
        if open_head.get_f_val() > self.min_f_val:
            new_min_f_val = open_head.get_f_val()
            for n in self.open_list:
                if self.w * new_min_f_val >= n.get_f_val() > self.w * self.min_f_val:
                    hpq.heappush(self.focal_list, (
                        n.num_of_conflicts, n.get_f_val(), n.h_val, np.random.random(), n
                    ))

    def pop_node(self) -> AstarNode:
        popped = hpq.heappop(self.focal_list)
        node: AstarNode = popped[-1]
        self.open_list.remove(node)
        node.in_openlist = False
        self.num_expanded += 1
        return node

    def push_node(self, node: AstarNode):
        if node.in_openlist:
            # update node if it was existed in open list
            hpq.heapify(self.open_list)
        else:
            hpq.heappush(self.open_list, node)
        node.in_openlist = True
        self.num_generated += 1
        if node.get_f_val() < self.w * self.min_f_val:
            hpq.heappush(self.focal_list,
                         (node.num_of_conflicts, node.get_f_val(), node.h_val, np.random.random()), node)

    def release_nodes(self):
        self.open_list.clear()
        self.focal_list.clear()
        self.all_nodes_table.clear()


if __name__ == "__main__":
    pass
