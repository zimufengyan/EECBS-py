# -*- coding:utf-8 -*-
# @FileName  :corridor_reasoning.py
# @Time      :2024/7/29 下午1:29
# @Author    :ZMFY
# Description:
import time
from copy import deepcopy
from typing import List, Tuple, Optional

import common as cm
from conflict import Conflict, Constraint, ConstraintType
from constraint_table import ConstraintTable
from nodes import HLNode
from single_agent_solver import SingleAgentSolver


class CorridorReasoning:
    def __init__(self, search_engines: List[SingleAgentSolver], initial_constraints: List[ConstraintTable]):
        self.search_engines = search_engines
        self.initial_constraints = initial_constraints
        self.accumulated_runtime = 0

    def run(self, conflict: Conflict, paths: List[cm.Path], node: HLNode) -> Conflict:
        st = time.perf_counter()
        corridor = self._find_corridor_conflict(conflict, paths, node)
        self.accumulated_runtime += time.perf_counter() - st
        return corridor

    def _find_corridor_conflict(self, conflict: Conflict, paths: List[cm.Path], node: HLNode) -> Optional[Conflict]:
        a = [conflict.a1, conflict.a2]
        agent, loc1, loc2, timestep, flag = conflict.constraint1[-1]
        curr = -1
        if self.search_engines[0].instance.get_degree(loc1) == 2:
            curr = loc1
            if loc2 >= 0:
                timestep -= 1
        elif loc2 >= 0 and self.search_engines[0].instance.get_degree(loc2) == 2:
            curr = loc2
        if curr <= 0:
            return None

        t = [self._get_entering_time(paths[a[i]], paths[a[1 - i]], timestep) for i in range(2)]
        if t[0] > t[1]:
            t.reverse()
            a.reverse()

        u = [paths[a[i]][t[i]].location for i in range(2)]
        if u[0] == u[1]:
            return None

        for i in range(2):
            found = False
            for tim in range(t[i], len(paths[a[i]])):
                if found:
                    break
                if paths[a[i]][tim].location == u[1 - i]:
                    found = True
            if not found:
                return None

        edge, corridor_length = self._get_corridor_length(paths[a[0]], t[0], u[1])
        ct1 = deepcopy(self.initial_constraints[conflict.a1])
        ct1.insert_node_to_ct(node, conflict.a1)
        t3 = self.search_engines[conflict.a1].get_travel_time(
            paths[conflict.a1][0].location, u[1], ct1, cm.MAX_TIMESTEP)
        ct1.insert_ec_to_ct(edge[0], edge[1], 0, cm.MAX_TIMESTEP)  # block the corridor in both directions
        ct1.insert_ec_to_ct(edge[1], edge[0], 0, cm.MAX_TIMESTEP)
        t3_ = self.search_engines[conflict.a1].get_travel_time(
            paths[conflict.a1][0].location, u[1], ct1, t3 + 2 * corridor_length + 1)
        ct2 = deepcopy(self.initial_constraints[conflict.a2])
        ct2.insert_node_to_ct(node, conflict.a2)
        t4 = self.search_engines[conflict.a2].get_travel_time(
            paths[conflict.a2][0].location, u[0], ct2, cm.MAX_TIMESTEP)
        ct2.insert_ec_to_ct(edge[0], edge[1], 0, cm.MAX_TIMESTEP)
        ct2.insert_ec_to_ct(edge[1], edge[0], 0, cm.MAX_TIMESTEP)
        t4_ = self.search_engines[conflict.a2].get_travel_time(
            paths[conflict.a2][0].location, u[0], ct2, t3 + corridor_length + 1)

        if abs(t3 - t4) <= corridor_length and t3_ > t3 and t4 < t4_:
            t1 = min(t3_ - 1, t4 + corridor_length)
            t2 = min(t4_ - 1, t3 + corridor_length)
            corridor = Conflict()
            corridor.corridor_conflict(conflict.a1, conflict.a2, u[1], u[0], t1, t2)
            if self._blocked(paths[corridor.a1], corridor.constraint1[0]) and \
                    self._blocked(paths[corridor.a2], corridor.constraint2[0]):
                return corridor

        return None

    def _find_corridor(self, conflict: Conflict, paths: List[cm.Path],
                       endpoints: List[int], endpoints_time: List[int]) -> int:
        """return the length of the corridor """
        if len(paths[conflict.a1]) <= 1 or len(paths[conflict.a2]) <= 1:
            return 0
        assert len(conflict.constraint1) == 1
        agent, loc1, loc2, t, flag = conflict.constraint1[-1]
        if t < 1:
            return 0
        if loc1 < 0:  # vertex conflict
            if self.search_engines[0].instance.get_degree(loc2) != 2:
                return 0  # not a corridor
            loc1 = loc2
        else:  # edge conflict
            if self.search_engines[1].instance.get_degree(loc1) != 2 \
                    and self.search_engines[0].instance.get_degree(loc2) != 2:
                return 0  # not a corridor

        # the first timestep when agent 1 exits the corridor
        # the first timestep when agent 2 exits the corridor
        endpoints_time[0] = self._get_exiting_time(paths[conflict.a1], t)
        endpoints_time[1] = self._get_exiting_time(paths[conflict.a2], t)
        endpoints[0] = paths[conflict.a1][endpoints[0]].location  # the exit location for agent 1
        endpoints[1] = paths[conflict.a2][endpoints[1]].location  # the exit location for agent 2
        if endpoints[0] == endpoints[1]:  # agents exit the corridor in the same direction
            return 0

        # count the distance between the two endpoints, and
        # check whether the corridor between the two exit locations traverse the conflict location,
        # which indicates whether the two agents come in different directions
        prev = endpoints[0]
        curr = paths[conflict.a1][endpoints[0] - 1].location
        traverse_the_conflicting_location = False
        corridor_length = -1
        while curr != endpoints[1]:
            if curr == loc2:
                traverse_the_conflicting_location = True
            neighbors = self.search_engines[0].instance.get_neighbors(curr)
            if len(neighbors) == 2:  # inside the corridor
                if neighbors[0] == prev:
                    prev = curr
                    curr = neighbors[-1]
                else:
                    assert neighbors[-1] == prev
                    prev = curr
                    curr = neighbors[0]
            else:  # exit the corridor without hitting endpoint2
                return 0  # indicating that the two agents move in the same direction
            corridor_length += 1

        if not traverse_the_conflicting_location:
            return 0

        # When k=2, it might just be a corner cell, which we do not want to recognize as a corridor
        if corridor_length == 2 and \
                (self.search_engines[0].instance.get_col_coordinate(endpoints[0]) !=
                 self.search_engines[0].instance.get_col_coordinate(endpoints[1])) and \
                (self.search_engines[0].instance.get_row_coordinate(endpoints[0]) !=
                 self.search_engines[0].instance.get_row_coordinate(endpoints[1])):
            return 0

        return corridor_length

    def _get_entering_time(self, path: List[cm.PathEntry], path2: List[cm.PathEntry], t: int) -> int:
        if t >= len(path):
            t = len(path) - 1
        loc = path[t].location
        while loc != path[0].location and loc != path2[-1].location and \
                self.search_engines[0].instance.get_degree(loc) == 2:
            t -= 1
            loc = path[t].location

        return t

    def _get_exiting_time(self, path: List[cm.PathEntry], t: int) -> int:
        if t >= len(path):
            t = len(path) - 1
        loc = path[t].location
        while loc != path[-1].location and \
                self.search_engines[0].instance.get_degree(loc) == 2:
            t += 1
            loc = path[t].location

        return t

    @staticmethod
    def _get_corridor_length(path: List[cm.PathEntry], t_start: int, loc_end: int) -> Tuple[Tuple[int, int], int]:
        curr, prev = path[t_start].location, -1
        length = 0  # distance to the start location
        t = t_start
        move_forward, update_edge = True, False
        edge = (0, 0)
        while curr != loc_end:
            t += 1
            nxt = path[t].location
            if nxt == curr:  # wait
                continue
            elif nxt == prev:  # turn around
                move_forward = ~move_forward
            if move_forward:
                if not update_edge:
                    edge = (curr, nxt)
                    update_edge = True
                length += 1
            else:
                length -= 1
            prev = curr
            curr = nxt

        return edge, length

    @staticmethod
    def _blocked(path: cm.Path, constraint: Constraint) -> bool:
        a, loc, t1, t2, flag = constraint
        assert flag == ConstraintType.RANGE
        for t in range(t1, t2):
            if t >= len(path) and loc == path[-1].location:
                return True
            elif t >= 0 and path[t].location == loc:
                return True

        return False


if __name__ == "__main__":
    pass
