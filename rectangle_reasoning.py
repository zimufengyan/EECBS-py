# -*- coding:utf-8 -*-
# @FileName  :rectangle_reasoning.py
# @Time      :2024/7/22 下午5:49
# @Author    :ZMFY
# Description:
import time
from typing import List, Tuple, Union

import numpy as np
from multipledispatch import dispatch

import common as cm
from conflict import Conflict, Constraint, ConstraintType, ConflictPriority
from instance import Instance
from mdd import MDD


class RectangleReasoning:
    """rectangle_strategy strategy"""

    def __init__(self, instance: Instance):
        self.instance = instance
        self.accumulated_runtime = 0

    def run(self, paths: List[cm.Path], timestep: int, a1: int, a2: int, mdd1: MDD, mdd2: MDD):
        st = time.perf_counter()
        rectangle = self._find_rectangle_conflict_by_rm(paths, timestep, a1, a2, mdd1, mdd2)
        self.accumulated_runtime += time.perf_counter() - st
        return rectangle

    def _find_rectangle_conflict_by_rm(self, paths: List[cm.Path], timestep: int,
                                       a1: int, a2: int, mdd1: MDD, mdd2: MDD) -> Conflict:
        rectangle = None
        # Rectangle reasoning for semi and non cardinal vertex conflicts
        s1s = self._get_start_candidates(paths[a1], mdd1, timestep)
        g1s = self._get_goal_candidates(paths[a1], mdd1, timestep)
        s2s = self._get_start_candidates(paths[a2], mdd2, timestep)
        g2s = self._get_goal_candidates(paths[a2], mdd2, timestep)
        location = self.instance.get_coordinate(paths[a1][timestep].location)

        # Try all possible combinations
        flag, area = -1, 0
        for t1_start in s1s:
            for t1_end in g1s:
                s1 = self.instance.get_coordinate(paths[a1][t1_start].location)
                g1 = self.instance.get_coordinate(paths[a1][t1_end].location)
                if self.instance.get_manhattan_distance(s1, g1) != t1_end - t1_start:
                    continue
                for t2_start in s2s:
                    for t2_end in g2s:
                        s2 = self.instance.get_coordinate(paths[a2][t2_start].location)
                        g2 = self.instance.get_coordinate(paths[a2][t2_end].location)
                        if self.instance.get_manhattan_distance(s2, g2) != t2_end - t2_start:
                            continue
                        if not self._is_rectangle_conflict(s1, s2, g1, g2):
                            continue
                        rg = self._get_rg(s1, g1, g2)
                        rs = self._get_rs(s1, s2, g1)
                        new_area = (abs(rs[0] - rg[0]) + 1) * (abs(rs[1] - rg[1]) + 1)
                        new_flag = self._classify_rectangle_conflict(s1, s2, g1, g2, rg)
                        if new_flag > flag or (new_flag == flag and new_area > area):
                            rg_t = timestep + abs(rg[0] - location[0]) + abs(rg[1] - location[1])
                            constrains1, constrains2 = [], []
                            succ = self._add_modified_barrier_constraints(a1, a2, rs, rg, s1, s2,
                                                                          rg_t, mdd1, mdd2, constrains1, constrains2)
                            if succ and self._blocked(paths[a1], constrains1) and self._blocked(paths[a2], constrains2):
                                flag = new_flag
                                area = new_area
                                rectangle = Conflict()
                                rectangle.rectangle_conflict(a1, a2, constrains1, constrains2)
                                if flag == 2:
                                    rectangle.priority = ConflictPriority.CARDINAL
                                    return rectangle
                                elif flag == 1:
                                    rectangle.priority = ConflictPriority.SEMI
                                else:
                                    rectangle.priority = ConflictPriority.NON

        return rectangle

    def _find_rectangle_conflict_by_gr(self, paths: List[cm.Path], timestep: int,
                                       a1: int, a2: int, mdd1: MDD, mdd2: MDD) -> Union[Conflict, None]:
        assert timestep > 0
        from1 = paths[a1][timestep - 1].location
        from2 = paths[a2][timestep - 1].location
        loc = paths[a1][timestep].location

        if from1 == from2 or from1 == loc or from2 == loc or \
                abs(from1 - from2) == 2 or abs(from1 - from2) == self.instance.cols * 2:
            # same direction, wait actions, or opposite direction
            return None
        b1 = []
        t_start = self._get_start_candidate(paths[a1], loc - from1, loc - from2, timestep)
        t_end = self._get_goal_candidate(paths[a1], loc - from1, loc - from2, timestep)
        have_barriers = self._extract_barriers(mdd1, loc, timestep, loc - from1, loc - from2,
                                               paths[a1][t_start].location, paths[a1][t_end].location, t_start, b1)
        if not have_barriers:
            return None

        b2 = []
        t_start = self._get_start_candidate(paths[a2], loc - from1, loc - from2, timestep)
        t_end = self._get_goal_candidate(paths[a2], loc - from1, loc - from2, timestep)
        have_barriers = self._extract_barriers(mdd1, loc, timestep, loc - from1, loc - from2,
                                               paths[a2][t_start].location, paths[a1][t_end].location, t_start, b2)
        if not have_barriers:
            return None

        # Try all possible combinations
        rs, rg, flag = self._generalized_rectangle(paths[a1], paths[a2], b1, b2, timestep)

        if flag < 0:
            return None
        rg_t = timestep + self.instance.get_manhattan_distance(self.instance.get_coordinate(loc), rg)

        constraints1, constraints2 = [], []
        if abs(loc - from1) == 1 or abs(loc - from2) > 1:
            # first agent moves horizontally and second agent moves vertically
            succ = self._add_modified_vertical_barrier_constraint(a1, mdd1, rg[1], rs[0], rg[0], rg_t, constraints1)
            assert succ
            succ = self._add_modified_horizontal_barrier_constraint(a2, mdd2, rg[0], rs[1], rg[1], rg_t, constraints2)
            assert succ
        else:
            succ = self._add_modified_horizontal_barrier_constraint(a1, mdd1, rg[0], rs[1], rg[1], rg_t, constraints1)
            assert succ
            succ = self._add_modified_vertical_barrier_constraint(a2, mdd2, rg[1], rs[0], rg[0], rg_t, constraints2)
            assert succ

        if not self._blocked(paths[a1], constraints1) or not self._blocked(paths[a2], constraints2):
            return None
        rectangle = Conflict()
        rectangle.rectangle_conflict(a1, a2, constraints1, constraints2)
        if flag == 2:
            rectangle.priority = ConflictPriority.CARDINAL
        elif flag == 1:
            rectangle.priority = ConflictPriority.SEMI
        else:
            rectangle.priority = ConflictPriority.NON
        return rectangle

    def _extract_barriers(self, mdd: MDD, loc: int, timestep: int, dir1: int, dir2: int,
                          start: int, goal: int, start_time: int, b: List[Constraint]) -> bool:
        sign1 = dir1 // abs(dir1)
        sign2 = dir2 // abs(dir2)
        if abs(dir1) == 1:  # vertical barriers
            num_barrier = sign1 * (self.instance.get_col_coordinate(goal) - self.instance.get_col_coordinate(start)) + 1
        else:
            num_barrier = sign1 * (self.instance.get_row_coordinate(goal) - self.instance.get_row_coordinate(start)) + 1

        extent_l = np.ones(num_barrier, dtype=int) * cm.MAX_TIMESTEP
        extent_u = np.ones(num_barrier, dtype=int) * -1
        block = np.zeros(num_barrier, dtype=bool)
        blocking = dict()
        n = mdd.levels[0][0]

        if abs(dir1) == 1:
            barrier_time = (timestep + sign1 * (self.instance.get_col_coordinate(n.location) -
                                                self.instance.get_col_coordinate(loc)))
            barrier_time += sign2 * (self.instance.get_row_coordinate(n.location) -
                                     self.instance.get_row_coordinate(loc))
        else:
            barrier_time = (timestep + sign1 * (self.instance.get_row_coordinate(n.location) -
                                                self.instance.get_row_coordinate(loc)))
            barrier_time += sign2 * (self.instance.get_col_coordinate(n.location) -
                                     self.instance.get_col_coordinate(loc))
        if barrier_time == 0:
            extent_l[0] = 0
            extent_u[0] = 0
            block[0] = True
        blocking[n] = block

        for t in range(1, len(mdd.levels)):
            for n in mdd.levels[t]:
                block = np.ones(num_barrier, dtype=bool)
                for parent in n.parents:
                    parent_block = blocking[parent]
                    for i in range(num_barrier):
                        if not parent_block[i]:
                            block[i] = False
                if abs(dir1) == 1:
                    barrier_id = sign1 * (self.instance.get_col_coordinate(n.location) -
                                          self.instance.get_col_coordinate(start))
                    barrier_time = (timestep + sign1 * (self.instance.get_col_coordinate(n.location) -
                                                        self.instance.get_col_coordinate(loc)))
                    barrier_time += sign2 * (self.instance.get_row_coordinate(n.location) -
                                             self.instance.get_row_coordinate(loc))
                else:
                    barrier_id = sign1 * (self.instance.get_row_coordinate(n.location) -
                                          self.instance.get_row_coordinate(start))
                    barrier_time = (timestep + sign1 * (self.instance.get_row_coordinate(n.location) -
                                                        self.instance.get_row_coordinate(loc)))
                    barrier_time += sign2 * (self.instance.get_col_coordinate(n.location) -
                                             self.instance.get_col_coordinate(loc))
                if 0 <= barrier_id < num_barrier and not block[barrier_id] and barrier_time == n.level:
                    if not (len(n.children) == 1 and extent_l[barrier_id] == cm.MAX_TIMESTEP and
                            abs(dir1) * abs(n.location - n.children[0].location) == self.instance.cols):
                        # the only child node is not on the same barrier
                        extent_l[barrier_id] = min(int(extent_l[barrier_id]), n.level)
                        extent_u[barrier_id] = max(int(extent_u[barrier_id]), n.level)
                blocking[n] = block

        n = mdd.levels[-1][0]
        block = blocking[n]
        for i in range(num_barrier):
            if block[i]:
                if abs(dir1) == 1:
                    barrier_start_y = self.instance.get_col_coordinate(start) + sign1 * i
                    barrier_end_y = barrier_start_y
                    time_offset = timestep + i - sign1 * (self.instance.get_col_coordinate(loc)
                                                          - self.instance.get_col_coordinate(start))
                    barrier_start_x = self.instance.get_row_coordinate(loc) + sign2 * (extent_l[i] - time_offset)
                    barrier_end_x = self.instance.get_row_coordinate(loc) + sign2 * (extent_u[i] - time_offset)
                else:
                    barrier_start_y = self.instance.get_row_coordinate(start) + sign1 * i
                    barrier_end_y = barrier_start_y
                    time_offset = timestep + i - sign1 * (self.instance.get_row_coordinate(loc)
                                                          - self.instance.get_row_coordinate(start))
                    barrier_start_x = self.instance.get_col_coordinate(loc) + sign2 * (extent_l[i] - time_offset)
                    barrier_end_x = self.instance.get_col_coordinate(loc) + sign2 * (extent_u[i] - time_offset)
                barrier_end_time = int(extent_u[i])
                b.append(Constraint(-1,  # for now, the agent index is not important,  so we just use -1 for simplicity.
                                    self.instance.linearize_coordinate(barrier_start_x, barrier_end_x),
                                    self.instance.linearize_coordinate(barrier_start_x, barrier_end_y),
                                    barrier_end_time, ConstraintType.BARRIER))

        return not len(b) == 0

    def _is_entry_barrier(self, b1: Constraint, b2: Constraint, dir1: int) -> bool:
        b1_l = self.instance.get_coordinate(b1.loc1)
        b2_l = self.instance.get_coordinate(b2.loc1)
        b1_u = self.instance.get_coordinate(b1.loc2)
        b2_u = self.instance.get_coordinate(b2.loc2)

        if dir1 == self.instance.cols and b1_u[0] >= b2_l[1] and b2_l[0] >= b1_l[0]:
            return True
        elif dir1 == -self.instance.cols and b1_u[0] <= b2_l[1] and b2_l[0] <= b1_l[0]:
            return True
        elif dir1 == 1 and b1_u[1] >= b2_l[1] >= b1_l[1]:
            return True
        elif dir1 == -1 and b1_u[1] <= b2_l[1] <= b1_l[1]:
            return True
        return False

    def _is_exit_barrier(self, b1: Constraint, b2: Constraint, dir1: int) -> bool:
        b1_l = self.instance.get_coordinate(b1.loc1)
        b2_l = self.instance.get_coordinate(b2.loc1)
        b1_u = self.instance.get_coordinate(b1.loc2)
        b2_u = self.instance.get_coordinate(b2.loc2)

        if dir1 == self.instance.cols and b2_u[0] <= b1_l[0]:
            return True
        elif dir1 == -self.instance.cols and b2_u[0] >= b1_l[0]:
            return True
        elif dir1 == 1 and b2_u[1] <= b1_l[1]:
            return True
        elif dir1 == 1 and b2_u[1] >= b1_l[1]:
            return True
        return False

    def get_intersection(self, b1: Constraint, b2: Constraint) -> Tuple[int, int]:
        b1_l = self.instance.get_coordinate(b1.loc1)
        b2_l = self.instance.get_coordinate(b2.loc1)
        b1_u = self.instance.get_coordinate(b1.loc2)
        b2_u = self.instance.get_coordinate(b2.loc2)

        if b1_l[0] == b1_u[0] and b2_l[1] == b2_u[1]:
            return b1_l[0], b2_l[1]
        else:
            return b2_l[0], b1_l[1]

    def _blocked_nodes(self, path: List[cm.PathEntry], rs: Tuple[int, int], rg: Tuple[int, int],
                       rg_t: int, dir1: int) -> bool:
        if abs(dir1) == 1:
            b_l = (rg[0], rs[1])
        else:
            b_l = (rs[0], rg[1])

        t_max = min(rg_t, len(path) - 1)
        t_b_l = rg_t - abs(b_l[0] - rg[0]) - abs(b_l[1] - rg[1])
        t_min = max(0, t_b_l)

        for t in range(t_min, t_max + 1):
            loc = self.instance.linearize_coordinate(b_l[0], b_l[1]) + (t - t_b_l) * dir1
            if path[t].location == loc:
                return True
        return False

    def _is_cut(self, b: Constraint, rs: Tuple[int, int], rg: Tuple[int, int]) -> bool:
        b_l = self.instance.get_coordinate(b.loc1)
        b_u = self.instance.get_coordinate(b.loc2)

        if b_l == b_u:
            if ((rs[0] <= b_l[0] and b_u[0] <= rg[0]) or (rs[0] >= b_l[0] and b_u[0] >= rg[0])) and \
                    ((rs[1] <= b_l[1] and b_u[1] <= rg[1]) or (rs[1] >= b_l[1] and b_u[1] >= rg[1])):
                return True
            return False

        if rs[0] <= b_l[0] <= b_u[0] <= rg[0] and b_l[1] == b_u[1]:
            return True
        elif rs[1] <= b_l[1] <= b_u[1] <= rg[1] and b_l[0] == b_u[0]:
            return True
        elif rs[0] >= b_l[0] >= b_u[0] >= rg[0] and b_l[1] == b_u[1]:
            return True
        elif rs[1] >= b_l[1] >= b_u[1] >= rg[1] and b_l[0] == b_u[0]:
            return True
        return False

    def _generalized_rectangle(self, path1: List[cm.PathEntry], path2: List[cm.PathEntry], b1: List[Constraint],
                               b2: List[Constraint], timestep: int) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        loc = path1[timestep].location
        dir1 = loc - path1[timestep - 1].location
        dir2 = loc - path2[timestep - 1].location
        best_rs, best_rg = (0, 0), (0, 0)
        best_type = -1
        for b1_entry in b1:
            for b2_entry in b2:
                if self._is_entry_barrier(b1_entry, b2_entry, dir1) and self._is_entry_barrier(b2_entry, b1_entry,
                                                                                               dir2):
                    rs = self.get_intersection(b1_entry, b2_entry)
                    i, j = len(b1) - 1, len(b2) - 1
                    while i >= 0 and j >= 0:
                        if not self._is_exit_barrier(b1[i], b2_entry, dir1):
                            break
                        if not self._is_exit_barrier(b2[j], b1_entry, dir2):
                            break
                        rg = self.get_intersection(b1[i], b2[j])
                        rg_t = timestep + self.instance.get_manhattan_distance(rg, self.instance.get_coordinate(loc))
                        if not self._blocked_nodes(path1, rs, rg, rg_t, dir2):
                            i -= 1
                            continue
                        if not self._blocked_nodes(path2, rs, rg, rg_t, dir1):
                            j -= 1
                            continue
                        cut1 = self._is_cut(b1[i], rs, rg)
                        cut2 = self._is_cut(b2[j], rs, rg)
                        flag = int(cut1) + int(cut2)
                        if flag > best_type:
                            best_rg = rg
                            best_rs = rs
                            best_type = flag
                            if best_type == 2:
                                return best_rs, best_rg, best_type
                        if not cut1:
                            i -= 1
                        elif not cut2:
                            j -= 1

        return best_rs, best_rg, best_type

    @dispatch(tuple, tuple, tuple, tuple, int, int)
    def _is_rectangle_conflict(self, s1: Tuple[int, int], s2: Tuple[int, int],
                               g1: Tuple[int, int], g2: Tuple[int, int],
                               g1_t: int, g2_t: int) -> bool:
        return (g1_t == abs(s1[0] - g1[0]) + abs(s1[1] - g1[1])  # Manhattan-optimal
                and g2_t == abs(s2[0] - g2[0]) + abs(s2[1] - g2[1])  # Manhattan-optimal
                and (s1[0] - g1[0]) * (s2[0] - g2[0]) >= 0  # Move in the same direction
                and (s1[1] - g1[1]) * (s2[1] - g2[1]) >= 0)  # Move in the same direction

    @dispatch(tuple, tuple, tuple, tuple)
    def _is_rectangle_conflict(self, s1: Tuple[int, int], s2: Tuple[int, int],
                               g1: Tuple[int, int], g2: Tuple[int, int]) -> bool:
        if s1 == s2:  # A standard cardinal conflict
            return False
        elif s1 == g1 or s2 == g2:
            return False

        if (s1[0] - g1[0]) * (s2[0] - g2[0]) < 0 or (s1[1] - g1[1]) * (s2[1] - g2[1]) < 0:
            # Not move in the same direction
            return False
        else:
            # not a cardinal vertex conflict
            return not ((s1[0] == g1[0] and s2[1] == g2[1]) or (s1[1] == g1[1] and s2[0] == g2[0]))

    @dispatch(tuple, tuple, tuple, tuple)
    def _classify_rectangle_conflict(self, s1: Tuple[int, int], s2: Tuple[int, int],
                                     g1: Tuple[int, int], g2: Tuple[int, int]) -> int:
        """
        Classify rectangle conflicts for CR/R
        Return 2 if it is a cardinal rectangle conflict
        Return 1 if it is a semi-cardinal rectangle conflict
        Return 0 if it is a non-cardinal rectangle conflict
        """
        cardinal1, cardinal2 = 0, 0
        if (s1[0] - s2[0]) * (g1[0] - g2[0]) <= 0:
            cardinal1 += 1
        if (s1[1] - s2[1]) * (g1[1] - g2[1]) <= 0:
            cardinal2 += 1
        return cardinal1 + cardinal2

    @dispatch(tuple, tuple, tuple, tuple, tuple)
    def _classify_rectangle_conflict(self, s1: Tuple[int, int], s2: Tuple[int, int],
                                     g1: Tuple[int, int], g2: Tuple[int, int], rg: Tuple[int, int]) -> int:
        """
        Classify rectangle conflicts for RM
        Return 2 if it is a cardinal rectangle conflict
        Return 1 if it is a semi-cardinal rectangle conflict
        Return 0 if it is a non-cardinal rectangle conflict
        """
        if (s2[0] - s1[0]) * (s1[0] - g1[0]) < 0 and (s2[1] - s1[1]) * (s1[1] - g1[1]) < 0:
            # s1 in the middle
            return 0
        elif (s1[0] - s2[0]) * (s2[0] - g2[0]) < 0 and (s1[1] - s2[1]) * (s2[1] - g2[1]) < 0:
            # s2 in the middle
            return 0

        cardinal1, cardinal2 = 0, 0
        if ((s1[0] == s2[0] and (s1[1] - s2[1]) * (s2[1] - rg[1]) >= 0) or
                s1[0] != s2[0] and (s1[0] - s2[0]) * (s2[0] - rg[0]) < 0):
            if rg[0] == g1[0]:
                cardinal1 = 1
            if rg[1] == g2[1]:
                cardinal2 = 1
        else:
            if rg[1] == g1[1]:
                cardinal1 = 1
            if rg[0] == g2[0]:
                cardinal2 = 1

        return cardinal1 + cardinal2

    @staticmethod
    def _get_rg(s1: Tuple[int, int], g1: Tuple[int, int], g2: Tuple[int, int]) -> Tuple[int, int]:
        # Compute rectangle corner Rg
        x, y = -1, -1
        if s1[0] == g1[0]:
            x = g1[0]
        elif s1[0] < g1[0]:
            x = min(g1[0], g2[0])
        else:
            x = max(g1[0], g2[0])
        if s1[1] == g1[1]:
            y = g1[1]
        elif s1[1] < g1[1]:
            y = min(g1[1], g2[1])
        else:
            y = max(g1[1], g2[1])
        return x, y

    @staticmethod
    def _get_rs(s1: Tuple[int, int], s2: Tuple[int, int], g1: Tuple[int, int]) -> Tuple[int, int]:
        # Compute rectangle corner Rs
        x, y = -1, -1
        if s1[0] == g1[0]:
            x = s1[0]
        elif s1[0] < g1[0]:
            x = max(s1[0], s2[0])
        else:
            x = min(s1[0], s2[0])
        if s1[1] == g1[1]:
            y = s1[1]
        elif s1[1] < g1[1]:
            y = max(s1[1], s2[1])
        else:
            y = min(s1[1], s2[1])
        return x, y

    def _get_start_candidates(self, path: cm.Path, mdd: MDD, timestep: int) -> List[int]:
        starts = []
        for t in range(timestep + 1):
            # Find start that is single and Manhattan-optimal to conflicting location
            if (len(mdd.levels[t]) == 1 and mdd.levels[t][0].location == path[t].location
                    and self.instance.get_manhattan_distance(path[t].location,
                                                             path[timestep].location) == timestep - t):
                starts.append(t)
        return starts

    def _get_goal_candidates(self, path: cm.Path, mdd: MDD, timestep: int) -> List[int]:
        """Compute goal candidates for RM"""
        goals = []
        for t in range(len(path) - 1, timestep - 1, -1):
            # Find end that is single and Manhattan-optimal to conflicting location
            if (len(mdd.levels[t]) == 1 and mdd.levels[t][0].location == path[t].location
                    and self.instance.get_manhattan_distance(path[t].location,
                                                             path[timestep].location) == t - timestep):
                goals.append(t)
        return goals

    @staticmethod
    def _get_start_candidate(path: cm.Path, dir1: int, dir2: int, timestep: int) -> int:
        for t in range(timestep, -1, -1):
            if path[t].location - path[t - 1].location != dir1 and path[t].location - path[t - 1].location != dir2:
                return t
        return 0

    @staticmethod
    def _get_goal_candidate(path: cm.Path, dir1: int, dir2: int, timestep: int) -> int:
        for t in range(timestep, len(path) - 1):
            # Find the earliest start that is single and Manhattan-optimal to conflicting location
            if path[t + 1].location - path[t].location != dir1 and path[t + 1].location - path[t].location != dir2:
                return t
        return len(path) - 1

    def _has_node_on_barrier(self, mdd: MDD, y_start: int, y_end: int, x: int, t_min: int, horizontal: bool) -> bool:
        """
        return true if the  barrier has MDD nodes.
        Assume the barrier is horizontal (otherwise switch x and y).
        (x, y_start) and (x, y_end) are the endpoints of the barrier, and
        t_min and t_min + abs(y_start - y_end) are their corresponding timesteps.
        """
        sign = 1 if y_start < y_end else -1
        t_max = t_min + abs(y_start - y_end)
        for t2 in range(t_min + 1, min(t_max, len(mdd.levels) - 1) + 1):
            if horizontal:
                loc = self.instance.linearize_coordinate(x, y_start + (t2 - t_min) * sign)
            else:
                loc = self.instance.linearize_coordinate(y_start + (t2 - t_min) * sign, x)
            for n in mdd.levels[t2]:
                if n.location == loc:
                    return True
        return False

    def _add_modified_barrier_constraints(self, a1: int, a2: int, rs: Tuple[int, int], rg: Tuple[int, int],
                                          s1: Tuple[int, int], s2: Tuple[int, int], rg_t: int,
                                          mdd1: MDD, mdd2: MDD,
                                          constraints1: List[Constraint],
                                          constraints2: List[Constraint]) -> bool:
        if (s2[0] - s1[0]) * (s1[0] - rg[0]) > 0 and (s2[1] - s1[1]) * (s1[1] - rg[1]) > 0:
            # s1 in the middle
            rs_t = rg_t - self.instance.get_manhattan_distance(rs, rg)
            # try horizontal first
            offset = 1 if rs[0] > rg[0] else -1
            found = self._has_node_on_barrier(mdd2, rs[1], rg[1], rs[0] + offset, rs_t - 1, True)
            if not found:
                # first agent moves vertically and second agent moves horizontally
                if not self._add_modified_horizontal_barrier_constraint(
                        a1, mdd1, rg[0], rs[1], rg[1], rg_t, constraints1):
                    return False
                if not self._add_modified_vertical_barrier_constraint(
                        a2, mdd2, rg[1], rs[0], rg[0], rg_t, constraints2):
                    return False
                return True
            # try vertical then
            offset = 1 if rs[1] > rg[1] else -1
            found = self._has_node_on_barrier(mdd2, rs[0], rg[0], rs[1] + offset, rs_t - 1, False)
            if not found:
                if not self._add_modified_horizontal_barrier_constraint(
                        a1, mdd1, rs[1], rs[0], rg[0], rg_t, constraints1):
                    return False
                if not self._add_modified_vertical_barrier_constraint(
                        a2, mdd2, rs[0], rs[1], rg[1], rg_t, constraints2):
                    return False
                return True
        elif (s1[0] - s2[0]) * (s2[0] - rg[0]) > 0 and (s1[1] - s2[1]) * (s2[1] - rg[1]) > 0:
            #  s2 in the middle
            rs_t = rg_t - self.instance.get_manhattan_distance(rs, rg)
            # try horizontal first
            offset = 1 if rs[0] > rg[0] else -1
            found = self._has_node_on_barrier(mdd1, rs[1], rg[1], rs[0] + offset, rs_t - 1, True)
            if not found:
                if not self._add_modified_horizontal_barrier_constraint(
                        a1, mdd1, rg[1], rs[0], rg[0], rg_t, constraints1):
                    return False
                if not self._add_modified_vertical_barrier_constraint(
                        a2, mdd2, rg[0], rs[1], rg[1], rg_t, constraints2):
                    return False
                return True
            offset = 1 if rs[1] > rg[1] else -1
            found = self._has_node_on_barrier(mdd1, rs[0], rg[0], rs[1] + offset, rs_t - 1, False)
            if not found:
                # first agent moves vertically and second agent moves horizontally
                if not self._add_modified_horizontal_barrier_constraint(
                        a1, mdd1, rg[0], rs[1], rg[1], rg_t, constraints1):
                    return False
                if not self._add_modified_vertical_barrier_constraint(
                        a2, mdd2, rg[1], rs[0], rg[0], rg_t, constraints2):
                    return False
                return True
        elif s1[0] == s2[0]:
            if (s1[1] - s2[1]) * (s2[1] - rg[1]) >= 0:
                if not self._add_modified_horizontal_barrier_constraint(
                        a1, mdd1, rg[1], rs[0], rg[0], rg_t, constraints1):
                    return False
                if not self._add_modified_vertical_barrier_constraint(
                        a2, mdd2, rg[0], rs[1], rg[1], rg_t, constraints2):
                    return False
            else:
                if not self._add_modified_horizontal_barrier_constraint(
                        a1, mdd1, rg[0], rs[1], rg[1], rg_t, constraints1):
                    return False
                if not self._add_modified_vertical_barrier_constraint(
                        a2, mdd2, rg[1], rs[0], rg[0], rg_t, constraints2):
                    return False
        elif (s1[0] - s2[0]) * (s2[0] - rg[0]) >= 0:
            if not self._add_modified_horizontal_barrier_constraint(
                    a1, mdd1, rg[0], rs[1], rg[1], rg_t, constraints1):
                return False
            if not self._add_modified_vertical_barrier_constraint(
                    a2, mdd2, rg[1], rs[0], rg[0], rg_t, constraints2):
                return False
        else:
            if not self._add_modified_horizontal_barrier_constraint(
                    a1, mdd1, rg[1], rs[0], rg[0], rg_t, constraints1):
                return False
            if not self._add_modified_vertical_barrier_constraint(
                    a2, mdd2, rg[0], rs[1], rg[1], rg_t, constraints2):
                return False
        return True

    def _add_modified_horizontal_barrier_constraint(self, agent: int, mdd: MDD, x: int, ri_y: int, rg_y: int, rg_t: int,
                                                    constrains: List[Constraint]) -> bool:
        sign = 1 if ri_y < rg_y else -1
        ri_t = rg_t - abs(ri_y - rg_y)
        t1 = -1
        t_min = max(ri_t, 0)
        t_max = min(rg_t, len(mdd.levels) - 1)

        for t2 in range(t_min, t_max + 1):
            loc = self.instance.linearize_coordinate(x, ri_y + (t2 - ri_t) * sign)
            it = None
            for n in mdd.levels[t2]:
                if n.location == loc:
                    it = n
                    break

            if it is None and t1 >= 0:
                # add constraints [t1, t2)
                loc1 = self.instance.linearize_coordinate(x, ri_y + (t1 - ri_t) * sign)
                loc2 = self.instance.linearize_coordinate(x, ri_y + (t2 - ri_t - 1) * sign)
                constrains.append(Constraint(agent, loc1, loc2, t2 - 1, ConstraintType.BARRIER))
                t1 = -1
                continue
            elif it is not None and t1 < 0:
                t1 = t2
            if it is not None and t2 == t_max:
                # add constraints [t1, t2]
                loc1 = self.instance.linearize_coordinate(x, ri_y + (t1 - ri_t) * sign)
                constrains.append(Constraint(agent, loc1, loc, t2, ConstraintType.BARRIER))

        if len(constrains) == 0:
            return False
        return True

    def _add_modified_vertical_barrier_constraint(self, agent: int, mdd: MDD, y: int, ri_x: int, rg_x: int, rg_t: int,
                                                  constrains: List[Constraint]) -> bool:
        sign = 1 if ri_x < rg_x else -1
        ri_t = rg_t - abs(ri_x - rg_x)
        t1 = -1
        t_min = max(ri_t, 0)
        t_max = min(rg_t, len(mdd.levels) - 1)

        for t2 in range(t_min, t_max + 1):
            loc = self.instance.linearize_coordinate((ri_x + (t2 - ri_t) * sign), y)
            it = None
            for n in mdd.levels[t2]:
                if n.location == loc:
                    it = n
                    break

            if it is None and t1 >= 0:
                # add constraints [t1, t2)
                loc1 = self.instance.linearize_coordinate((ri_x + (t1 - ri_t) * sign), y)
                loc2 = self.instance.linearize_coordinate((ri_x + (t2 - ri_t - 1) * sign), y)
                constrains.append(Constraint(agent, loc1, loc2, t2 - 1, ConstraintType.BARRIER))
                t1 = -1
                continue
            elif it is not None and t1 < 0:
                t1 = t2
            if it is not None and t2 == t_max:
                # add constraints [t1, t2]
                loc1 = self.instance.linearize_coordinate((ri_x + (t1 - ri_t) * sign), y)
                constrains.append(Constraint(agent, loc1, loc, t2, ConstraintType.BARRIER))

        if len(constrains) == 0:
            return False
        return True

    def _blocked(self, path: cm.Path, constraints: List[Constraint]) -> bool:
        for constraint in constraints:
            a, x, y, t, flag = constraint
            assert flag == ConstraintType.BARRIER
            x1, y1 = self.instance.get_row_coordinate(x), self.instance.get_col_coordinate(x)
            x2, y2 = self.instance.get_row_coordinate(y), self.instance.get_col_coordinate(y)
            if x1 == x2:
                if y1 < y2:
                    for i in range(min(y2 - y1, t)):
                        if self._traverse(path, self.instance.linearize_coordinate(x1, y2 - i), t - i):
                            return True
                else:
                    for i in range(min(y1 - y2, t)):
                        if self._traverse(path, self.instance.linearize_coordinate(x1, y2 + i), t - i):
                            return True
            else:  # y1 == y2
                if x1 < x2:
                    for i in range(min(x2 - x1, t)):
                        if self._traverse(path, self.instance.linearize_coordinate(x2 - i, y1), t - i):
                            return True
                else:
                    for i in range(min(x1 - x2, t)):
                        if self._traverse(path, self.instance.linearize_coordinate(x2 + i, y1), t - i):
                            return True
        return False

    @staticmethod
    def _traverse(path: cm.Path, loc: int, t: int) -> bool:
        if t >= len(path):
            return loc == path[-1].location
        else:
            return t >= 0 and path[t].location == loc


if __name__ == "__main__":
    pass
