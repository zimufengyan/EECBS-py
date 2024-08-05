# -*- coding:utf-8 -*-
# @FileName  :reservation_table.py
# @Time      :2024/7/29 下午1:31
# @Author    :ZMFY
# Description:

from typing import List, Tuple

import common as cm
from constraint_table import ConstraintTable

Interval = Tuple[int, int, bool]  # [t_min, t_max), num_of_collisions
Sit = List[List[Interval]]  # location -> [t_min, t_max), num_of_collisions


class ReservationTable:
    def __init__(self, constraint_table: ConstraintTable, goal_location: int):
        self.constraint_table = constraint_table
        self.goal_location = goal_location
        self.sit: Sit = [[] for _ in range(constraint_table.map_size)]

    def _insert_to_sit(self, location: int, t_min: int, t_max: int):
        assert 0 <= t_min < t_max and self.sit[location]
        it = 0
        while it < len(self.sit[location]):
            interval = self.sit[location][it]
            if t_min >= interval[1]:
                it += 1
            elif t_max <= interval[0]:
                break
            elif interval[0] < t_min and interval[1] <= t_max:
                self.sit[location][it] = (interval[0], t_min, interval[2])
                it += 1
            elif t_min <= interval[0] and t_max < interval[1]:
                self.sit[location][it] = (t_max, interval[1], interval[2])
                break
            elif interval[0] < t_min and t_max < interval[1]:
                self.sit[location].insert(it, (interval[0], t_min, interval[2]))
                self.sit[location][it + 1] = (t_max, interval[1], interval[2])
                break
            else:  # t_min <= interval[0] and interval[1] <= t_max
                del self.sit[location][it]

    def _insert_soft_constraint_to_sit(self, location: int, t_min: int, t_max: int):
        assert 0 <= t_min < t_max and self.sit[location]
        it = 0
        while it < len(self.sit[location]):
            interval = self.sit[location][it]
            if t_min >= interval[1] or interval[2]:
                it += 1
            elif t_max <= interval[0]:
                break
            else:
                i_min = interval[0]
                i_max = interval[1]
                if i_min < t_min and i_max <= t_max:
                    if (it < len(self.sit[location]) - 1 and
                            (location != self.goal_location or i_max != self.constraint_table.length_min) and
                            i_max == self.sit[location][it + 1][0] and self.sit[location][it + 1][2]):
                        self.sit[location][it] = (i_min, t_min, False)
                        it += 1
                        self.sit[location][it] = (t_min, self.sit[location][it][1], True)
                    else:
                        self.sit[location].insert(it, (i_min, t_min, False))
                        it += 1
                        self.sit[location][it] = (t_min, i_max, True)
                elif t_min <= i_min and t_max < i_max:
                    if (it > 0 and
                            (location != self.goal_location or i_min != self.constraint_table.length_min) and
                            i_min == self.sit[location][it - 1][1] and self.sit[location][it - 1][2]):
                        self.sit[location][it - 1] = (self.sit[location][it - 1][0], t_max, True)
                    else:
                        self.sit[location].insert(it, (i_min, t_max, True))
                        it += 1
                    self.sit[location][it] = (t_max, i_max, False)
                elif i_min < t_min and t_max < i_max:
                    self.sit[location].insert(it, (i_min, t_min, False))
                    self.sit[location].insert(it + 1, (t_min, t_max, True))
                    it += 2
                    self.sit[location][it] = (t_max, i_max, False)
                else:  # t_min <= interval[0] and interval[1] <= t_max
                    if (it > 0 and
                            (location != self.goal_location or i_min != self.constraint_table.length_min) and
                            i_min == self.sit[location][it - 1][1] and self.sit[location][it - 1][2]):
                        if (it < len(self.sit[location]) - 1 and
                                (location != self.goal_location or i_max != self.constraint_table.length_min) and
                                i_max == self.sit[location][it + 1][0] and self.sit[location][it + 1][2]):
                            self.sit[location][it - 1] = (
                                self.sit[location][it - 1][0], self.sit[location][it + 1][1], True)
                            del self.sit[location][it + 1]
                            del self.sit[location][it]
                        else:
                            self.sit[location][it - 1] = (self.sit[location][it - 1][0], i_max, True)
                            del self.sit[location][it]
                        it -= 1
                    else:
                        if (it < len(self.sit[location]) - 1 and
                                (location != self.goal_location or i_max != self.constraint_table.length_min) and
                                i_max == self.sit[location][it + 1][0] and self.sit[location][it + 1][2]):
                            self.sit[location][it] = (i_min, self.sit[location][it + 1][1], True)
                            del self.sit[location][it + 1]
                        else:
                            self.sit[location][it] = (i_min, i_max, True)
                        it += 1

    def _update_sit(self, location):
        # update SIT at the given location
        assert not self.sit[location]
        max_timestep = cm.MAX_TIMESTEP

        # 目标位置的长度约束
        if location == self.goal_location:
            if self.constraint_table.length_min > self.constraint_table.length_max:
                self.sit[location].append((0, 0, False))
                return
            if 0 < self.constraint_table.length_min:
                self.sit[location].append((0, self.constraint_table.length_min, False))
            assert self.constraint_table.length_min >= 0
            self.sit[location].append(
                (self.constraint_table.length_min, min(self.constraint_table.length_max + 1, max_timestep), False))
        else:
            self.sit[location].append((0, min(self.constraint_table.length_max, max_timestep - 1) + 1, False))

        # 负约束
        if location in self.constraint_table.ct:
            for time_range in self.constraint_table.ct[location]:
                self._insert_to_sit(location, time_range[0], time_range[1])

        # 正约束
        if location < self.constraint_table.map_size:
            for landmark_k, landmark_v in self.constraint_table.landmarks.items():
                if landmark_v != location:
                    self._insert_to_sit(location, landmark_k, landmark_k + 1)

        # 软约束
        if len(self.constraint_table.cat) != 0:
            for t in range(len(self.constraint_table.cat[location])):
                if self.constraint_table.cat[location][t]:
                    self._insert_soft_constraint_to_sit(location, t, t + 1)
            if self.constraint_table.cat_goals[location] < max_timestep:
                self._insert_soft_constraint_to_sit(
                    location, self.constraint_table.cat_goals[location].item(), max_timestep + 1)

    def _get_earliest_arrival_time(self, src: int, tgt: int, lower_bound: int, upper_bound: int) -> int:
        for t in range(lower_bound, upper_bound):
            if not self.constraint_table.constrained(src, tgt, t):
                return t

        return -1

    def _get_earliest_no_collision_arrival_time(self, src: int, tgt: int, interval: Interval,
                                                lower_bound: int, upper_bound: int) -> int:
        for t in range(max(lower_bound, interval[0]), min(upper_bound, interval[1])):
            if not self.constraint_table.has_edge_conflict(src, tgt, t):
                return t

        return -1

    def get_safe_intervals(self, from_location, to_location, lower_bound: int,
                           upper_bound: int) -> List[Tuple[int, int, int, bool, bool]]:
        """return <upper_bound, low, high,  vertex collision, edge collision>"""
        result: List[Tuple[int, int, int, bool, bool]] = []
        if lower_bound >= upper_bound:
            return result

        if not self.sit[to_location]:
            self._update_sit(to_location)

        for interval in self.sit[to_location]:
            if lower_bound >= interval[1]:
                continue
            elif upper_bound <= interval[0]:
                break

            # the interval overlaps with [lower_bound, upper_bound)
            t1 = self._get_earliest_arrival_time(
                from_location, to_location,
                max(lower_bound, interval[0]), min(upper_bound, interval[1])
            )
            if t1 < 0:  # the interval is not reachable
                continue
            elif interval[2]:  # the interval has collisions
                result.append((interval[1], t1, interval[1], True, False))
            else:  # the interval does not have collisions
                # Check if the move action has collisions or not
                t2 = self._get_earliest_no_collision_arrival_time(from_location, to_location, interval, t1, upper_bound)
                if t1 == t2:
                    result.append((interval[1], t1, interval[1], False, False))
                elif t2 < 0:
                    result.append((interval[1], t1, interval[1], False, True))
                else:
                    result.append((interval[1], t1, t2, False, True))
                    result.append((interval[1], t2, interval[1], False, False))

        return result

    def get_first_safe_interval(self, location) -> Interval:
        if len(self.sit[location]) == 0:
            self._update_sit(location)
        return self.sit[location][0]

    def find_safe_interval(self, interval: Interval, location: int, t_min: int) -> Tuple[Interval, bool]:
        """find a safe interval with t_min as given"""
        if t_min >= min(self.constraint_table.length_max, cm.MAX_TIMESTEP - 1) + 1:
            return interval, False
        if len(self.sit[location]) == 0:
            self._update_sit(location)

        for i in self.sit[location]:
            if i[0] <= t_min < i[1]:
                interval = (t_min, i[1], i[2])
                return interval, True
            elif t_min < i[0]:
                break

        return interval, False


if __name__ == "__main__":
    pass
