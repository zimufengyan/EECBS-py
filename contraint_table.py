# -*- coding:utf-8 -*-
# @FileName  :contraint_table.py
# @Time      :2024/7/19 下午6:44
# @Author    :ZMFY
# Description:

from typing import List, Dict, Tuple, Union

from multipledispatch import dispatch

import common as cm
from conflict import Constraint, ConstraintType
from nodes import CBSNode, HLNode


class ConstraintTable:
    def __init__(self, num_col: int, map_size: int):
        self.num_col = num_col
        self.map_size = map_size

        self.length_min = 0
        self.length_max = cm.MAX_TIMESTEP

        # constraint table, location -> time range, or edge -> time range
        self.ct: Dict[int, List[Tuple[int, int]]] = dict()
        self.ct_max_timestep = 0
        self.cat: List[List[bool]] = []  # conflict avoidance table
        self.cat_max_timestep = 0
        self.cat_goals: List[int] = []

        # <timestep, location>: the agent must be at the given location at the given timestep
        self.landmarks: Dict[int, int] = dict()

    def clear(self):
        self.ct.clear()
        self.landmarks.clear()
        self.cat_goals.clear()

    def _get_edge_index(self, src: int, tgt: int) -> int:
        return (1 + src) * self.map_size + tgt

    def _decode_barrier(self, x: int, y: int, t: int) -> List[Tuple[int, int]]:
        """return the location-time pairs on the barrier in an increasing order of their timesteps"""
        rst: List[Tuple[int, int]] = []
        x1, y1 = x // self.num_col, x % self.num_col
        x2, y2 = y // self.num_col, y % self.num_col
        if x1 == x2:
            for i in range(min(abs(y2 - y1), t), -1, -1):
                rst.append((x1 * self.num_col + y2 - i, t - i))
        else:  # y1 == y2
            for i in range(min(abs(x2 - x1), t), -1, -1):
                rst.append(((x2 - i) * self.num_col + y1, t - i))

        return rst

    def _insert_landmark(self, loc: int, t: int):
        """insert a landmark, i.e., the agent has to be at the given location at the given timestep"""
        it = self.landmarks.get(t)
        if it is None:
            self.landmarks[t] = loc
        else:
            assert it == loc

    def get_holding_time(self, location: int, earliest_timestep: int) -> int:
        """the earliest timestep that the agent can hold the location after earliest_timestep"""
        rst = earliest_timestep
        it = self.ct.get(location, None)
        if it is not None:
            for time_range in it:
                rst = max(rst, time_range[1])

        for landmark, loc in self.landmarks.items():
            if loc != location:
                rst = max(rst, int(landmark) + 1)

        return rst

    def get_max_timestep(self) -> int:
        """everything is static after the max timestep"""
        rst = max(max(self.ct_max_timestep, self.cat_max_timestep), self.length_min)
        if self.length_max < cm.MAX_TIMESTEP:
            rst = max(rst, self.length_max)
        if not len(self.landmarks) == 0:
            rst = max(rst, self.landmarks[-1])
        return rst

    def get_last_collision_timestep(self, location: int) -> int:
        rst = -1
        if not len(self.cat):
            for t in range(len(self.cat[location]) - 1, rst, -1):
                if self.cat[location][t]:
                    return t

        return -1

    @dispatch(int, int)
    def constrained(self, loc: int, t: int) -> bool:
        assert loc >= 0
        if loc < self.map_size:
            it = self.landmarks.get(t, None)
            if it is not None and it != loc:
                # violate the positive vertex constraint
                return True

        it = self.ct.get(loc, None)
        if it is None:
            return False
        for constraint in it:
            if constraint[0] <= t < constraint[1]:
                return True
        return False

    @dispatch(int, int, int)
    def constrained(self, curr_loc: int, next_loc: int, next_t: int) -> bool:
        return self.constrained(self._get_edge_index(curr_loc, next_loc), next_t)

    def get_num_of_conflicts_for_step(self, curr_id: int, next_id: int, next_timestep) -> int:
        rst = 0
        if len(self.cat) != 0:
            if len(self.cat[next_id]) > next_timestep and self.cat[next_id][next_timestep]:
                rst += 1
            if curr_id != next_id and len(self.cat[next_id]) >= next_timestep and len(
                    self.cat[curr_id]) > next_timestep:
                rst += 1
            if self.cat_goals[next_id] < next_timestep:
                rst += 1

        return rst

    def has_conflict_for_step(self, curr_id: int, next_id: int, next_timestep) -> bool:
        if not len(self.cat) == 0:
            if len(self.cat[next_id]) > next_timestep and self.cat[next_id][next_timestep]:
                return True
            if (curr_id != next_id and len(self.cat[next_id]) >= next_timestep
                    and len(self.cat[curr_id]) > next_timestep
                    and self.cat[next_id][next_timestep - 1]
                    and self.cat[curr_id][next_timestep]):
                return True
            if self.cat_goals[next_id] < next_timestep:
                return True
        return False

    def has_edge_conflict(self, curr_id: int, next_id: int, next_timestep) -> bool:
        assert curr_id != next_id
        return (len(self.cat) != 0 and curr_id != next_id
                and len(self.cat[next_id]) >= next_timestep
                and len(self.cat[curr_id]) > next_timestep
                and self.cat[next_id][next_timestep - 1]
                and self.cat[curr_id][next_timestep])

    def get_future_num_of_collisions(self, loc: int, t: int) -> int:
        rst = 0
        if len(self.cat) != 0:
            for timestep in range(t + 1, len(self.cat[loc])):
                rst += int(self.cat[loc][timestep])
        return rst

    def init_from_other(self, other):
        self.length_min = other.length_min
        self.length_max = other.length_max
        self.num_col = other.num_col
        self.map_size = other.map_size
        self.ct = other.ct
        self.ct_max_timestep = other.ct_max_timestep
        self.cat = other.cat
        self.cat_goals = other.cat_goals
        self.cat_max_timestep = other.cat_max_timestep
        self.landmarks = other.landmarks

    def insert_node_to_ct(self, node: Union[HLNode, CBSNode], agent: id):
        """build the constraint table for the given agent at the give node"""
        curr = node
        while curr.parent is not None:
            self.insert_constraints_to_ct(curr.constraints, agent)
            curr = curr.parent

    def insert_constraints_to_ct(self, constraints: List[Constraint], agent: id):
        """insert constraints for the given agent to the constraint table"""
        if len(constraints) == 0:
            return
        a, x, y, t, con_type = constraints[0]
        if con_type == ConstraintType.LEQLENGTH:
            assert len(constraints) == 1
            if agent == a:
                # this agent has to reach its goal at or before timestep t.
                self.length_max = min(self.length_max, t)
            else:
                # other agents cannot stay at x at or after timestep t
                self.insert_vc_to_ct(x, t, cm.MAX_TIMESTEP)
        elif con_type == ConstraintType.GLENGTH:
            assert len(constraints) == 1
            if agent == a:
                #  this agent has to be at x at timestep t
                self._insert_landmark(x, t)
            else:
                # other agents cannot stay at x at timestep t
                self.insert_vc_to_ct(x, t, t + 1)
        elif con_type == ConstraintType.POSITIVE_EDGE:
            assert len(constraints) == 1
            if agent == a:
                # this agent has to be at x at timestep t - 1 and be at y at timestep t
                self._insert_landmark(x, t - 1)
                self._insert_landmark(y, t)
            else:
                # other agents cannot stay at x at timestep t - 1, be at y at timestep t,
                # or traverse edge (y, x) from timesteps t - 1 to t
                self.insert_vc_to_ct(x, t - 1, t)
                self.insert_vc_to_ct(y, t, t + 1)
                self.insert_ec_to_ct(y, x, t, t + 1)
        elif con_type == ConstraintType.VERTEX:
            if agent == a:
                for constraint in constraints:
                    a, x, y, t, con_type = constraint
                    self.insert_vc_to_ct(x, t, t + 1)
        elif con_type == ConstraintType.EDGE:
            assert len(constraints) == 1
            if agent == a:
                self.insert_ec_to_ct(x, y, t, t + 1)
        elif con_type == ConstraintType.BARRIER:
            if agent == a:
                for constraint in constraints:
                    a, x, y, t, con_type = constraint
                    states = self._decode_barrier(x, y, t)
                    for state in states:
                        self.insert_vc_to_ct(state[0], state[1], state[1] + 1)
        elif con_type == ConstraintType.RANGE:
            assert len(constraints) == 1
            if agent == a:
                # the agent cannot stay at x from timestep y to timestep t.
                self.insert_vc_to_ct(x, y, t + 1)

    def insert_path_to_ct(self, path: cm.Path):
        """insert a path to the constraint table"""
        prev_location = path[0].location
        prev_timestep = 0
        for timestep in range(len(path)):
            curr_location = path[timestep].location
            if prev_location != curr_location:
                self.insert_vc_to_ct(prev_location, prev_timestep, timestep)  # add vertex conflict
                self.insert_ec_to_ct(curr_location, prev_location, timestep, timestep + 1)  # add edge conflict
                prev_location = curr_location
                prev_timestep = timestep
        self.insert_vc_to_ct(path[-1].location, len(path) - 1, cm.MAX_TIMESTEP)

    def insert_vc_to_ct(self, loc: int, t_min: int, t_max: int):
        """insert a vertex constraint to the constraint table"""
        assert loc >= 0
        self.ct[loc].append((t_min, t_max))
        if cm.MAX_TIMESTEP > t_max > self.ct_max_timestep:
            self.ct_max_timestep = t_max
        elif t_max == cm.MAX_TIMESTEP and t_min == self.ct_max_timestep:
            self.ct_max_timestep = t_min

    def insert_ec_to_ct(self, src: int, tgt: int, t_min: int, t_max: int):
        """insert an edge constraint to the constraint table"""
        self.insert_vc_to_ct(self._get_edge_index(src, tgt), t_min, t_max)

    def insert_paths_to_cat(self, agent: int, paths: List[cm.Path]):
        """build the conflict avoidance table using a set of paths"""
        for ag in range(len(paths)):
            if ag == agent or paths[ag] is None:
                continue
            self.insert_path_to_ct(paths[ag])

    def insert_path_to_cat(self, path: cm.Path):
        """insert a path to the conflict avoidance table"""
        if len(self.cat) == 0:
            self.cat: List[List[bool]] = [[] for _ in range(self.map_size)]
            self.cat_goals = [cm.MAX_TIMESTEP for _ in range(self.map_size)]
        assert self.cat_goals[path[-1].location] == cm.MAX_TIMESTEP
        self.cat_goals[path[-1].location] = len(path) - 1
        for timestep in range(len(path)):
            loc = path[timestep].location
            self.cat[loc].append(True)
        self.cat_max_timestep = max(self.cat_max_timestep, len(path) - 1)


if __name__ == "__main__":
    pass
