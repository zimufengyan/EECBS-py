# -*- coding:utf-8 -*-
# @FileName  :nodes.py
# @Time      :2024/7/19 下午7:27
# @Author    :ZMFY
# Description:

import random
from enum import Enum
from typing import List, Tuple, Union

from common import Path
from conflict import Conflict, Constraint, ConstraintType


class NodeSelection(Enum):
    NODE_RANDOM = 1
    NODE_H = 2
    NODE_DEPTH = 3
    NODE_CONFLICTS = 4
    NODE_CONFLICTPAIRS = 5
    NODE_MVC = 6


class LLNode:
    def __init__(self, location=0, g_val=0, h_val=0, parent=None, timestep=0, num_of_conflicts=0, in_openlist=False):
        self.location = location
        self.g_val = g_val
        self.h_val = h_val
        self.parent: Union[LLNode, None] = parent
        self.timestep = timestep
        self.num_of_conflicts = num_of_conflicts
        self.in_openlist = in_openlist

        # the action is to wait at the goal vertex or not. This is used for >length constraints
        self.wait_at_goal = False
        self.is_goal = False

    @property
    def f_val(self):
        return self.g_val + self.h_val

    def get_f_val(self):
        return self.f_val

    def init_from_other(self, other):
        self.location = other.location
        self.g_val = other.g_val
        self.h_val = other.h_val
        self.parent = other.parent
        self.timestep = other.timestep
        self.num_of_conflicts = other.num_of_conflicts
        self.in_openlist = other.in_openlist
        self.wait_at_goal = other.wait_at_goal
        self.is_goal = other.is_goal

    def __lt__(self, other):
        """used by OPEN (heap) to compare nodes (top of the heap has min f-val, and then highest g-val)"""
        if self.f_val == other.f_val:
            if self.h_val == other.h_val:
                return random.random() > 0.5  # break ties randomly
            # break ties towards smaller h_vals (closer to goal location)
            return self.h_val < other.h_val
        return self.f_val < other.f_val


class HLNode:
    """a virtual base class for high-level node"""

    def __init__(self):
        self.g_val = 0  # sum of costs for CBS, and sum of min f for ECBS
        self.h_val = 0  # admissible h
        self.cost_to_go = 0  # informed but inadmissible h
        self.distance_to_go = 0  # distance to the goal state
        self.depth = 0  # depath of this CT node
        self.makespan = 0  # makespan over all paths
        self.h_computed = False
        self.time_expanded = 0
        self.time_generated = 0

        # for debug
        self.chosen_form = 'none'  # chosen from the open/focal/cleanup least
        self.f_of_best_in_cleanup = 0
        self.f_hat_of_best_in_cleanup = 0
        self.d_of_best_in_cleanup = 0
        self.f_of_best_in_open = 0
        self.f_hat_of_best_in_open = 0
        self.d_of_best_in_open = 0
        self.f_of_best_in_focal = 0
        self.f_hat_of_best_in_focal = 0
        self.d_of_best_in_focal = 0

        # conflicts in the current paths
        self.conflicts: List[Conflict] = []
        self.constraints: List[Constraint] = []
        self.unknown_conf: List[Conflict] = []

        # the chosen conflict
        self.conflict: Conflict = None

        # online learning
        self.ditance_error = 0
        self.cost_error = 0
        self.fully_expanded = False

        self.parent: Union[HLNode, None] = None
        self.children: List[HLNode] = []

    @property
    def f_val(self):
        return self.g_val + self.h_val

    @property
    def f_hat_val(self):
        raise NotImplementedError

    @property
    def num_new_paths(self):
        raise NotImplementedError

    def get_replanned_agents(self):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def update_distance_to_go(self):
        pass

    def print_constraints(self, idx: int):
        pass


class CBSNode(HLNode):
    def __init__(self):
        super().__init__()
        self.paths: List[Tuple[int, Path]] = []

    def get_name(self):
        return "CBS Node"

    def get_num_new_paths(self):
        return len(self.paths)

    def get_f_hat_val(self):
        return self.g_val + self.cost_to_go

    def get_replanned_agents(self):
        rst: List[int] = []
        for path in self.paths:
            rst.append(path[0])
        return rst

    def __lt__(self, other):
        """used to compare nodes in the OPEN list"""
        if self.get_f_hat_val() == other.get_f_hat_val():
            if self.get_f_val() == other.get_f_val():
                if self.distance_to_go == other.distance_to_go:
                    return self.h_val < other.h_val
                return self.distance_to_go < other.distance_to_go
            return self.get_f_val() < other.get_f_val()
        return self.get_f_hat_val() < other.get_f_hat_val()


class ConstraintsHasher:
    def __init__(self, a, n: HLNode):
        self.a: int = a
        self.n: HLNode = n

    def __eq__(self, other):
        if self.a != other.a:
            return False

        cons1, cons2 = set(), set()
        curr = self.n
        while curr.parent is not None:
            if (curr.constraints[0].flag == ConstraintType.LEQLENGTH
                    or curr.constraints[0].flag == ConstraintType.POSITIVE_VERTEX
                    or curr.constraints[0].flag == ConstraintType.POSITIVE_EDGE
                    or curr.constraints[0].agent == self.a):
                for con in curr.constraints:
                    cons1.add(con)
            curr = curr.parent

        curr = other.n
        while curr.parent is not None:
            if (curr.constraints[0].flag == ConstraintType.LEQLENGTH
                    or curr.constraints[0].flag == ConstraintType.POSITIVE_VERTEX
                    or curr.constraints[0].flag == ConstraintType.POSITIVE_EDGE
                    or curr.constraints[0].agent == other.a):
                for con in curr.constraints:
                    cons2.add(con)
            curr = curr.parent

        return cons1 == cons2

    def __hash__(self):
        curr = self.n
        cons_hash = 0
        while curr.parent is not None:
            if (curr.constraints[0].flag == ConstraintType.LEQLENGTH
                    or curr.constraints[0].flag == ConstraintType.POSITIVE_VERTEX
                    or curr.constraints[0].flag == ConstraintType.POSITIVE_EDGE
                    or curr.constraints[0].agent == self.a):
                for con in curr.constraints:
                    cons_hash += (3 * con.agent + 5 * con.loc1 + 7 * con.loc2 + 11 * con.t)
            curr = curr.parent

        return hash(cons_hash)


class AstarNode(LLNode):
    def __init__(self, loc: int, g_val: int, h_val: int, parent: Union[LLNode, None],
                 timestep: int, num_of_conflicts=0, in_openlist=False):
        super().__init__(loc, g_val, h_val, parent, timestep, num_of_conflicts, in_openlist)

    def __hash__(self):
        loc_hash = hash(self.location)
        time_hash = hash(self.timestep)
        return hash(loc_hash ^ (time_hash << 1))

    def __eq__(self, other):
        return (other is not None and self.location == other.location
                and self.timestep == other.timestep
                and self.wait_at_goal == other.wait_at_goal)

    def get_hash_key(self):
        return self.location ^ (self.timestep << 1) + int(self.wait_at_goal)


class MDDNode:
    def __init__(self, curr_loc: int, t: int = 0, parent=None):
        self.location = curr_loc
        self.parent: Union[MDDNode, None] = parent
        self.level = t if parent is None else parent.level + 1
        self.parents: List[MDDNode] = [] if parent is None else [parent]
        self.children: List[MDDNode] = []
        self.cost = 0  # minimum cost of path traversing this MDD node

    def __eq__(self, other):
        return self.location == other.location and self.level == other.level


class SyncMDDNode:
    def __init__(self, curr_loc: int, parent=None):
        self.parents: List[SyncMDDNode] = []
        self.children: List[SyncMDDNode] = []
        if parent is not None:
            self.parents.append(parent)
        self.location = curr_loc
        self.coexisting_nodes_from_other_mdds: List[MDDNode] = []

    def __eq__(self, other):
        return self.location == other.location


class ECBSNode(HLNode):
    def __init__(self):
        super().__init__()
        self.sum_of_costs = 0  # sum of costs of the paths
        self.paths: List[Tuple[int, Path]] = []  # new paths <idx id, <path, min f>>

    @property
    def f_hat_val(self):
        return self.sum_of_costs + self.cost_to_go

    def get_name(self):
        return "ECBS Node"

    @property
    def num_new_paths(self):
        return len(self.paths)

    def get_replanned_agents(self) -> List[int]:
        rst = []
        for path in self.paths:
            rst.append(path[0])
        return rst

    def __lt__(self, other):
        """used to compare nodes in the OPEN list"""
        if self.sum_of_costs + self.cost_to_go == other.sum_of_costs + other.cost_to_go:
            if self.f_val == other.f_val:
                if self.distance_to_go == other.distance_to_go:
                    return self.h_val < other.h_val
                return self.distance_to_go < other.distance_to_go
            return self.f_val < other.f_val
        return self.sum_of_costs < other.sum_of_costs


if __name__ == "__main__":
    pass
