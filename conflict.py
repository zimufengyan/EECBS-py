# -*- coding:utf-8 -*-
# @FileName  :conflict.py
# @Time      :2024/7/19 下午6:43
# @Author    :ZMFY
# Description:

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

import common as cm


class ConflictType(Enum):
    MUTEX = 0
    TARGET = 1
    CORRIDOR = 2
    RECTANGLE = 3
    STANDARD = 4
    TYPE_COUNT = 5


class ConstraintType(Enum):
    LEQLENGTH = 0
    GLENGTH = 1
    RANGE = 2
    BARRIER = 3
    VERTEX = 4
    EDGE = 5
    POSITIVE_VERTEX = 6
    POSITIVE_EDGE = 7
    CONSTRAINT_COUNT = 8


class ConflictPriority(Enum):
    CARDINAL = 0
    PSEUDO_CARDINAL = 1
    SEMI = 2
    NON = 3
    UNKNOWN = 4
    PRIORITY_COUNT = 5


class ConflictSelection(Enum):
    RANDOM = 0
    EARLIEST = 1
    CONFLICTS = 2
    MCONSTRAINTS = 3
    FCONSTRAINTS = 4
    WIDTH = 5
    SINGLETONS = 6


class Constraint:
    def __init__(self, agent, loc1, loc2, t, flag):
        self.agent: int = agent
        self.loc1: int = loc1
        self.loc2: int = loc2
        self.t: int = t
        self.flag: ConstraintType = flag

    def __iter__(self):
        return iter((self.agent, self.loc1, self.loc2, self.t, self.flag))

    def __repr__(self):
        type_str = ""
        flag = self.flag
        if flag == ConstraintType.VERTEX:
            type_str = "V"
        elif flag == ConstraintType.POSITIVE_VERTEX:
            type_str = "V+"
        elif flag == ConstraintType.EDGE:
            type_str = "E"
        elif flag == ConstraintType.POSITIVE_EDGE:
            type_str = "E+"
        elif flag == ConstraintType.BARRIER:
            type_str = "B"
        elif flag == ConstraintType.RANGE:
            type_str = "R"
        elif flag == ConstraintType.GLENGTH:
            type_str = "G"
        elif flag == ConstraintType.LEQLENGTH:
            type_str = "L"

        return f"{Constraint.__name__}<{self.agent},{self.loc1},{self.loc2},{self.t},{type_str}>"


class Conflict:
    def __init__(self):
        self.a1: int = 0
        self.a2: int = 0
        self.constraint1: List[Constraint] = []
        self.constraint2: List[Constraint] = []
        self.con_type: ConflictType = ConflictType.MUTEX
        self.priority: ConflictPriority = ConflictPriority.UNKNOWN

        # used as the tie-breaking creteria for conflict selection
        self.secondary_priority: float = 0

    def __iter__(self):
        return iter((self.a1, self.a2, self.constraint1, self.constraint2, self.con_type,
                     self.priority, self.secondary_priority))

    def __lt__(self, other):
        """return true if self has lower priority"""
        if self.priority == other.priority:
            if self.con_type == other.con_type:
                if self.secondary_priority == other.secondary_priority:
                    return random.random() < 0.5
                return self.secondary_priority <= other.secondary_priority
            return self.con_type.value <= other.con_type.value
        return self.priority.value <= other.priority.value

    def get_conflict_id(self):
        return self.con_type.value

    def vertex_conflict(self, a1: int, a2: int, v: int, t: int):
        self._clear_constrains()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v, -1, t, ConstraintType.VERTEX))
        self.constraint2.append(Constraint(a2, v, -1, t, ConstraintType.VERTEX))
        self.con_type = ConflictType.STANDARD

    def edge_conflict(self, a1: int, a2: int, v1: int, v2: int, t: int):
        self._clear_constrains()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v1, v2, t, ConstraintType.EDGE))
        self.constraint2.append(Constraint(a2, v2, v1, t, ConstraintType.EDGE))
        self.con_type = ConflictType.STANDARD

    def corridor_conflict(self, a1: int, a2: int, v1: int, v2: int, t1: int, t2: int):
        self._clear_constrains()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v1, 0, t1, ConstraintType.RANGE))
        self.constraint2.append(Constraint(a2, v2, 0, t2, ConstraintType.RANGE))
        self.con_type = ConflictType.CORRIDOR

    def rectangle_conflict(self, a1: int, a2: int, constraint1: List[Constraint],
                           constraint2: List[Constraint]) -> bool:
        self.a1 = a1
        self.a2 = a2
        self.constraint1 = constraint1
        self.constraint2 = constraint2
        self.con_type = ConflictType.RECTANGLE
        return True

    def target_conflict(self, a1: int, a2: int, v: int, t: int):
        self._clear_constrains()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v, -1, t, ConstraintType.LEQLENGTH))
        self.constraint2.append(Constraint(a1, v, -1, t, ConstraintType.GLENGTH))
        self.con_type = ConflictType.TARGET

    def mutex_conflict(self, a1: int, a2: int):
        self._clear_constrains()
        self.a1 = a1
        self.a2 = a2
        self.con_type = ConflictType.MUTEX
        self.priority = ConflictPriority.CARDINAL

    def _clear_constrains(self):
        self.constraint1.clear()
        self.constraint2.clear()

    def __repr__(self):
        priority_str = ""
        if self.priority == ConflictPriority.CARDINAL:
            priority_str = "cardinal "
        elif self.priority == ConflictPriority.PSEUDO_CARDINAL:
            priority_str = "pseudo-cardinal "
        elif self.priority == ConflictPriority.SEMI:
            priority_str = "semi-cardinal "
        elif self.priority == ConflictPriority.NON:
            priority_str = "non-cardinal "

        type_str = ""
        if self.con_type == ConflictType.STANDARD:
            type_str = "standard"
        elif self.con_type == ConflictType.RECTANGLE:
            type_str = "rectangle"
        elif self.con_type == ConflictType.CORRIDOR:
            type_str = "corridor"
        elif self.con_type == ConflictType.TARGET:
            type_str = "target"
        elif self.con_type == ConflictType.MUTEX:
            type_str = "mutex"

        constraint1_str = ",".join(map(str, self.constraint1))
        constraint2_str = ",".join(map(str, self.constraint2))

        return (f"{priority_str}{type_str} conflict: {self.a1} with {constraint1_str} and "
                f"{self.a2} with {constraint2_str}")


if __name__ == "__main__":
    agent, x, y, t, flag = Constraint(1, 1, 1, 1, ConstraintType.RANGE)
    print(agent, x, y, t, flag)
