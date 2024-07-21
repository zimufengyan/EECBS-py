# -*- coding:utf-8 -*-
# @FileName  :conflict.py
# @Time      :2024/7/19 下午6:43
# @Author    :ZMFY
# Description:

from typing import List, Dict, Set, Tuple, Union
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

import common as cm


class ConflictType(Enum):
    MUTEX = 1
    TARGET = 2
    CORRIDOR = 3
    RECTANGLE = 4
    STANDARD = 5
    TYPE_COUNT = 6


class ConstraintType(Enum):
    LEQLENGTH = 1
    GLENGTH = 2
    RANGE = 3
    BARRIER = 4
    VERTEX = 5
    EDGE = 6
    POSITIVE_VERTEX = 7
    POSITIVE_EDGE = 8
    CONSTRAINT_COUNT = 9


class ConflictPriority(Enum):
    CARDINAL = 1
    PSEUDO_CARDINAL = 2
    SEMI = 3
    NON = 4
    UNKNOWN = 5
    PRIORITY_COUNT = 6


class ConflictSelection(Enum):
    RANDOM = 1
    EARLIEST = 2
    CONFLICTS = 3
    MCONSTRAINTS = 4
    FCONSTRAINTS = 5
    WIDTH = 6
    SINGLETONS = 7


@dataclass
class Constraint:
    agent: int
    loc1: int
    loc2: int
    t: int
    flag: ConstraintType

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

    def __lt__(self, other):
        """return true if self has lower priority"""
        if self.priority == other.priority:
            if self.con_type == other.con_type:
                if self.secondary_priority == other.secondary_priority:
                    return random.randint(1, 100000) % 2
                return self.secondary_priority < other.secondary_priority
            return self.con_type.value < other.con_type.value
        return self.priority < other.priority

    def get_conflict_id(self):
        return self.con_type.value

    def vertex_conflict(self, a1: int, a2: int, v: int, t: int):
        self.constraint1.clear()
        self.constraint2.clear()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v, -1, t, ConstraintType.VERTEX))
        self.constraint2.append(Constraint(a2, v, -1, t, ConstraintType.VERTEX))
        self.con_type = ConflictType.STANDARD

    def edge_conflict(self, a1: int, a2: int, v1: int, v2: int, t: int):
        self.constraint1.clear()
        self.constraint2.clear()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v1, v2, t, ConstraintType.EDGE))
        self.constraint2.append(Constraint(a2, v2, v1, t, ConstraintType.EDGE))
        self.con_type = ConflictType.STANDARD

    def corridor_conflict(self, a1: int, a2: int, v1: int, v2: int, t1: int, t2: int):
        self.constraint1.clear()
        self.constraint2.clear()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v1, 0, t1, ConstraintType.RANGE))
        self.constraint2.append(Constraint(a2, v2, 0, t2, ConstraintType.RANGE))
        self.con_type = ConflictType.CORRIDOR

    def rectangle_conflict(self, a1: int, a2: int, constraint1: List[Constraint],
                           constraint2: List[Constraint]) -> bool:
        self.a1 = a1
        self.a2 = a2
        self.constraint1 = deepcopy(constraint1)
        self.constraint2 = deepcopy(constraint2)
        self.con_type = ConflictType.RECTANGLE
        return True

    def target_conflict(self, a1: int, a2: int, v: int, t: int):
        self.constraint1.clear()
        self.constraint2.clear()
        self.a1 = a1
        self.a2 = a2
        self.constraint1.append(Constraint(a1, v, -1, t, ConstraintType.LEQLENGTH))
        self.constraint2.append(Constraint(a1, v, -1, t, ConstraintType.GLENGTH))
        self.con_type = ConflictType.TARGET

    def mutex_conflict(self, a1: int, a2: int):
        self.constraint1.clear()
        self.constraint2.clear()
        self.a1 = a1
        self.a2 = a2
        self.con_type = ConflictType.MUTEX
        self.priority = ConflictPriority.CARDINAL

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
    con = Constraint(1, 1, 1, 1, ConstraintType.RANGE)
    print(con)
