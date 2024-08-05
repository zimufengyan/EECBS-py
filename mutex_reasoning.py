# -*- coding:utf-8 -*-
# @FileName  :mutex_reasoning.py
# @Time      :2024/7/29 下午12:47
# @Author    :ZMFY
# Description:
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from copy import deepcopy

from mdd import MDD
from nodes import MDDNode, ConstraintsHasher, CBSNode
from instance import Instance
from incremental_pairwise_mutex_propagation import IPMutexPropagation
from constraint_table import ConstraintTable
from conflict import Conflict, Constraint
from single_agent_solver import SingleAgentSolver
from constraint_propagation import ConstraintPropagation


class MutexReasoning:
    def __init__(self, instance: Instance, initial_constraints: List[ConstraintTable]):
        self.instance = instance
        self.initial_constraints = initial_constraints
        self.search_engines: List[SingleAgentSolver] = []   # used to find (single) agents' paths and mdd
        self.lookup_table: Dict[ConstraintsHasher, Dict[ConstraintsHasher, Conflict]] = dict()
        self.accumulated_runtime = 0

    def run(self, a1: int, a2: int, node: CBSNode, mdd_1: MDD, mdd_2:MDD) -> Optional[Conflict]:
        st = time.perf_counter()
        conflict = self._find_mutex_conflict(a1, a2, node, mdd_1, mdd_2)
        self.accumulated_runtime += time.perf_counter() - st
        return conflict

    def _find_mutex_conflict(self, a1: int, a2: int, node: CBSNode, mdd_1: MDD, mdd_2: MDD) -> Optional[Conflict]:
        cp = ConstraintPropagation(mdd_1, mdd_2)
        cp.init_mutex()
        cp.fwd_mutex_prop()
        if cp.feasible(len(mdd_1.levels) - 1, len(mdd_2.levels) - 1) >= 0:
            return None

        swapped = False
        if a1 > a2:
            a1, a2 = a2, a1
            swapped = True

        c_1 = ConstraintsHasher(a1, node)
        c_2 = ConstraintsHasher(a2, node)

        mutex_conflict = None
        if self.lookup_table.get(c_1, None) is not None:
            mutex_conflict = self.lookup_table[c_1][c_2]

        if mutex_conflict is None:
            # generate constraint
            mutex_conflict = Conflict()
            mutex_conflict.mutex_conflict(a1, a2)

            mdd_1_copy = deepcopy(mdd_1)
            mdd_2_copy = deepcopy(mdd_2)

            ct1 = self.initial_constraints[a1]
            ct2 = self.initial_constraints[a2]
            ct1.insert_node_to_ct(node, a1)
            ct2.insert_node_to_ct(node, a2)

            ip = IPMutexPropagation(mdd_1_copy, mdd_2_copy, self.search_engines[a1], self.search_engines[a2], ct1, ct2)
            a, b = ip.gen_constraints()
            for con in a:
                con.agent = a1
                mutex_conflict.constraint1.append(con)

            for con in b:
                con.agent = a2
                mutex_conflict.constraint2.append(con)

            self.lookup_table[c_1][c_2] = mutex_conflict

        conflict_ret: Conflict = deepcopy(mutex_conflict)

        if swapped:
            conflict_ret.a1, conflict_ret.a2 = conflict_ret.a2, conflict_ret.a1
            conflict_ret.constraint1, conflict_ret.constraint2 = conflict_ret.constraint2, conflict_ret.constraint1

        return conflict_ret


if __name__ == "__main__":
    pass
