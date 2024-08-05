# -*- coding:utf-8 -*-
# @FileName  :incremental_pairwise_mutex_propagation.py
# @Time      :2024/7/23 下午3:18
# @Author    :ZMFY
# Description:

import numpy as np
from typing import List, Dict, Tuple, Union

from mdd import MDD
from single_agent_solver import SingleAgentSolver
from constraint_table import ConstraintTable
from conflict import Constraint
from constraint_propagation import ConstraintPropagation


class IPMutexPropagation:
    def __init__(self, mdd_0: MDD, mdd_1: MDD, se_0: SingleAgentSolver, se_1: SingleAgentSolver,
                 cons_0: ConstraintTable, cons_1: ConstraintTable, incr_limit=20):
        self.mdd_0 = mdd_0
        self.mdd_1 = mdd_1
        self.search_engine_0 = se_0
        self.search_engine_1 = se_1
        self.cons_0 = cons_0
        self.cons_1 = cons_1
        self.incr_limit = incr_limit

        self.init_len_0 = len(mdd_0.levels)
        self.init_len_1 = len(mdd_1.levels)
        self.final_len_0 = -1
        self.final_len_1 = -1

    def gen_constraints(self) -> Tuple[List[Constraint], List[Constraint]]:
        """similar to pairwise ICTS"""
        pass


if __name__ == "__main__":
    pass
