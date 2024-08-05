# -*- coding:utf-8 -*-
# @FileName  :single_agent_solver.py
# @Time      :2024/7/19 下午8:10
# @Author    :ZMFY
# Description:

import numpy as np
from typing import List, Tuple, Dict
import heapq as hpq

from instance import Instance
from nodes import LLNode, HLNode
import common as cm
from constraint_table import ConstraintTable


class SingleAgentSolver:
    def __init__(self, instance: Instance, agent: int):
        self.instance = instance
        self.agent = agent
        self.start_location = int(instance.start_locations[agent])
        self.goal_location = int(instance.goal_locations[agent])

        self.runtime_build_ct = 0
        self.runtime_build_cat = 0
        self.num_expanded = 0
        self.num_generated = 0
        self.my_heuristic = np.ones(self.instance.map_size, dtype=int) * cm.MAX_TIMESTEP

        self.min_f_val = 0  # minimal f value in OPEN
        self.w = 1          # suboptimal bound

        self.open_list: cm.PrioritySet = cm.PrioritySet()
        self.focal_list = cm.PrioritySet()

        self._compute_heuristics()

    @staticmethod
    def get_name():
        raise NotImplementedError

    def get_travel_time(self, start: int, end: int, constraint_table: ConstraintTable, upper_bound: int):
        raise NotImplementedError

    def find_optimal_path(self, node: HLNode, initial_constraint: ConstraintTable,
                          paths: List[cm.Path], agent:int, lower_bound: int):
        raise NotImplementedError

    def find_suboptimal_path(self, node: HLNode, initial_constraint: ConstraintTable,
                             paths: List[cm.Path], agent:int, lower_bound: int, w: float) -> Tuple[cm.Path, int]:
        raise NotImplementedError

    def get_next_locations(self, curr: int) -> List[int]:
        # including itself and its neighbors
        rst = self.instance.get_neighbors(curr)
        rst.append(curr)
        return rst

    def get_neighbors(self, curr: int) -> List[int]:
        return self.instance.get_neighbors(curr)

    def _get_dh_heuristic(self, src: int, tgt: int) -> int:
        return abs(self.my_heuristic[src] - self.my_heuristic[tgt])

    def compute_heuristic(self, src: int, tgt: int) -> int:
        return max(self._get_dh_heuristic(src, tgt), self.instance.get_manhattan_distance(src, tgt))

    def _compute_heuristics(self):
        class Node:
            def __init__(self, location, value):
                self.location = location
                self.value = value

            def __lt__(self, other):
                return self.value < other.value

        if len(self.my_heuristic) != self.instance.map_size:
            self.my_heuristic = np.ones(self.instance.map_size, dtype=int) * cm.MAX_TIMESTEP

        # generate a heap that can save nodes
        root = Node(self.goal_location, 0)
        self.my_heuristic[self.goal_location] = 0

        heap = []
        hpq.heappush(heap, root)

        while heap:
            curr = hpq.heappop(heap)
            for next_location in self.instance.get_neighbors(curr.location):
                if self.my_heuristic[next_location] > curr.value + 1:
                    self.my_heuristic[next_location] = curr.value + 1
                    next_node = Node(next_location, curr.value + 1)
                    hpq.heappush(heap, next_node)

    def reset(self):
        self.num_expanded = 0
        self.num_generated = 0

    def pop_node(self):
        node = self.focal_list.pop()
        self.open_list.remove(node)
        node.in_openlist = False
        self.num_expanded += 1
        return node

    def _update_focal_list(self):
        raise NotImplementedError

    @staticmethod
    def _update_path(goal, path: List[cm.PathEntry]):
        raise NotImplementedError


if __name__ == "__main__":
    pass
