# -*- coding:utf-8 -*-
# @FileName  :constraint_propagation.py
# @Time      :2024/7/23 下午3:24
# @Author    :ZMFY
# Description:


import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
from multipledispatch import dispatch

from mdd import MDD, collect_mdd_level
from nodes import MDDNode
from conflict import Constraint, ConstraintType

NodePair = Tuple[Optional[MDDNode], Optional[MDDNode]]
EdgePair = Tuple[NodePair, NodePair]


class ConstraintPropagation:
    def __init__(self, mdd0: MDD, mdd1: MDD):
        self.mdd0 = mdd0
        self.mdd1 = mdd1

        self.fwd_mutexes: Set[EdgePair] = set()
        self.bwd_mutexes: Set[EdgePair] = set()

    @staticmethod
    def _is_edge_mutex(ep: EdgePair):
        return ep[0][1] is not None

    def _should_be_fwd_mutexed_by_node(self, node_a: MDDNode, node_b: MDDNode) -> bool:
        for node_a_from in node_a.parents:
            for node_b_from in node_b.parents:
                if self.has_fwd_mutex_by_nodes(node_b_from, node_a_from):
                    continue
                if self.has_fwd_mutex_by_pair(((node_a_from, node_a), (node_b_from, node_b))):
                    continue
                return False

        return True

    def _should_be_fwd_mutexed_by_edge(self, node_a: MDDNode, node_a_to: MDDNode,
                                       node_b: MDDNode, node_b_to: MDDNode) -> bool:
        pass

    def _should_be_bwd_mutexed_by_node(self, node_a: MDDNode, node_b: MDDNode) -> bool:
        for node_a_to in node_a.children:
            for node_b_to in node_b.children:
                if self.has_mutex_by_nodes(node_b_to, node_a_to):
                    continue
                if self.has_mutex_by_pair(((node_a, node_a_to), (node_b, node_b_to))):
                    continue
                return False

        return True

    def _should_be_bwd_mutexed_by_edge(self, node_a: MDDNode, node_a_to: MDDNode,
                                       node_b: MDDNode, node_b_to: MDDNode) -> bool:
        pass

    def _add_bwd_node_mutex(self, node_a: MDDNode, node_b: MDDNode):
        if self.has_mutex_by_pair(((node_a, node_b), (node_b, node_a))):
            return
        self.bwd_mutexes.add(((node_a, None), (node_b, None)))

    def _add_fwd_node_mutex(self, node_a: MDDNode, node_b: MDDNode):
        if not self.has_fwd_mutex_by_pair(((node_a, None), (node_b, None))):
            return

        self.fwd_mutexes.add(((node_a, None), (node_b, None)))

    def _add_fwd_edge_mutex(self, node_a: MDDNode, node_a_to: MDDNode,
                            node_b: MDDNode, node_b_to: MDDNode):
        if not self.has_fwd_mutex_by_pair(((node_a, node_a_to), (node_b, node_b_to))):
            return

        self.fwd_mutexes.add(((node_a, node_a_to), (node_b, node_b_to)))

    def init_mutex(self):
        num_level = min(len(self.mdd0.levels), len(self.mdd1.levels))

        # node mutex
        for i in range(num_level):
            loc2mdd = collect_mdd_level(self.mdd0, i)
            for it_1 in self.mdd1.levels[i]:
                if loc2mdd.get(it_1.location, None) is not None:
                    self._add_fwd_node_mutex(loc2mdd[it_1.location], it_1)

        # edge mutex
        # loc2mdd_this_level = dict()
        loc2mdd_next_level = collect_mdd_level(self.mdd1, 0)

        for i in range(num_level - 1):
            loc2mdd_this_level = loc2mdd_next_level
            loc2mdd_next_level = collect_mdd_level(self.mdd1, i + 1)
            for node_0 in self.mdd0.levels[i]:
                loc_0 = node_0.location
                if loc2mdd_next_level.get(loc_0, None) is None:
                    continue
                node_1_to = loc2mdd_next_level[loc_0]

                for node_0_to in node_0.children:
                    loc_1 = node_0_to.location
                    if loc2mdd_this_level.get(loc_1, None) is None:
                        continue

                    node_1 = loc2mdd_this_level[loc_1]
                    for ptr in node_1.children:
                        if ptr == node_1_to:
                            self._add_fwd_edge_mutex(node_0, node_0_to, node_1, node_1_to)

    def fwd_mutex_prop(self):
        open_queue = deque(self.fwd_mutexes)

        while open_queue:
            mutex = open_queue.popleft()

            if self._is_edge_mutex(mutex):
                node_to_1 = mutex[0][1]
                node_to_2 = mutex[1][1]

                if self.has_fwd_mutex_by_nodes(node_to_1, node_to_2):
                    continue

                if not self._should_be_fwd_mutexed_by_node(node_to_1, node_to_2):
                    continue

                new_mutex = ((node_to_1, None), (node_to_2, None))

                self.fwd_mutexes.add(new_mutex)
                open_queue.append(new_mutex)
            else:
                # Node mutex
                node_a = mutex[0][0]
                node_b = mutex[1][0]

                # Check their children
                for node_a_ch in node_a.children:
                    for node_b_ch in node_b.children:
                        if self.has_fwd_mutex_by_nodes(node_a_ch, node_b_ch):
                            continue

                        if not self._should_be_bwd_mutexed_by_node(node_a_ch, node_b_ch):
                            continue

                        new_mutex = ((node_a_ch, None), (node_b_ch, None))

                        self.fwd_mutexes.add(new_mutex)
                        open_queue.append(new_mutex)

    def bwd_mutex_prop(self):
        open_queue = deque(self.bwd_mutexes)  # TODO: bwd or fwd ???

        while open_queue:
            mutex = open_queue.popleft()

            if self._is_edge_mutex(mutex):
                node_from_1 = mutex[0][0]
                node_from_2 = mutex[1][0]

                if self.has_mutex_by_nodes(node_from_1, node_from_2):
                    continue

                if not self._should_be_bwd_mutexed_by_node(node_from_1, node_from_2):
                    continue

                new_mutex = ((node_from_1, None), (node_from_2, None))

                self.bwd_mutexes.add(new_mutex)
                open_queue.append(new_mutex)
            else:
                # Node mutex
                node_a = mutex[0][0]
                node_b = mutex[1][0]

                # Check their parents
                for node_a_pa in node_a.parents:
                    for node_b_pa in node_b.parents:
                        if self.has_mutex_by_nodes(node_a_pa, node_b_pa):
                            continue

                        if not self._should_be_bwd_mutexed_by_node(node_a_pa, node_b_pa):
                            continue

                        new_mutex = ((node_a_pa, None), (node_b_pa, None))

                        self.bwd_mutexes.add(new_mutex)
                        open_queue.append(new_mutex)

    def has_mutex_by_pair(self, e: EdgePair):
        return (e[0], e[1]) in self.bwd_mutexes or (e[1], e[0]) in self.bwd_mutexes or self.has_fwd_mutex_by_pair(e)

    def has_mutex_by_nodes(self, a: MDDNode, b: MDDNode):
        return self.has_mutex_by_pair(((a, None), (b, None)))

    def has_fwd_mutex_by_pair(self, e: EdgePair):
        return (e[0], e[1]) in self.fwd_mutexes or (e[1], e[0]) in self.fwd_mutexes

    def has_fwd_mutex_by_nodes(self, a: MDDNode, b: MDDNode):
        return self.has_fwd_mutex_by_pair(((a, None), (b, None)))

    def mutexed(self, level_0: int, level_1: int):
        mdd_s = self.mdd0
        mdd_l = self.mdd1

        if level_0 > level_1:
            level_0, level_1 = level_1, level_0
            mdd_s, mdd_l = mdd_l, mdd_s

        if level_0 > len(mdd_s.levels):
            print("ERROR!")
        if level_1 > len(mdd_l.levels):
            print("ERROR!")

        goal_ptr_i = mdd_s.goal_at(level_0)

        for node in mdd_l.levels[level_0]:
            if node.cost <= level_1 and not self.has_fwd_mutex_by_nodes(goal_ptr_i, node):
                return False

        return True

    def feasible(self, level_0: int, level_1: int):
        return self._feasible(level_0, level_1) < 0

    def _feasible(self, level_0: int, level_1: int) -> int:
        mdd_s = self.mdd0
        mdd_l = self.mdd1

        if level_0 > level_1:
            level_0, level_1 = level_1, level_0
            mdd_s, mdd_l = mdd_l, mdd_s

        if level_0 > len(mdd_s.levels):
            print("ERROR!")
            return -1
        if level_1 > len(mdd_l.levels):
            print("ERROR!")
            return -1

        goal_ptr_i = mdd_s.goal_at(level_0)
        dfs_stack = []  # stack is essentially a list

        for node in mdd_l.levels[level_0]:
            if node.cost <= level_1 and not self.has_fwd_mutex_by_nodes(goal_ptr_i, node):
                dfs_stack.append(node)

        if not dfs_stack:
            return -1

        goal_ptr_j = mdd_l.goal_at(level_1)
        not_allowed_loc = goal_ptr_i.location
        closed = set()

        while dfs_stack:
            ptr = dfs_stack.pop()

            if ptr == goal_ptr_j:
                return 1

            if ptr in closed:
                continue

            closed.add(ptr)

            for child_ptr in ptr.children:
                if child_ptr in closed:
                    continue
                if child_ptr.location == not_allowed_loc:
                    continue
                dfs_stack.append(child_ptr)

        return -2

    def semi_cardinal(self, level: int, loc: int):
        raise NotImplementedError

    def generate_constraints(self, level_0: int, level_1: int) -> Tuple[List[Constraint], List[Constraint]]:
        mdd_s = self.mdd0
        mdd_l = self.mdd1
        reversed = False

        if level_0 > level_1:
            level_0, level_1 = level_1, level_0
            mdd_s, mdd_l = mdd_l, mdd_s
            reversed = True

        goal_ptr_i = mdd_s.goal_at(level_0)

        mutexed = []
        non_mutexed = []
        for node in mdd_l.levels[level_0]:
            if node.cost <= level_1:
                if not self.has_fwd_mutex_by_nodes(goal_ptr_i, node):
                    non_mutexed.append(node)
                else:
                    mutexed.append(node)

        if non_mutexed:
            l = level_0
            cons_set_1 = set()
            level_i = {goal_ptr_i}
            level_j = {node for node in mdd_l.levels[level_0] if node.cost <= level_1}

            for l in range(level_0, -1, -1):
                for ptr_i in level_i:
                    if not any(not self.has_fwd_mutex_by_nodes(ptr_i, ptr_j) for ptr_j in level_j):
                        continue

                for ptr_j in level_j:
                    if not any(not self.has_fwd_mutex_by_nodes(ptr_i, ptr_j) for ptr_i in level_i):
                        cons_set_1.add((l, ptr_j.location))

                level_i_prev = {parent for ptr_i in level_i for parent in ptr_i.parents}
                level_j_prev = {parent for ptr_j in level_j for parent in ptr_j.parents}
                level_i = level_i_prev
                level_j = level_j_prev

            goal_ptr_j = mdd_l.goal_at(level_1)
            not_allowed_loc = goal_ptr_i.location
            closed = set()
            dfs_stack = deque(non_mutexed)

            while dfs_stack:
                ptr = dfs_stack.popleft()

                if ptr == goal_ptr_j:
                    print("ERROR: Non mutexed pair of MDDs")
                    return [], []

                if ptr in closed:
                    continue

                closed.add(ptr)

                for child_ptr in ptr.children:
                    if child_ptr in closed or child_ptr.location == not_allowed_loc:
                        cons_set_1.add((child_ptr.level, child_ptr.location))
                        continue

                    dfs_stack.appendleft(child_ptr)

            length_con = Constraint(0, goal_ptr_i.location, -1, level_0 - 1, ConstraintType.LEQLENGTH)
            cons_vec_1 = [Constraint(1, loc, -1, t, ConstraintType.VERTEX) for t, loc in cons_set_1]
            cons_vec_1.append(length_con)

            if reversed:
                return cons_vec_1, [length_con]
            return [length_con], cons_vec_1

        cons_0, cons_1, blue_0, blue_1 = set(), set(), set(), set()

        for lvl in range(level_0 + 1):
            nodes_i = [node for node in mdd_s.levels[lvl] if node.cost <= level_0]
            nodes_j = [node for node in mdd_l.levels[lvl] if node.cost <= level_1]

            for it_i in nodes_i:
                if all(self.has_fwd_mutex_by_nodes(it_i, it_j) for it_j in nodes_j):
                    blue_0.add(it_i)
                    if any(parent not in blue_0 for parent in it_i.parents):
                        cons_0.add(it_i)

            for it_j in nodes_j:
                if all(self.has_fwd_mutex_by_nodes(it_i, it_j) for it_i in nodes_i):
                    blue_1.add(it_j)
                    if any(parent not in blue_1 for parent in it_j.parents):
                        cons_1.add(it_j)

        cons_vec_0 = [Constraint(0, node.location, -1, node.level, ConstraintType.VERTEX) for node in cons_0]
        cons_vec_1 = [Constraint(1, node.location, -1, node.level, ConstraintType.VERTEX) for node in cons_1]

        if reversed:
            return cons_vec_1, cons_vec_0
        return cons_vec_0, cons_vec_1


if __name__ == "__main__":
    pass
