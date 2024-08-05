# -*- coding:utf-8 -*-
# @FileName  :sipp.py
# @Time      :2024/8/3 下午1:03
# @Author    :ZMFY
# Description:
import time
from copy import deepcopy
from typing import Tuple, List, Dict

import common as cm
from constraint_table import ConstraintTable
from instance import Instance
from nodes import SIPPNode, HLNode
from reservation_table import ReservationTable
from single_agent_solver import SingleAgentSolver


class SIPP(SingleAgentSolver):
    def __init__(self, instance: Instance, agent: int):
        super().__init__(instance, agent)
        self.all_nodes_table: Dict[SIPPNode, List[SIPPNode]] = dict()
        self.useless_nodes: List[SIPPNode] = []

    @staticmethod
    def get_name():
        return "SIPP"

    def find_optimal_path(self, node: HLNode, initial_constraint: ConstraintTable,
                          paths: List[cm.Path], agent: int, lower_bound: int):
        return self.find_suboptimal_path(node, initial_constraint, paths, agent, lower_bound, 1)[0]

    def find_suboptimal_path(self, node: HLNode, initial_constraint: ConstraintTable,
                             paths: List[cm.Path], agent: int, lower_bound: int, w: float) -> Tuple[cm.Path, int]:
        """
        return the path and the lowerbound
        find path by SIPP
        Returns a shortest path that satisfies the constraints of the give node  while
        minimizing the number of internal conflicts (that is conflicts with known_paths for other agents found so far).
        lowerbound is an underestimation of the length of the path in order to speed up the search.
        """
        self.reset()
        self.w = w

        # build constraint table
        st = time.perf_counter()
        constraint_table = deepcopy(initial_constraint)
        constraint_table.insert_node_to_ct(node, agent)
        self.runtime_build_ct = time.perf_counter() - st
        holding_time = constraint_table.get_holding_time(self.goal_location, constraint_table.length_min)
        st = time.perf_counter()
        constraint_table.insert_paths_to_cat(agent, paths)
        self.runtime_build_cat = time.perf_counter() - st

        # build reservation table
        reservation_table = ReservationTable(constraint_table, self.goal_location)

        path: cm.Path = []
        self.reset()
        interval = reservation_table.get_first_safe_interval(self.start_location)
        if interval[0] > 0:
            return path, 0

        # generate start and add it to the OPEN list
        start = SIPPNode(self.start_location, 0, max(self.my_heuristic[self.start_location], holding_time),
                         None, 0, interval[1], interval[1], interval[2], interval[2])
        self.min_f_val = max(holding_time, max(int(start.f_val), lower_bound))
        self._push_node_to_open_and_focal(start)
        iterations = 0

        while len(self.open_list) > 0:
            iterations += 1
            self._update_focal_list()  # update FOCAL if min f-val increased
            curr: SIPPNode = self.pop_node()

            # check if the popped node is a goal node
            if curr.location == self.goal_location \
                    and not curr.wait_at_goal and curr.timestep >= holding_time:
                self._update_path(curr, path)
                break

            # 遍历当前位置的邻居
            for next_location in self.instance.get_neighbors(curr.location):
                # 移动到相邻的位置
                for i in reservation_table.get_safe_intervals(curr.location, next_location, curr.timestep + 1,
                                                              curr.high_expansion + 1):
                    next_high_generation, next_timestep, next_high_expansion, next_v_collision, next_e_collision = i
                    # 计算通过当前节点到达下一个位置的代价
                    next_g_val = next_timestep
                    next_h_val = max(self.my_heuristic[next_location], curr.f_val - next_g_val)  # 路径最大值
                    if next_g_val + next_h_val > reservation_table.constraint_table.length_max:
                        continue
                    next_conflicts = curr.num_of_conflicts + \
                                     int(curr.collision_v) * max(next_timestep - curr.timestep - 1, 0) + \
                                     int(next_v_collision) + int(next_e_collision)
                    nxt = SIPPNode(next_location, next_g_val, next_h_val, curr, next_timestep,
                                   next_high_generation, next_high_expansion, next_v_collision, next_conflicts)
                    if self.dominance_check(nxt):
                        self._push_node_to_open_and_focal(nxt)
                    else:
                        del nxt

            # 在当前位置等待
            interval, succ = reservation_table.find_safe_interval(interval, curr.location, curr.high_expansion)
            if curr.high_expansion == curr.high_generation and \
                    succ and interval[0] + curr.h_val <= reservation_table.constraint_table.length_max:
                next_timestep = interval[0]
                next_h_val = max(self.my_heuristic[curr.location], curr.f_val - next_timestep)  # 路径最大值
                next_collisions = curr.num_of_conflicts + \
                                  int(curr.collision_v) * max(next_timestep - curr.timestep - 1, 0) + \
                                  int(interval[2])
                nxt = SIPPNode(curr.location, next_timestep, next_h_val, curr, next_timestep,
                               interval[1], interval[1], interval[2], next_collisions)
                if curr.location == self.goal_location:
                    nxt.wait_at_goal = True
                if self.dominance_check(nxt):
                    self._push_node_to_open_and_focal(nxt)
                else:
                    del nxt

        self._release_nodes()
        # print(f'iterations for agent {agent}: {iterations}')
        return path, self.min_f_val

    def find_path(self, constraint_table: ConstraintTable) -> cm.Path:
        """return A path that minimizes collisions, breaking ties by cost"""
        self.reset()
        reservation_table = ReservationTable(constraint_table, self.goal_location)
        path: cm.Path = []
        interval = reservation_table.get_first_safe_interval(self.start_location)
        if interval[0] > 0:
            return path
        holding_time = constraint_table.get_holding_time(self.goal_location, constraint_table.length_min)
        last_target_collision_time = constraint_table.get_last_collision_timestep(self.goal_location)
        # generate start and add it to the OPEN & FOCAL list
        h = max(max(self.my_heuristic[self.start_location], holding_time), last_target_collision_time + 1)
        start = SIPPNode(self.start_location, 0, h, None,
                         0, interval[1], interval[1], interval[2], interval[2])
        self._push_node_to_focal(start)

        while self.focal_list:
            curr = self.focal_list.top()
            self.focal_list.pop()
            curr.in_openlist = False
            self.num_expanded += 1
            assert curr.location >= 0
            # check if the popped node is a goal
            if curr.is_goal:
                self._update_path(curr, path)
                break
            elif curr.location == self.goal_location and not curr.wait_at_goal and curr.timestep >= holding_time:
                future_collisions = constraint_table.get_future_num_of_collisions(curr.location, curr.timestep)
                if future_collisions == 0:
                    self._update_path(curr, path)
                    break
                goal = SIPPNode(*curr)
                goal.is_goal = True
                goal.h_val = 0
                goal.num_of_conflicts += future_collisions
                if self.dominance_check(goal):
                    self._push_node_to_focal(goal)
                else:
                    del goal

            for next_location in self.instance.get_neighbors(curr.location):
                for i in reservation_table.get_safe_intervals(curr.location, next_location, curr.timestep + 1,
                                                              curr.high_expansion + 1):
                    next_high_generation, next_timestep, next_high_expansion, next_v_collision, next_e_collision = i
                    if next_timestep + self.my_heuristic[next_location] > constraint_table.length_max:
                        break
                    next_collisions = curr.num_of_conflicts + int(curr.collision_v) * max(
                        next_timestep - curr.timestep - 1, 0) + int(next_v_collision) + int(next_e_collision)
                    next_h_val = max(self.my_heuristic[next_location],
                                     (holding_time if next_collisions > 0 else curr.f_val) - next_timestep)
                    nxt = SIPPNode(next_location, next_timestep, next_h_val, curr, next_timestep, next_high_generation,
                                   next_high_expansion, next_v_collision, next_collisions)
                    if self.dominance_check(nxt):
                        self._push_node_to_focal(nxt)
                    else:
                        del next

            interval, succ = reservation_table.find_safe_interval(interval, curr.location, curr.high_expansion)
            if curr.high_expansion == curr.high_generation and succ and \
                    interval[0] + curr.h_val <= reservation_table.constraint_table.length_max:
                next_timestep = interval[0]
                next_h_val = max(curr.h_val, (holding_time if interval[2] else curr.f_val) - next_timestep)
                next_collisions = curr.num_of_conflicts + int(curr.collision_v) * max(next_timestep - curr.timestep - 1,
                                                                                      0) + int(interval[2])
                nxt = SIPPNode(curr.location, next_timestep, next_h_val, curr, next_timestep, interval[1], interval[1],
                               interval[2], next_collisions)
                nxt.wait_at_goal = (curr.location == self.goal_location)
                if self.dominance_check(nxt):
                    self._push_node_to_focal(nxt)
                else:
                    del nxt

        self._release_nodes()
        return path

    def get_travel_time(self, start: int, end: int, constraint_table: ConstraintTable, upper_bound: int):
        self.reset()
        self.min_f_val = -1  # this disables focal list
        length = cm.MAX_TIMESTEP
        root = SIPPNode(
            start, 0, self.compute_heuristic(start, end), None,
            0, 1, 1, 0, 0
        )
        self._push_node_to_open_and_focal(root)
        static_timestep = constraint_table.get_max_timestep()  # 这个时间步之后所有内容都是静态的

        while not self.open_list.empty():
            curr = self.open_list.pop()

            if curr.location == end:
                length = curr.g_val
                break

            next_locations = list(self.instance.get_neighbors(curr.location))
            next_locations.append(curr.location)

            for next_location in next_locations:
                next_timestep = curr.timestep + 1
                next_g_val = curr.g_val + 1

                if static_timestep <= curr.timestep:
                    if curr.location == next_location:
                        continue
                    next_timestep -= 1

                if (not constraint_table.constrained(next_location, next_timestep) and
                        not constraint_table.constrained(curr.location, next_location, next_timestep)):
                    # 如果该网格未被阻塞
                    next_h_val = self.compute_heuristic(next_location, end)

                    if next_g_val + next_h_val >= upper_bound:  # 路径成本大于上限
                        continue

                    nxt = SIPPNode(next_location, next_g_val, next_h_val, None, next_timestep,
                                   next_timestep + 1, next_timestep + 1, False, 0)

                    if self.dominance_check(nxt):
                        self._push_node_to_open_and_focal(nxt)
                    else:
                        del nxt

        self._release_nodes()
        return length

    @staticmethod
    def _update_path(goal: SIPPNode, path: List[cm.PathEntry]):
        # num_collisions = goal.num_of_conflicts
        path += [cm.PathEntry() for _ in range(goal.timestep - len(path) + 1)]

        curr = goal
        while curr.parent is not None:  # 非根节点
            prev = curr.parent
            t = prev.timestep + 1
            while t < curr.timestep:
                path[t].location = prev.location  # 在前一个位置等待
                t += 1
            path[curr.timestep].location = curr.location  # 移动到当前节点位置
            curr = prev

        assert curr.timestep == 0
        path[0].location = curr.location

    def _push_node_to_open_and_focal(self, node: SIPPNode):
        self.num_generated += 1
        self.open_list.add(node, *node.get_sort_tuple_for_open())
        if node.f_val <= self.w * self.min_f_val:
            self.focal_list.add(node, *node.get_sort_tuple_for_focal())
        if self.all_nodes_table.get(node, None) is None:
            self.all_nodes_table[node] = []
        self.all_nodes_table[node].append(node)

    def _push_node_to_focal(self, node: SIPPNode):
        self.num_generated += 1
        if self.all_nodes_table.get(node, None) is None:
            self.all_nodes_table[node] = []
        self.all_nodes_table[node].append(node)
        node.in_openlist = True
        self.focal_list.add(node, *node.get_sort_tuple_for_focal())  # we only use focal list; no open list is used

    def _erase_node_from_list(self, node: SIPPNode):
        if self.open_list.empty():
            # we only have focal list
            self.focal_list.remove(node)
        elif self.focal_list.empty():
            # we only have open list
            self.open_list.remove(node)
        else:
            self.open_list.remove(node)
            if node.f_val <= self.w * self.min_f_val:
                self.focal_list.remove(node)

    def _update_focal_list(self):
        open_head: SIPPNode = self.open_list.top()
        if open_head.f_val > self.min_f_val:
            for n in self.open_list:
                if self.w * self.min_f_val < n.f_val <= self.w * open_head.f_val:
                    self.focal_list.add(n, *n.get_sort_tuple_for_focal())
            self.min_f_val = open_head.f_val

    def _release_nodes(self):
        self.open_list.clear()
        self.focal_list.clear()
        self.all_nodes_table.clear()
        self.useless_nodes.clear()

    def dominance_check(self, new_node: SIPPNode) -> bool:
        """return true if the new node is not dominated by any old node"""
        ptr = self.all_nodes_table.get(new_node, None)
        if ptr is None:
            return True

        for old_node in ptr:
            if old_node.timestep <= new_node.timestep and old_node.num_of_conflicts <= new_node.num_of_conflicts:
                # 新节点被旧节点支配
                return False
            elif old_node.timestep >= new_node.timestep and old_node.num_of_conflicts >= new_node.num_of_conflicts:
                # 旧节点被新节点支配
                if old_node.in_openlist:
                    self._erase_node_from_list(old_node)  # 从 OPEN 和/或 FOCAL 列表中删除它
                self.useless_nodes.append(old_node)
                self.all_nodes_table[new_node].remove(old_node)
                self.num_generated -= 1  # 因为稍后我们会在将新节点插入列表时增加 num_generated
                return True
            elif old_node.timestep < new_node.high_expansion and new_node.timestep < old_node.high_expansion:
                # 区间重叠 --> 我们需要拆分节点以使其不重叠
                if old_node.timestep <= new_node.timestep:
                    assert old_node.num_of_conflicts > new_node.num_of_conflicts
                    old_node.high_expansion = new_node.timestep
                else:  # 即 old_node.timestep > new_node.timestep
                    assert old_node.num_of_conflicts <= new_node.num_of_conflicts
                    new_node.high_expansion = old_node.timestep

        return True


if __name__ == "__main__":
    pass
