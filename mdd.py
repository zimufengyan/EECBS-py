# -*- coding:utf-8 -*-
# @FileName  :mdd.py
# @Time      :2024/7/19 下午7:27
# @Author    :ZMFY
# Description:

import heapq as hpq
import time
from collections import deque
from typing import List, Dict, Union, Set
from multipledispatch import dispatch
from copy import deepcopy

import common as cm
from contraint_table import ConstraintTable
from nodes import MDDNode, SyncMDDNode, ConstraintsHasher, HLNode
from single_agent_solver import SingleAgentSolver


class MDD:
    def __init__(self):
        self.ct: ConstraintTable = None
        self.solver: SingleAgentSolver = None
        self.num_of_levels = None
        self.levels: List[List[MDDNode]] = []       # (level, location)

    @dispatch(ConstraintTable, SingleAgentSolver)
    def build_mdd(self, constraint_table: ConstraintTable, solver: SingleAgentSolver) -> bool:
        class Node:
            def __init__(self, location, timestep, h_val=-1):
                self.location = location
                self.timestep = timestep
                self.h_val = h_val
                self.parents: List[Node] = []
                self.mdd_node: Union[MDDNode, None] = None

            def __eq__(self, other):
                return self.location == other.location and self.timestep == other.timestep

            def __lt__(self, other):
                """used by OPEN (heap) to compare nodes (top of the heap has min f-val, and then highest g-val)"""
                return self.timestep + self.h_val < other.timestep + other.h_val

            def __hash__(self):
                return hash(self.location * (self.timestep << 1))

            def get_hash_key(self):
                return self.location * (self.timestep << 1)

        self.solver = solver
        holding_time = constraint_table.get_holding_time(solver.goal_location, constraint_table.length_min)
        root = Node(solver.start_location, solver.my_heuristic[solver.start_location])

        open_list = [root]
        all_nodes_table: Dict[int, Node] = {root.get_hash_key(): root}
        goal_node = None
        upper_bound = constraint_table.length_max

        while len(open_list) > 0:
            curr: Node = hpq.heappop(open_list)
            if (goal_node is None
                    and curr.location == solver.goal_location  # arrive at the goal location
                    and curr.timestep >= holding_time):  # the idx can hold the goal location afterward
                if len(curr.parents) != 1 or curr.parents[0].location != solver.goal_location:
                    # skip the case where curr only have parent node who locates at goal_location
                    goal_node = curr
                    upper_bound = curr.timestep
                    continue
            if curr.timestep + curr.h_val > upper_bound:
                continue
            next_locations = solver.get_next_locations(curr.location)
            for next_location in next_locations:
                # Try every possible move. We only add backward edges in this step.
                next_timestep = curr.timestep + 1
                if (constraint_table.constrained(next_location, next_timestep)
                        or constraint_table.constrained(curr.location, next_location, next_timestep)):
                    continue
                next_h_val = solver.my_heuristic[next_location]
                if next_timestep + next_h_val > upper_bound:
                    continue
                nxt = Node(next_location, next_timestep, next_h_val)
                nxt_key = nxt.get_hash_key()
                existing_node = all_nodes_table.get(nxt_key, None)
                if existing_node is None:
                    nxt.parents.append(curr)
                    hpq.heappush(open_list, nxt)
                    all_nodes_table[nxt_key] = nxt
                else:
                    existing_node.parents.append(curr)  # then add corresponding parent link and child link
                    if (goal_node is None
                            and existing_node.location == solver.goal_location  # arrive at the goal location
                            and existing_node.timestep >= holding_time  # the idx can hold the goal location afterward
                            and curr.location == solver.goal_location):
                        # # skip the case where curr only have parent node who locates at goal_location
                        goal_node = existing_node
                        upper_bound = existing_node.timestep

        # Backward
        assert goal_node is not None
        self.levels: List[List[MDDNode]] = [[] for _ in range(goal_node.timestep + 1)]
        q_lst: List[Node] = []
        goal_node.mdd_node = MDDNode(goal_node.location, goal_node.timestep)
        q_lst.append(goal_node)
        self.levels[-1].append(goal_node.mdd_node)

        while len(q_lst) > 0:
            curr = q_lst.pop()
            for i, parent in enumerate(curr.parents):
                if curr == goal_node and parent.location == goal_node.location:
                    continue  # the parent of the goal node should not be at the goal location
                if parent.mdd_node is None:  # a new node
                    curr.parents[i].mdd_node = MDDNode(parent.location, parent.timestep)
                    self.levels[parent.timestep].append(curr.parents[i].mdd_node)
                curr.parents[i].mdd_node.children.append(curr.mdd_node)  # add forward edge
                curr.mdd_node.parents.append(curr.parents[i].mdd_node)  # add backward edge

        assert len(self.levels[0]) != 0
        all_nodes_table.clear()
        assert self.levels[-1][0].location == solver.goal_location
        return True

    @dispatch(ConstraintTable, int, SingleAgentSolver)
    def build_mdd(self, constraint_table: ConstraintTable, num_of_levels: int, solver: SingleAgentSolver) -> bool:
        self.solver = solver
        root = MDDNode(self.solver.start_location, parent=None)
        open_list = deque([root])
        closed_list: List[MDDNode] = [root]
        self.levels = [[] for _ in range(num_of_levels)]

        while len(open_list) > 0:
            curr = open_list.popleft()
            # Here we suppose all edge cost equals 1
            if curr.level == num_of_levels - 1:
                self.levels[-1].append(curr)
                assert len(open_list) == 0
                break
            # We want (g + 1)+h <= f = numOfLevels - 1, so h <= numOfLevels - g - 2.
            # -1 because it's the bound of the children.
            heuristic_bound = num_of_levels - curr.level - 2
            next_locations = solver.get_next_locations(curr.location)
            for next_location in next_locations:
                # Try every possible move. We only add backward edges in this step.
                if (self.solver.my_heuristic[next_location] <= heuristic_bound
                        and not constraint_table.constrained(next_location, curr.level + 1)
                        and not constraint_table.constrained(curr.location, next_location, curr.level + 1)):
                    find = False
                    for child in closed_list[::-1]:
                        if child.level != curr.level + 1:
                            break
                        if curr.location == child.location:  # If the child node exists
                            child.parents.append(curr)  # then add corresponding parent link and child link
                            find = True
                            break
                    if not find:  # Else generate a new mdd node
                        child_node = MDDNode(next_location, parent=curr)
                        child_node.cost = num_of_levels - 1
                        open_list.append(child_node)
                        closed_list.append(child_node)

        assert len(self.levels[-1]) == 1

        # Backward
        goal_node: MDDNode = self.levels[-1][-1]
        for i, parent in enumerate(goal_node.parents):
            if parent.location == goal_node.location:
                # the parent of the goal node should not be at the goal location
                continue
            self.levels[num_of_levels - 2].append(parent)
            goal_node.parents[i].children.append(goal_node)  # add forward edge
        for t in range(num_of_levels - 2, -1, -1):
            for i, node in enumerate(self.levels[t]):
                for j, parent in enumerate(node.parents):
                    if len(parent.children) == 0:  # a new node
                        self.levels[t - 1].append(parent)
                    self.levels[t][i].parents[j].children.append(node)  # add forward edge

        for it in closed_list:
            if len(it.children) == 0 and it.level < num_of_levels - 1:
                del it
        closed_list.clear()
        assert self.levels[-1][0].location == solver.goal_location
        return True

    def find(self, location: int, level: int) -> Union[MDDNode, None]:
        if level < len(self.levels):
            for i in range(len(self.levels[level])):
                if self.levels[level][i].location == location:
                    return self.levels[level][i]
        return None

    def clear(self):
        if len(self.levels) == 0:
            return

        # does it matter ??
        for i in range(len(self.levels)):
            for j in range(len(self.levels[i])):
                del self.levels[i][j]
        self.levels.clear()

    def delete_node(self, node: MDDNode):
        self.levels[node.level].remove(node)
        for i, child in enumerate(node.children):
            node.children[i].parents.remove(node)
            if len(node.children[i].parents) == 0:
                self.delete_node(node.children[i])
        for i in range(len(node.parents)):
            node.parents[i].children.remove(node)
            if len(node.parents[i].children) == 0:
                self.delete_node(node.parents[i])

    def increase_by(self, ct: ConstraintTable, d_level: int, solver: SingleAgentSolver):
        old_height = len(self.levels)
        num_of_levels = old_height + d_level
        self.levels = [[] for _ in range(num_of_levels)]
        for l in range(num_of_levels - 1):
            heuristic_bound = num_of_levels - l - 2 + 0.001
            node_map = collect_mdd_level(self, l + 1)
            for it in self.levels[l]:
                next_locations = solver.get_next_locations(it.location)
                for new_loc in next_locations:
                    if (self.solver.my_heuristic[new_loc] <= heuristic_bound
                            and not ct.constrained(new_loc, it.level + 1)
                            and not ct.constrained(it.location, new_loc, it.level + 1)):  # valid move
                        if node_map.get(new_loc, None) is None:
                            new_node = MDDNode(new_loc, parent=it)
                            self.levels[l + 1].append(new_node)
                            node_map[new_loc] = new_node
                        else:
                            node_map[new_loc].parents.append(it)

        # Backward
        for l in range(old_height, num_of_levels):
            goal_node = None
            for it in self.levels[l]:
                if it.location == solver.goal_location:
                    goal_node = it
                    break
            bfs_q = deque([goal_node])
            closed_list: Set[MDDNode] = set()

            while len(bfs_q) > 0:
                ptr: MDDNode = bfs_q.popleft()
                ptr.cost = 1
                for i, parent in enumerate(ptr.parents):
                    ptr.parents[i].children.append(ptr)     # add forward edge
                    if parent not in closed_list and parent.cost == 0:
                        bfs_q.append(parent)
                        closed_list.add(parent)

        # Delete useless nodes (nodes who don't have any children)
        for l in range(num_of_levels - 1):
            for it in self.levels[l]:
                if len(it.children) == 0:
                    self.levels[l].remove(it)

    def goal_at(self, level: int) -> Union[MDDNode, None]:
        if level >= len(self.levels):
            return None

        for ptr in self.levels[level]:
            if ptr.location == self.solver.goal_location and ptr.cost == level:
                return ptr

        return None

    def print_nodes(self):
        for level in self.levels:
            out = f"{level[0].level}\t"
            for loc in level:
                out += f"{loc.location}, "
            print(out)


class SyncMDD:
    def __init__(self):
        self.levels: List[List[SyncMDDNode]] = []

    def fing(self, location: int, level: int) -> Union[SyncMDDNode, None]:
        pass

    def delete_node(self, node: SyncMDDNode, level: int):
        pass

    def clear(self):
        pass


class MDDTable:
    def __init__(self, initial_constraints: List[ConstraintTable], search_engines: List[SingleAgentSolver]):
        self.initial_constraints = initial_constraints
        self.search_engines = search_engines
        self.accumulated_runtime: float = 0     # runtime of building MDDs
        self.num_released_mdds = 0      # number of released MDDs ( to save memory)
        self.max_num_of_mdds = 10000        # per idx
        self.lookup_table: List[Dict[ConstraintsHasher, MDD]] = []

    def init(self, number_of_agenets: int):
        self.lookup_table = [dict() for _ in range(number_of_agenets)]

    def clear(self):
        for mdds in self.lookup_table:
            for k, v in mdds.items():
                del v
        self.lookup_table.clear()

    def find_mdd(self, node: HLNode, agent: int) -> Union[MDD, None]:
        c = ConstraintsHasher(agent, node)
        got = self.lookup_table[c.a].get(c, None)
        if got is not None:
            return got
        else:
            return None

    def get_mdd(self, node: HLNode, idx: int, mdd_levels=-1):
        c = ConstraintsHasher(idx, node)
        got = self.lookup_table[c.a].get(c, None)
        if got is not None:
            assert mdd_levels < 0 or len(got.levels) == mdd_levels
            return got
        self.release_mdd_memory(idx)
        st = time.perf_counter()
        mdd = MDD()
        ct = deepcopy(self.initial_constraints[idx])
        if mdd_levels >= 0:
            mdd.build_mdd(ct, mdd_levels, self.search_engines[idx])
        else:
            mdd.build_mdd(ct, self.search_engines[idx])
        if not len(self.lookup_table) == 0:
            self.lookup_table[c.a][c] = mdd
        self.accumulated_runtime += time.perf_counter() - st
        return mdd

    def release_mdd_memory(self, idx: int):
        if idx < 0 or len(self.lookup_table) == 0 or len(self.lookup_table[idx]) < self.max_num_of_mdds:
            return

        min_length = cm.MAX_TIMESTEP
        for mdd in self.lookup_table[idx].keys():
            min_length = min(min_length, len(self.lookup_table[idx][mdd].levels))
        for mdd in self.lookup_table[idx].keys():
            if len(self.lookup_table[idx][mdd].levels) == min_length:
                del self.lookup_table[idx][mdd]
                self.lookup_table[idx].pop(mdd)
                self.num_released_mdds += 1


def collect_mdd_level(mdd, i: int):
    loc2mdd = dict()
    for it in mdd.levels[i]:
        loc = it.location
        loc2mdd[loc] = it
    return loc2mdd


if __name__ == "__main__":
    pass
