# -*- coding:utf-8 -*-
# @FileName  :instance.py
# @Time      :2024/7/21 下午1:36
# @Author    :ZMFY
# Description:

import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Union
from multipledispatch import dispatch
import random

import common as cm


class Instance(object):
    RANDOM_WALK_STEPS = 100000

    def __init__(self, map_fname, agent_fname, num_of_agents=0,
                 num_of_rows=0, num_of_cols=0, num_of_obstacles=0, warehouse_width=0):
        self.map_fname = map_fname
        self.agent_fname = agent_fname
        self.num_of_agents = num_of_agents
        self.num_of_rows = num_of_rows
        self.num_of_cols = num_of_cols
        self.num_of_obstacles = num_of_obstacles
        self.warehouse_width = warehouse_width

        succ = self.load_map()
        if not succ:
            if (self.num_of_rows > 0 and self.num_of_cols > 0
                    and 0 < self.num_of_obstacles < self.num_of_rows * self.num_of_cols):
                self._generate_connected_random_grid(self.num_of_rows, self.num_of_cols, self.num_of_obstacles)
                self.save_map()
            else:
                raise RuntimeError(f"Map file {self.map_fname} not found.")

        succ = self.load_agents()
        if not succ:
            if self.num_of_agents > 0:
                self._generate_random_agents(self.warehouse_width)
                self.save_agents()
            else:
                raise RuntimeError(f"Agent file {self.agent_fname} not found.")

        self.map_size: int = self.num_of_rows * self.num_of_cols
        self.my_map = np.zeros(self.map_size, dtype=bool)
        self.start_locations: np.ndarray = np.zeros(self.num_of_agents, dtype=int)
        self.goal_locations: np.ndarray = np.zeros(self.num_of_agents, dtype=int)

    @property
    def cols(self):
        return self.num_of_cols

    @property
    def rows(self):
        return self.num_of_rows

    def is_obstacle(self, loc: int) -> bool:
        return bool(self.my_map[loc])

    def valid_move(self, curr: int, nxt: int):
        if nxt < 0 or nxt > self.map_size:
            return False
        if self.my_map[nxt]:
            return False
        return self.get_manhattan_distance(curr, nxt) < 2

    def get_neighbors(self, curr: int) -> List[int]:
        neighbors: List[int] = []
        candidates = [curr + 1, curr - 1, curr + self.num_of_cols, curr - self.num_of_cols]
        for nxt in candidates:
            if self.valid_move(curr, nxt):
                neighbors.append(nxt)
        return neighbors

    def linearize_coordinate(self, row, col) -> int:
        return self.num_of_cols * row + col

    def get_row_coordinate(self, loc: int) -> int:
        return loc // self.num_of_cols

    def get_col_coordinate(self, loc: int) -> int:
        return loc % self.num_of_cols

    def get_coordinate(self, loc: int) -> Tuple[int, int]:
        return loc // self.num_of_cols, loc % self.num_of_cols

    @dispatch(int, int)
    def get_manhattan_distance(self, loc1: int, loc2: int) -> int:
        loc1_x = self.get_row_coordinate(loc1)
        loc2_x = self.get_row_coordinate(loc2)
        loc1_y = self.get_col_coordinate(loc1)
        loc2_y = self.get_col_coordinate(loc2)
        return abs(loc1_x - loc2_x) + abs(loc1_y - loc2_y)

    @dispatch(tuple, tuple)
    def get_manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def get_degree(self, loc: int) -> int:
        assert 0 <= loc < self.map_size and not self.my_map[loc]
        degree = 0
        if loc - self.num_of_cols >= 0 and not self.my_map[loc - self.num_of_cols]:
            degree += 1
        if loc + self.num_of_cols < self.map_size and not self.my_map[loc + self.num_of_cols]:
            degree += 1
        if loc % self.num_of_cols > 0 and not self.my_map[loc - 1]:
            degree += 1
        if loc % self.num_of_cols < self.num_of_cols - 1 and not self.my_map[loc + 1]:
            degree += 1
        return degree

    def load_map(self) -> bool:
        try:
            with open(self.map_fname, 'r') as myfile:
                line = myfile.readline().strip()

                if line[0] == 't':  # Nathan's benchmark
                    myfile.readline()  # Skip line
                    line = myfile.readline().strip()
                    num_of_rows = int(line.split()[1])  # read number of rows

                    line = myfile.readline().strip()
                    num_of_cols = int(line.split()[1])  # read number of cols

                    myfile.readline()  # skip "map"
                else:  # my benchmark
                    num_of_rows, num_of_cols = map(int, line.split(','))

                self.num_of_rows = num_of_rows
                self.num_of_cols = num_of_cols
                self.map_size = self.num_of_cols * self.num_of_rows
                self.my_map = np.zeros(self.map_size, dtype=bool)

                # read map (and start/goal locations)
                for i in range(self.num_of_rows):
                    line = myfile.readline().strip()
                    for j in range(self.num_of_cols):
                        self.my_map[self.linearize_coordinate(i, j)] = (line[j] != '.')

            return True

        except IOError:
            return False

    def print_map(self):
        for i in range(self.num_of_rows):
            line = ""
            for j in range(self.num_of_cols):
                if self.my_map[self.linearize_coordinate(i, j)]:
                    line += '@'
                else:
                    line += '.'
            print(line)

    def save_map(self):
        with open(self.map_fname, 'w') as myfile:
            myfile.write(f"{self.num_of_rows},{self.num_of_cols}")
            for i in range(self.num_of_rows):
                for j in range(self.num_of_cols):
                    line = ""
                    if self.my_map[self.linearize_coordinate(i, j)]:
                        line += '@'
                    else:
                        line += '.'
                    myfile.write(line)

    def load_agents(self) -> bool:
        try:
            with open(self.agent_fname, 'r') as myfile:
                line = myfile.readline().strip()

                if line[0] == 'v':  # Nathan's benchmark
                    if self.num_of_agents == 0:
                        raise ValueError("The number of agents should be larger than 0")

                    self.start_locations = np.zeros(self.num_of_agents, dtype=int)
                    self.goal_locations = np.zeros(self.num_of_agents, dtype=int)

                    for i in range(self.num_of_agents):
                        line = myfile.readline().strip().split('\t')
                        start_col = int(line[4])
                        start_row = int(line[5])
                        self.start_locations[i] = self.linearize_coordinate(start_row, start_col)
                        goal_col = int(line[6])
                        goal_row = int(line[7])
                        self.goal_locations[i] = self.linearize_coordinate(goal_row, goal_col)
                else:  # My benchmark
                    line = line.split(',')
                    self.num_of_agents = int(line[0])
                    self.start_locations = np.zeros(self.num_of_agents, dtype=int)
                    self.goal_locations = np.zeros(self.num_of_agents, dtype=int)

                    for i in range(self.num_of_agents):
                        line = myfile.readline().strip().split(',')
                        start_row = int(line[0])
                        start_col = int(line[1])
                        self.start_locations[i] = self.linearize_coordinate(start_row, start_col)
                        goal_row = int(line[2])
                        goal_col = int(line[3])
                        self.goal_locations[i] = self.linearize_coordinate(goal_row, goal_col)

            return True
        except IOError:
            return False

    def print_agents(self):
        for i in range(self.num_of_agents):
            print(f"Agent {i} : "
                  f"S=({self.get_row_coordinate(int(self.start_locations[i]))},"
                  f"{self.get_col_coordinate(int(self.start_locations[i]))}); "
                  f"G=({self.get_row_coordinate(int(self.goal_locations[i]))},"
                  f"{self.get_col_coordinate(int(self.goal_locations[i]))})")

    def save_agents(self):
        with open(self.agent_fname, 'w') as myfile:
            myfile.write(f'{self.num_of_agents}')
            for i in range(self.num_of_agents):
                myfile.write(f"Agent {i} : "
                             f"S=({self.get_row_coordinate(int(self.start_locations[i]))},"
                             f"{self.get_col_coordinate(int(self.start_locations[i]))}); "
                             f"G=({self.get_row_coordinate(int(self.goal_locations[i]))},"
                             f"{self.get_col_coordinate(int(self.goal_locations[i]))})")

    def _generate_connected_random_grid(self, rows, cols, obstacles):
        """initialize new [rows x cols] map with random obstacles"""
        print(f"Generate a {rows} x {cols} grid with {obstacles} obstacles")
        self.num_of_rows = rows + 2
        self.num_of_cols = cols + 2
        self.map_size = self.num_of_rows * self.num_of_cols
        self.my_map = np.zeros(self.map_size, dtype=bool)

        for j in range(self.num_of_cols):
            self.my_map[self.linearize_coordinate(0, j)] = True
            self.my_map[self.linearize_coordinate(self.num_of_rows - 1, j)] = True
        for i in range(self.num_of_rows):
            self.my_map[self.linearize_coordinate(i, 0)] = True
            self.my_map[self.linearize_coordinate(i, self.num_of_cols - 1)] = True

        for i in range(obstacles):
            loc = np.random.randint(0, self.map_size)
            if self._add_obstacle(loc):
                self.print_map()

    def _generate_random_agents(self, warehouse_width):
        print(f"Generate {self.num_of_agents} random start and goal locations")
        starts = np.zeros(self.map_size, dtype=bool)
        goals = np.zeros(self.map_size, dtype=bool)
        self.start_locations = np.zeros(self.num_of_agents, dtype=int)
        self.goal_locations = np.zeros(self.num_of_agents, dtype=int)

        if warehouse_width == 0:
            # Generate agents randomly
            k = 0
            while k < self.num_of_agents:
                x = np.random.randint(self.num_of_rows)
                y = np.random.randint(self.num_of_cols)
                start = self.linearize_coordinate(x, y)
                if self.my_map[start] or starts[start]:
                    continue

                # Update start
                self.start_locations[k] = start
                starts[start] = True

                # Find goal
                flag = False
                goal = np.random.randint(self.map_size)
                while self.my_map[goal] or goals[goal]:
                    goal = np.random.randint(self.map_size)

                # Update goal
                self.goal_locations[k] = goal
                goals[goal] = True

                k += 1
        else:
            # Generate agents for warehouse scenario
            k = 0
            while k < self.num_of_agents:
                x = np.random.randint(self.num_of_rows)
                y = np.random.randint(warehouse_width)
                if k % 2 == 0:
                    y = self.num_of_cols - y - 1
                start = self.linearize_coordinate(x, y)
                if starts[start]:
                    continue
                # Update start
                self.start_locations[k] = start
                starts[start] = True

                k += 1

            # Choose random goal locations
            k = 0
            while k < self.num_of_agents:
                x = np.random.randint(self.num_of_rows)
                y = np.random.randint(warehouse_width)
                if k % 2 == 1:
                    y = self.num_of_cols - y - 1
                goal = self.linearize_coordinate(x, y)
                if goals[goal]:
                    continue
                # Update goal
                self.goal_locations[k] = goal
                goals[goal] = True
                k += 1

    def _add_obstacle(self, obstacle: int) -> bool:
        """add this obstacle only if the map is still connected"""
        if self.my_map[obstacle]:
            return False

        self.my_map[obstacle] = True
        obstacle_x = self.get_row_coordinate(obstacle)
        obstacle_y = self.get_col_coordinate(obstacle)
        x = np.array([obstacle_x, obstacle_x + 1, obstacle_x, obstacle_x - 1])
        y = np.array([obstacle_y - 1, obstacle_y, obstacle_y + 1, obstacle_y])

        start = 0
        goal = 1
        while start < 3 and goal < 4:
            if (x[start] < 0 or x[start] >= self.num_of_rows or
                    y[start] < 0 or y[start] >= self.num_of_cols or
                    self.my_map[self.linearize_coordinate(x[start], y[start])]):
                start += 1
            elif goal <= start:
                goal = start + 1
            elif (x[goal] < 0 or x[goal] >= self.num_of_rows or
                  y[goal] < 0 or y[goal] >= self.num_of_cols or
                  self.my_map[self.linearize_coordinate(x[goal], y[goal])]):
                goal += 1
            elif self._is_connected(self.linearize_coordinate(x[start], y[start]),
                                    self.linearize_coordinate(x[goal],
                                                              y[goal])):  # cannot find a path from start to goal
                start = goal
                goal += 1
            else:
                self.my_map[obstacle] = False
                return False

        return True

    def _is_connected(self, start: int, goal: int) -> bool:
        """run BFS to find a path between start and goal, return true if a path exists."""
        open_lst = deque([start])
        closed_lst = np.zeros(self.map_size, dtype=int)
        closed_lst[start] = 1
        while len(open_lst) > 0:
            curr = open_lst.popleft()
            if curr == goal:
                return True
            for nxt in self.get_neighbors(curr):
                if closed_lst[nxt]:
                    continue
                open_lst.append(nxt)
                closed_lst[nxt] = 1

        return False

    def _random_walk(self, curr: int, steps: int) -> int:
        for walk in range(steps):
            lst = self.get_neighbors(curr)
            random.shuffle(lst)
            for nxt in lst:
                if self.valid_move(curr, nxt):
                    curr = nxt
                    break

        return curr


if __name__ == "__main__":
    pass
