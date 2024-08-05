# -*- coding:utf-8 -*-
# @FileName  :common.py
# @Time      :2024/7/19 下午6:43
# @Author    :ZMFY
# Description:

from typing import List, Dict, Set, Tuple, Union
from dataclasses import dataclass
import sys
import heapq as hpq

MAX_TIMESTEP = sys.maxsize // 2
MAX_COST = sys.maxsize // 2
MAX_NODES = sys.maxsize // 2


@dataclass
class PathEntry:
    location: int = -1

    def __repr__(self):
        return str(self.location)


Path = List[PathEntry]


class PrioritySet(object):
    """
    priority queue, min-heap
    """

    def __init__(self):
        """
        no duplication allowed
        """
        self.heap_ = []
        self.set_ = set()
        self.dic_ = dict()  # Use the id() as keys to ensure uniqueness because we had overload the __hash__ and __eq__

    def add(self, d, *args):
        """
        will check for duplication and avoid.
        """
        assert len(self.heap_) >= len(self.dic_)
        key = id(d)
        if self.dic_.get(key, None) is None:
            item = (*args, d) if len(args) != 0 else (1, d)
            hpq.heappush(self.heap_, item)
            self.dic_[key] = d
            # print(f"{id(self)}: add", len(self.heap_), len(self.set_))

    def pop(self):
        """
        impl detail: return the first(min) item that is in self.set_
        """
        assert len(self.heap_) >= len(self.dic_) > 0
        # if len(self.set_) == 0:
        #     raise StopIteration
        popped = hpq.heappop(self.heap_)
        d = popped[-1]
        # while d not in self.set_:
        #     popped = hpq.heappop(self.heap_)
        #     d = popped[-1]
        key = id(d)
        while self.dic_.get(key, None) is None:
            popped = hpq.heappop(self.heap_)
            d = popped[-1]
            key = id(d)
            # print(f"{id(self)}: pop", len(self.heap_), len(self.set_))

        # self.set_.remove(d)
        self.dic_.pop(key)
        # print(f"{id(self)}: pop", len(self.heap_), len(self.set_))
        return d

    def __len__(self):
        return len(self.dic_)

    def __iter__(self):
        return iter(self.dic_.values())

    def print(self):
        print(self.heap_)
        print(self.dic_.values())
        return

    def has(self, d):
        return self.dic_.get(id(d), None) is not None

    def remove(self, d):
        """
        implementation: only remove from self.set_, not remove from self.heap_ list.
        """
        assert len(self.heap_) >= len(self.dic_)
        # if d not in self.set_:
        #     return False
        key = id(d)
        if self.dic_.get(key, None) is None:
            return False
        # self.set_.remove(d)
        self.dic_.pop(key)
        i = 0
        while i < len(self.heap_):
            p = self.heap_[i][-1]
            # if hash(d) == hash(p) and d == p:
            #     del self.heap_[i]
            if id(p) == key:
                del self.heap_[i]
            else:
                i += 1
        # print(f"{id(self)}: remove", len(self.heap_), len(self.set_))
        return True

    def update(self):
        assert len(self.heap_) >= len(self.dic_) > 0
        hpq.heapify(self.heap_)

    def clear(self):
        # self.set_.clear()
        self.heap_.clear()
        self.dic_.clear()

    def top(self):
        # print('\n')
        while True:
            assert len(self.heap_) >= len(self.dic_) > 0
            d = hpq.nsmallest(1, self.heap_)[0][-1]
            if self.has(d):
                return d
            hpq.heappop(self.heap_)
            # print(f'{id(self)}: pop, now |heap_|={len(self.heap_)}, |set_|={len(self.set_)}')

    def empty(self) -> bool:
        return len(self.dic_) == 0


def is_same_path(path1: Path, path2: Path) -> bool:
    if len(path1) != len(path2):
        return False
    return all(p.location == q.location for p, q in zip(path1, path2))


if __name__ == "__main__":
    pass
