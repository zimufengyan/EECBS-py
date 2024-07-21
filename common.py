# -*- coding:utf-8 -*-
# @FileName  :common.py
# @Time      :2024/7/19 下午6:43
# @Author    :ZMFY
# Description:

from typing import List, Dict, Set, Tuple, Union
from dataclasses import dataclass
import sys


MAX_TIMESTEP = sys.maxsize // 2
MAX_COST = sys.maxsize // 2
MAX_NODES = sys.maxsize // 2


@dataclass
class PathEntry:
    location: int = -1
    

class Path(list):
    def __add__(self, other):
        for entry in other:
            if not isinstance(entry, PathEntry):
                raise TypeError(f"element of 'other' must be PathEntry instances, not {type(entry)}")
            super().__add__(other)
        
    def append(self, other):
        if not isinstance(other, PathEntry):
            raise TypeError(f"element of 'other' must be PathEntry instances, not {type(other)}")
        super().append(other)


def is_same_path(path1: Path, path2: Path) -> bool:
    if len(path1) != len(path2):
        return False
    for p, q in zip(path1, path2):
        if p.location != q.location:
            return False

    return True


if __name__ == "__main__":
    pass
