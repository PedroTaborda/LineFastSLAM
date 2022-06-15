from __future__ import annotations
from dataclasses import dataclass, field
import os
import shlex
from typing import TypedDict

import numpy as np

@dataclass
class UsimMap:
    landmarks: TypedDict('Landmark', {'id': int, 'position': np.ndarray})

    # List[tuple[x0, y0, x1, y1]] each line is represented by two points on the line
    lines: list[tuple[float, float, float, float]] = None

    def __post_init__(self):
        self.lines = self.compute_lines()

    def compute_lines(self):
        landmarks_lines = []
        for landmark_id in list(self.landmarks.keys()):
            if landmark_id+1 in self.landmarks:
                x0, y0, _ = self.landmarks[landmark_id]
                x1, y1, _ = self.landmarks[landmark_id+1]
                landmarks_lines.append((x0, y0, x1, y1))
        return landmarks_lines


def load_map(map_file: os.PathLike):
    landmarks = {}
    with open(map_file, 'r') as map_file:
        for idx, line in enumerate(map_file):
            instruction = shlex.split(line, comments=True)
            if not instruction:
                continue
            cmd, *args = instruction
            if cmd.strip() == 'landmark':
                if len(args) != 4:
                    raise ValueError(f"Wrong number of arguments for 'landmark' instruction at line {idx} of file '{map_file}'. \nExpected 3, got {len(args)}.\n--{line}")
                id = int(args[0])
                x = float(args[1])
                y = float(args[2])
                orientation = float(args[3])
                landmarks[id] = np.array([x, y, orientation])
    map_w_landmarks = UsimMap(landmarks=landmarks)
    return map_w_landmarks
    
if __name__ == '__main__':
    map_path = 'map1.map'
    env = load_map(map_path)
