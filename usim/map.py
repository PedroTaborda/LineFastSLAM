from dataclasses import dataclass, field
import os
import shlex
from typing import TypedDict

import numpy as np

@dataclass
class Map:
    landmarks: TypedDict('Landmark', {'id': int, 'position': np.ndarray})

    lines: list = field(init=False)
    def compute_lines(self):
        landmarks_lines = []
        for landmark_id in list(self.landmarks.keys()):
            if landmark_id+1 in self.landmarks:
                x0, y0 = self.landmarks[landmark_id]
                x1, y1 = self.landmarks[landmark_id+1]
                landmarks_lines.append((x0, y0, x1, y1))
        return landmarks_lines
                

def load_map(map_file: os.PathLike):
    landmarks = {}
    with open(map_file, 'r') as map_file:
        for idx, line in enumerate(map_file):
            print(line)
            instruction = shlex.split(line, comments=True)
            print(instruction)
            if not instruction:
                continue
            cmd, *args = instruction
            if cmd.strip() == 'landmark':
                if len(args) != 3:
                    raise ValueError(f"Wrong number of arguments for 'landmark' instruction at line {idx} of file '{map_file}'. \nExpected 3, got {len(args)}.\n--{line}")
                id = int(args[0])
                x = float(args[1])
                y = float(args[2])
                landmarks[id] = np.array([x, y])
    map_w_landmarks = Map(landmarks=landmarks)
    map_w_landmarks.lines = map_w_landmarks.compute_lines()
    return map_w_landmarks
    
if __name__ == '__main__':
    map_path = 'map1.map'
    env = load_map(map_path)
