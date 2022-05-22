from dataclasses import dataclass

from math import cos, sin
import numpy as np

@dataclass
class RobotSettings:
    ...

@dataclass 
class RobotData:
    theta: np.ndarray
    x: np.ndarray
    y: np.ndarray
    
class Robot:
    def __init__(self, settings: RobotSettings, initial_conditions: list = [0, 0, 0]) -> None:
        self.settings: RobotSettings = settings
        self.data = {}

        self.data['theta'] = [initial_conditions[0]]
        self.data['x'] = [initial_conditions[1]]
        self.data['y'] = [initial_conditions[2]]

    def simulation_step(self, angular_velocity: float, 
                        linear_velocity: float, sampling_time: float) -> None:

        self.data['theta'] += [np.rad2deg(np.deg2rad(self.data['theta'][-1]) + angular_velocity * sampling_time)]
        self.data['x'] += [self.data['x'][-1] + np.cos(np.deg2rad(self.data['theta'][-1])) * linear_velocity * sampling_time]

        self.data['y'] += [self.data['y'][-1] + np.sin(np.deg2rad(self.data['theta'][-1])) * linear_velocity * sampling_time]