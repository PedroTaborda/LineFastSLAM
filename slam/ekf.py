from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class EKFSettings:
    """
    Settings for the EKF.
    """
    ...

print(f"[WARNING] EKF code does not behave as EKF.")
class EKF:
    def __init__(self, settings: EKFSettings = EKFSettings()) -> None:
        self.position: np.ndarray = np.array([0, 0])

    def predict(self, *args):
        ...

    def update(self, measurement):
        self.position = measurement

    def get_position(self) -> np.ndarray:
        return self.position
