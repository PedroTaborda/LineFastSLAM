import numpy as np

from ekf.ekf import EKF

print(f"[WARNING] Map code does not behave as Map.")
class Map:
    def __init__(self) -> None:
        self.landmarks: dict[int, EKF] = {}
    def update(self, pose: np.ndarray, observation: tuple[int, np.ndarray]):
        landmark_id, landmark_position = observation
        if landmark_id not in self.landmarks:
            self.landmarks[landmark_id] = EKF(landmark_position)
            return 1.0
        else:
            prev_loc = self.landmarks[landmark_id].get_mu()
            self.landmarks[landmark_id].update(landmark_position)
            return 1.0 / (np.linalg.norm(prev_loc - landmark_position) + 1)

