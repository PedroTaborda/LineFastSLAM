from dataclasses import dataclass
import math
import copy

import scipy.stats
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import Ellipse

from ekf.ekf import EKF, EKFSettings


r = 0.2  # How much to "trust" measurements

print(f"[WARNING] Unrealistic landmarks.")
@dataclass
class LandmarkSettings(EKFSettings):
    """Settings for the EKF representing a landmark.
    
    A landmark is represented by its position in the xy plane.
    By default, there is a linear measurement model, but this can be
    changed by setting the `h` and `Dh_` functions at measurement time.
    """
    mu0: np.ndarray = np.array([0, 0])
    cov0: np.ndarray = np.diag([0.1, 0.1]) # also wrong but enough for now
    g: callable = lambda x, u, m: x
    Dgx: callable = lambda x, u, m: np.eye(2)
    Dgm: callable = lambda x, u, m: np.zeros((2, 2))
    h: callable = lambda x, n: x + r * np.eye(2) @ n
    Dhx: callable = lambda x, n: np.eye(2)
    Dhn: callable = lambda x, n: r * np.eye(2)

class Landmark(EKF):
    def __init__(self, settings: LandmarkSettings, mu0: np.ndarray):
        settings.mu0 = mu0
        super().__init__(settings)
        self.drawn = False
        self.confidence_interval = 0.99 # draw ellipse for this confidence interval
        self.latest_z = None

    def predict(self):
        super().predict(u=0)

    def update(self, z):
        super().update(z)
        self.latest_z = z
        
    def _draw(self, ax, color_ellipse='C00', color_p='C01', color_z='C02'):
        """Draw the landmark on the given matplotlib axis.

        This drawing includes an ellipse which is the level curve of the
        probability distribution of the landmark for p=confidence_interval.
        It also includes a marker for the mean of this distribution and another
        for the latest observation.
        """
        if self.latest_z is None:
            return
        p = self.get_mu()
        z = self.latest_z
        if not self.drawn:
            self.drawn = True    
            self.std_ellipse: Ellipse = Ellipse((0, 0), 1, 1, facecolor='none', edgecolor=color_ellipse)
            ax.add_patch(self.std_ellipse)
            self.p_handle: PathCollection = ax.scatter(p[0], p[1], marker='x', c=color_p)
            self.z_handle: PathCollection = ax.scatter(z[0], z[1], marker='1', c=color_z)

        # number of std's to include in confidence ellipse
        n_stds = -scipy.stats.norm.ppf((1-self.confidence_interval)/2)

        # Plot ellipse
        self.std_ellipse.set_center(self.get_mu())
        [w, v] = np.linalg.eig(self.get_cov())
        self.std_ellipse.set_width(np.sqrt(w[0])*n_stds*2)
        self.std_ellipse.set_height(np.sqrt(w[1])*n_stds*2)
        angle_deg = math.atan2(v[1, 0], v[0, 0]) * 180/np.pi
        self.std_ellipse.set_angle(angle_deg)

        # Plot estimated position
        self.p_handle.set(offsets=self.get_mu())

        # Plot latest observation
        self.z_handle.set(offsets=z)

    def __del__(self):
        if self.drawn:
            self.std_ellipse.remove()
            self.p_handle.remove()
            self.z_handle.remove()


print(f"[WARNING] Map code does not behave as Map.")
class Map:
    def __init__(self) -> None:
        self.landmarks: dict[int, Landmark] = {}
    def update(self, pose: np.ndarray, observation: tuple[int, np.ndarray]):
        landmark_id, landmark_position = observation
        if landmark_id not in self.landmarks:
            self.landmarks[landmark_id] = Landmark(
                LandmarkSettings(),
                mu0=landmark_position
            )
            return 1.0
        else:
            self.landmarks[landmark_id].predict()
            prev_loc = self.landmarks[landmark_id].get_mu()
            self.landmarks[landmark_id].update(landmark_position)
            return 1.0 / (np.linalg.norm(prev_loc - landmark_position) + 1)

    def _draw(self, ax, **plot_kwargs):
        for landmark in self.landmarks.values():
            landmark._draw(ax, **plot_kwargs)

    def copy(self):
        return copy.copy(self)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a map
    map = Map()

    # Create a figure
    fig, ax = plt.subplots(1, 1)

    # Draw the map
    map._draw(ax)

    poses = np.array([
        [0, 0.1, 0],
        [0, 0.2, 0],
        [0, 0.3, 0],
        [0, 0.4, 0],
        [0, 0.5, 0],
        [0, 0.6, 0],
        [0, 0.7, 0],
        [0, 0.8, 0],
        [0, 0.9, 0],
        [0, 1, 90],
        [0.1, 1, 90],
        [0.2, 1, 90],
        [0.3, 1, 90],
        [0.4, 1, 90],
        [0.5, 1, 90],
        [0.6, 1, 90],
        [0.7, 1, 90],
        [0.8, 1, 90],
        [0.9, 1, 90],
        [1, 1, 90],
        [1, 0.9, 180],
        [1, 0.8, 180],
        [1, 0.7, 180],
        [1, 0.6, 180],
        [1, 0.5, 180],
        [1, 0.4, 180],
        [1, 0.3, 180],
        [1, 0.2, 180],
        [1, 0.1, 180],
        [1, 0, 180],
        [0.9, 0, 270],
        [0.8, 0, 270],
        [0.7, 0, 270],
        [0.6, 0, 270],
        [0.5, 0, 270],
        [0.4, 0, 270],
        [0.3, 0, 270],
        [0.2, 0, 270],
        [0.1, 0, 270],
        [0, 0, 270]
    ])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    for i, pose in enumerate(poses):
        # make an observation with noise
        map.update(pose, (0, np.array([0.5, 0.5]) + np.random.randn(2) * 0.1))
        map.update(pose, (1, np.array([0.2, 0.7]) + np.random.randn(2) * 0.1))
        map._draw(ax)
        plt.pause(0.3)

    plt.show()
