from dataclasses import dataclass
import math
import copy

import scipy.stats
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.patches import Ellipse

from ekf.ekf import EKF, EKFSettings

print(f"[WARNING] Unrealistic landmarks.")


@dataclass
class LandmarkSettings(EKFSettings):
    """Settings for the EKF representing a landmark.
    
    A landmark is represented by its position in the xy plane.
    By default, there is a linear measurement model, but this can be
    changed by setting the `h` and `Dh_` functions at measurement time.
    """
    mu0: np.ndarray = np.array([0, 0])
    cov0: np.ndarray = np.diag([0.1, 0.1])  # also wrong but enough for now
    g: callable = lambda x, u, m: x
    get_Dgx: callable = lambda x, u, m: np.eye(2)
    get_Dgm: callable = lambda x, u, m: np.zeros((2, 2))


r = 0.2  # std_dev of default linear observation model


@dataclass
class Observation():
    """ Observation of a landmark.
    Observation model:
        z = h(x, n)
        where
        x - landmark position (unknown)
        n - multi-normal observation noise with identity covariance matrix,
                0 mean and same dimensions as z
        z - observations (known)

        h - invertible and differentiable function
    """
    landmark_id: int = 0
    z: np.ndarray = np.array([0, 0])
    h: callable = lambda x, n: x + r * np.eye(2) @ n
    h_inv: callable = lambda z: z
    get_Dhx: callable = lambda x, n: np.eye(2)
    get_Dhn: callable = lambda x, n: r * np.eye(2)

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

    def update(self, obs: Observation):
        if obs.landmark_id not in self.landmarks:
            x0 = obs.h_inv(obs.z)
            Dhn = obs.get_Dhn(x0, np.zeros_like(obs.z))
            Dhx_inv = np.linalg.inv(obs.get_Dhx(x0, np.zeros_like(obs.z)))
            self.landmarks[obs.landmark_id] = Landmark(
                LandmarkSettings(
                    mu0=x0,
                    cov0=Dhx_inv @ Dhn @ Dhn.T @ Dhx_inv.T
                ),
                mu0=x0
            )
            return 1.0
        else:
            self.landmarks[obs.landmark_id].set_sensor_model(obs.h, obs.get_Dhx, obs.get_Dhn)
            likelyhood = self.landmarks[obs.landmark_id].get_likelihood(obs.z)
            self.landmarks[obs.landmark_id].predict()
            self.landmarks[obs.landmark_id].update(obs.z)
            return likelyhood
    def _draw(self, ax, **plot_kwargs):
        for landmark in self.landmarks.values():
            landmark._draw(ax, **plot_kwargs)

    def copy(self):
        return copy.copy(self)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()

    # Create a map
    map = Map()

    # Create a figure
    fig, ax = plt.subplots(1, 1)

    # Draw the map
    map._draw(ax)

    # std deviation of r and fi, noise cov is n_gain @ n_gain.T
    n_gain = np.diag([2, 1])

    poses = np.array([
        [0, 0.1, 90],
        [0, 0.2, 90],
        [0, 0.3, 90],
        [0, 0.4, 90],
        [0, 0.5, 90],
        [0, 0.6, 90],
        [0, 0.7, 90],
        [0, 0.8, 90],
        [0, 0.9, 90],
        [0, 1, 0],
        [0.1, 1, 0],
        [0.2, 1, 0],
        [0.3, 1, 0],
        [0.4, 1, 0],
        [0.5, 1, 0],
        [0.6, 1, 0],
        [0.7, 1, 0],
        [0.8, 1, 0],
        [0.9, 1, 0],
        [1, 1, 0],
        [1, 0.9, 270],
        [1, 0.8, 270],
        [1, 0.7, 270],
        [1, 0.6, 270],
        [1, 0.5, 270],
        [1, 0.4, 270],
        [1, 0.3, 270],
        [1, 0.2, 270],
        [1, 0.1, 270],
        [1, 0, 270],
        [0.9, 0, 180],
        [0.8, 0, 180],
        [0.7, 0, 180],
        [0.6, 0, 180],
        [0.5, 0, 180],
        [0.4, 0, 180],
        [0.3, 0, 180],
        [0.2, 0, 180],
        [0.1, 0, 180],
        [0, 0, 180]
    ])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    for i, pose in enumerate(poses):
        px, py, theta = pose
        plt.scatter(px, py, marker=(3, 0, theta-90), c='r')

        def h(x, n):    # x is landmark position
            diff = x - np.array([px, py])
            r = np.linalg.norm(diff)
            fi = np.arctan2(diff[1], diff[0]) - theta
            z = np.array([r, fi])
            return z + n_gain @ n

        def h_inv(z):
            r, fi = z
            x = px + r * np.cos(fi + theta)
            y = py + r * np.sin(fi + theta)
            return np.array([x, y])

        def get_Dhx(x, n):
            px, py = x
            r = np.linalg.norm(x)
            return np.array([[px/r,  py/r], [-py/r ** 2, px/r ** 2]])

        def get_Dhn(x, n):
            return n_gain
        landmark_id = 0

        x_real_landmark_0 = np.array([0.5, 0.5])
        # make an observation with noise
        z = h(x_real_landmark_0, rng.normal(size=(2,)))
        obs = Observation(landmark_id=0, z=z, h=h, h_inv=h_inv, get_Dhx=get_Dhx, get_Dhn=get_Dhn)
        map.update(obs)

        obs.landmark_id = 1
        x_real_landmark_1 = np.array([0.7, 0.7])
        # make an observation with noise
        z = h(x_real_landmark_1, rng.normal(size=(2,)))
        map.update(obs)
        map._draw(ax)
        plt.pause(0.3)

    plt.show()
