from __future__ import annotations
import numpy as np
from ekf.ekf import EKFSettings, EKF
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Ellipse
import math
import scipy.stats


def plot():
    #global p, myEKF
    mu = myEKF.get_mu()
    cov = myEKF.get_cov()
    confidence_interval = 0.99

    norm_dist = scipy.stats.norm
    # number of std's to include in cobfidence ellipse
    n_stds = -norm_dist.ppf((1-confidence_interval)/2)

    # Plot ellipse
    est_ellipse.set_center(mu)
    [w, v] = np.linalg.eig(cov)
    est_ellipse.set_width(np.sqrt(w[0])*n_stds*2)
    est_ellipse.set_height(np.sqrt(w[1])*n_stds*2)
    angle_deg = math.atan2(v[1, 0], v[0, 0]) * 180/np.pi
    est_ellipse.set_angle(angle_deg)

    # Plot real position
    p_handle.set(offsets=p)

    # Plot last observation
    z_handle.set(offsets=z)

    plt.draw()


n_gain = np.diag([1, 4])
parameters = [np.eye(2), n_gain]
def act():
    global p, z

    # just to remove z from visible area
    z = np.array([-999, -999])
    # Rotation  degrees per action
    a = 15*np.pi/180
    A = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    m_gain = np.array([[0.9, 1.2], [1.2, 0.3]])
    u = [A, m_gain]
    p = g(p, u, rng.normal(size=(2,)))
    myEKF.predict(u)
    plot()


def obs():
    global z

    z = h(p, parameters, rng.normal(size=(2,)))
    myEKF.update(z)
    plot()


def both():
    act()
    obs()


def auto(N=500, T=1e-9, obs_p=0):
    while(N > 0):
        N = N - 1
        if obs_p == 0:
            both()
        elif(rng.uniform() < obs_p):
            obs()
        else:
            act()
        plt.pause(T)


if __name__ == "__main__":
    rng = np.random.default_rng()
    # Initial position
    p = np.array([10, 0])
    z = np.array([-999, -999])

    def g(x, u, m=None):
        A, m_gain = u
        if m is None: return A @ x
        return A @ x  + m_gain @ m

    def Dgx(x, u):
        A, m_gain = u
        return A
    def Dgm(x, u):
        A, m_gain = u
        return m_gain
    def h(x, parameters, n=None):
        C, n_gain = parameters 
        if n is None: return C@x
        return C@x + n_gain @ n
    def Dhx(x, parameters):
        C, n_gain = parameters 
        return C
    def Dhn(x, parameters):
        C, n_gain = parameters
        return n_gain

    x0 = p
    cov0 = np.diag([1, 3])
    EKFsets = EKFSettings(x0, cov0, g, Dgx, Dgm)
    myEKF = EKF(EKFsets)
    myEKF.set_sensor_model(h, Dhx, Dhn)
    myEKF.set_parameters([np.eye(2), n_gain])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    est_ellipse: Ellipse = Ellipse((0, 0), 1, 1, facecolor='none', edgecolor='C00')
    ax.add_patch(est_ellipse)
    p_handle: PathCollection = ax.scatter(0, 0, marker='x', c='C01')
    z_handle: PathCollection = ax.scatter(0, 0, marker='1', c='C02')
    L = 30
    ax.set_xlim([-L, L])
    ax.set_ylim([-L, L])
    # ax.axis('equal')
    plot()
    plt.show(block=False)
    print("Run in interactive mode and use above functions")
