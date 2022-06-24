from __future__ import annotations
import numpy as np
from ekf.ekf import EKFSettings, EKF
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.patches import Ellipse
import math
import scipy.stats

def diff(rh_th1, rh_th2):
    return np.array([rh_th1[0] - rh_th2[0], np.mod(rh_th1[1] - rh_th2[1] + np.pi, 2*np.pi) - np.pi])


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
    z_handle.set(offsets=h_inv(z, [sensor_p, theta, n_gain]))

    plt.draw()



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

n_gain = np.diag([1, np.pi/6])
sensor_p, theta = np.array([5, 5]), np.pi/2
def obs():
    global z
    parameters = [sensor_p, theta, n_gain]
    z = h(p, parameters, rng.normal(size=(2,)))
    print(z, np.rad2deg(z[1]))
    myEKF.update(z, diff)
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


rng = np.random.default_rng()
# Initial position
x0 = np.array([10, 0])    
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

def h(x, parameters, n=None):    # x is landmark position
    s, theta, n_gain = parameters
    diff = x - s
    r = np.linalg.norm(diff)
    fi = np.arctan2(diff[1], diff[0]) - theta
    z = np.array([r, fi])
    if n is None:
        return z
    return z + n_gain @ n

def h_inv(z, parameters):
    s, theta, n_gain = parameters
    r, fi = z
    x = s[0] + r * np.cos(fi + theta)
    y = s[1] + r * np.sin(fi + theta)
    return np.array([x, y])

def get_Dhx(x, parameters):
    s, theta, n_gain = parameters
    dx, dy = x - s
    r = np.linalg.norm(x-s)
    return np.array([[dx/r,  dy/r], [-dy/r ** 2, dx/r ** 2]])

def get_Dhn(x, parameters):
    s, theta, n_gain = parameters
    return n_gain

p = x0
cov0 = np.diag([1, 3])
EKFsets = EKFSettings(x0, cov0, g, Dgx, Dgm)
myEKF = EKF(EKFsets)
myEKF.set_sensor_model(h, get_Dhx, get_Dhn)
myEKF.set_parameters([sensor_p, theta, n_gain])

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
est_ellipse: Ellipse = Ellipse((0, 0), 1, 1, facecolor='none', edgecolor='C00')
ax.add_patch(est_ellipse)
p_handle: PathCollection = ax.scatter(0, 0, marker='x', c='C01')
z_handle: PathCollection = ax.scatter(0, 0, marker='1', c='C02')
ax.scatter(sensor_p[0], sensor_p[1], marker='^', c='C02')
L = 30
ax.set_xlim([-L, L])
ax.set_ylim([-L, L])
# ax.axis('equal')
plot()
plt.show(block=False)
