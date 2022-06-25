from __future__ import annotations
from cmath import inf
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sensor_data.sensor_data as sd
import slam.offline as off
import slam.fastslam as fs
import slam.map as sm


def plot_map(estimated_map: sm.Map, trajectory: list[tuple[int, np.ndarray]], sensor_data: sd.SensorData, ax: plt.Axes):
    t_traj = np.array([traj[0] for traj in trajectory]) 
    t_lid = (np.array([lid[0] for lid in sensor_data.lidar]) - sensor_data.lidar[0][0])*1e-9

    pc_plot_handle: plt.Line2D = ax.plot([], [], markersize=0.1, linestyle='', marker='.', c='#000000', zorder=-10)[0]

    i_traj = 0
    for t_lid, pc_raw in sensor_data.lidar:
        t_lid = (t_lid - sensor_data.lidar[0][0])*1e-9
        while i_traj < t_traj.shape[0] and t_traj[i_traj] < t_lid:
            i_traj += 1
        i_traj -= 1
        if i_traj < 0:
            i_traj = 0

        instant_traj, pose = trajectory[i_traj]
        off.plot_pc(pc_plot_handle, pc_raw, pose)

    estimated_map._draw(ax, color_ellipse='C01', color_p='C01', color_z='C01')
    ax.axis('equal')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    ax.relim()
    ax.autoscale_view(False, True, True)


def load_slam_result(path: os.PathLike) -> tuple[fs.SLAMResult, fs.FastSLAMSettings]:
    with open(path, 'rb') as f:
        result_tuple = pickle.load(f)
    return result_tuple

def line_distance(line1, line2):
    r1, a1 = line1 # polar coords of closest point to origin
    r2, a2 = line2
    xy1 = r1 * np.array([np.cos(a1), np.sin(a1)])
    xy2 = r2 * np.array([np.cos(a2), np.sin(a2)])
    # Orientation difference can only be in [-pi/2, pi/2]
    a_diff = np.mod(a2-a1 + np.pi/2, np.pi) - np.pi/2
    if a_diff > np.deg2rad(15):
        return inf
    return np.linalg.norm(xy1 - xy2)
    

def get_line_distances(map: sm.Map) -> np.ndarray:
    lines = [landmark for id, landmark in map.landmarks.items() if id < 0]
    Nlines = len(lines)
    distlist = list()
    for i in range(Nlines):
        for j in range(i+1, Nlines):
          distlist.append(line_distance(lines[i].get_mu(), lines[j].get_mu()))
    # print(distlist)
    return np.array(distlist)

def get_closest_dists(map: sm.Map, dist_list : list[float]):
    line_dists = get_line_distances(map)
    closest_dists = [min(line_dists, key=lambda x: np.abs(x-d)) for d in dist_list]
    return closest_dists


def show_typical_dists(map : sm.Map):
    closest = get_closest_dists(map, [15.78, 1.70])

    print(f"The closest distances are {closest}")
    
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='b401e317ef90bc8dea8fdda348700e6d6f55cc79')
    args = parser.parse_args()

    FilePath = os.path.join('data', "slammed", args.file)
    result, settings = load_slam_result(FilePath)
    print(f"The map has {len(result.map.landmarks)} landmarks. ", end='')
    show_typical_dists(result.map)
