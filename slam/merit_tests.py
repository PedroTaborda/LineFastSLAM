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
import slam.mass as mass


def plot_map(estimated_map: sm.Map, trajectory: list[tuple[int, np.ndarray]], sensor_data: sd.SensorData, ax: plt.Axes,
             t0: float = 0, tf: float = np.inf):
    t_traj = np.array([traj[0] for traj in trajectory]) 
    t_lid = (np.array([lid[0] for lid in sensor_data.lidar]) - sensor_data.lidar[0][0])*1e-9

    pc_plot_handle: plt.Line2D = ax.plot([], [], markersize=0.1, linestyle='', marker='.', c='#000000', zorder=-10)[0]

    i_traj = 0
    for t_lid, pc_raw in sensor_data.lidar:
        t_lid = (t_lid - sensor_data.lidar[0][0])*1e-9
        if t_lid < t0 or t_lid > tf:
            continue
        while i_traj < t_traj.shape[0] - 1 and t_traj[i_traj] < t_lid:
            i_traj += 1
        if i_traj < 0:
            i_traj = 0

        instant_traj, pose = trajectory[i_traj]
        off.plot_pc(pc_plot_handle, pc_raw, pose)

    for landmark in estimated_map.landmarks.values():
        landmark.drawn = False

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

def get_corridor_length(map: sm.Map) -> float:
    lines = [landmark for id, landmark in map.landmarks.items() if id < 0]
    Nlines = len(lines)
    distlist = list()
    for i in range(Nlines):
        for j in range(i+1, Nlines):
            r1, a1 = lines[i].get_mu()
            r2, a2 = lines[j].get_mu()
            xy1 = r1 * np.array([np.cos(a1), np.sin(a1)])
            xy2 = r2 * np.array([np.cos(a2), np.sin(a2)])

            diff = lambda x,y: np.mod(x-y + np.pi/2, np.pi) - np.pi/2
            if abs(diff(a1, a2)) > np.deg2rad(15):
                distlist.append(-inf)
            elif abs(diff(a1, np.pi)) > np.deg2rad(15) and abs(diff(a1, -np.pi)) > np.deg2rad(15):
                distlist.append(-inf)
            elif abs(diff(a2, np.pi)) > np.deg2rad(15) and abs(diff(a2, -np.pi)) > np.deg2rad(15):
                distlist.append(-inf)
            else:
                distlist.append(np.linalg.norm(xy1-xy2))

    return max(distlist)

def get_corridor_width(map: sm.Map) -> float:
    return get_closest_dists(map, [1.70])[0]

def show_typical_dists(map : sm.Map):
    closest = get_closest_dists(map, [15.78, 1.70])

    print(f"The closest distances are {closest}")

def traj_mse(slam_result: fs.SLAMResult):
    actual_trajectory: list[tuple[float, np.ndarray]] = slam_result.actual_trajectory
    estimated_trajectory: list[tuple[float, np.ndarray]] = slam_result.trajectory

    def diff_t2(rh_th1, rh_th2):
        return np.block([rh_th1[:2] - rh_th2[:2], np.mod(rh_th1[2] - rh_th2[2] + np.pi, 2*np.pi) - np.pi])

    if len(actual_trajectory) != len(estimated_trajectory):
        raise ValueError(f"Actual trajectory and estimated trajectory have different lengths ({len(actual_trajectory)} and {len(estimated_trajectory)})")
    
    errorxyt = np.zeros((len(actual_trajectory), 3))
    time = np.zeros((len(actual_trajectory),))
    for idx, ((t, actual_pose), (_, estimated_pose)) in enumerate(zip(actual_trajectory, estimated_trajectory)):
        errorxyt[idx, :] = diff_t2(actual_pose, estimated_pose) # x
        time[idx] = t
    
    rmsexyt = np.sqrt(np.square(errorxyt).mean(axis=0))
    return time, errorxyt, rmsexyt


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='b401e317ef90bc8dea8fdda348700e6d6f55cc79')
    args = parser.parse_args()

    if False:
        FilePath = os.path.join('data', "slammed", args.file)
        result, settings = load_slam_result(FilePath)
        print(f"The map has {len(result.map.landmarks)} landmarks. ", end='')
        show_typical_dists(result.map)
    

    N = [2, 5, 10, 20, 50, 100]

    rmsexyt = np.zeros((len(N),3))
    
    def has_n(N):
        def has_this_n(slam_res, settings):
            return slam_res.actual_trajectory is not None and settings.num_particles == N;  
        return has_this_n

    for i, n in enumerate(N):
        res_tuples = mass.load_files_where(has_n(n))
        print(f"N={n}")

        rmsexyt_n = []
        for res, settings in res_tuples:
            _, _, mse = traj_mse(res)
            rmsexyt_n.append(mse)
            print(f"{mass.dif_repr(settings)}: mse= {rmsexyt_n[-1]}")

        rmsexyt[i] = np.mean(rmsexyt_n, axis=0)

    print(f"{rmsexyt}")
    fig, ax = plt.subplots(1, 1)

    linex = ax.plot(N, rmsexyt[:, 0], label="x", color="C00")[0]
    liney = ax.plot(N, rmsexyt[:, 1], label="y", color="C01")[0]
    ax.set_xscale('log')

    ax2 = ax.twinx()
    ax2.yaxis.tick_right()
    linet = ax2.plot(N, rmsexyt[:, 2], label="theta", color="C02")[0]

    # ax2.legend()
    # ax.legend()
    ax.set_xticklabels(N)
    ax.set_xticks(N)
    ax2.spines['right'].set_color("C02")
    ax2.yaxis.label.set_color("C02")
    ax2.tick_params(axis='y', colors='C02')

    ax.set_xlabel("$N$")
    ax.set_ylabel("RMSE (m)")
    ax2.set_ylabel("RMSE ($^o$)")

    ax.set_ylim([0, ax.get_ylim()[1]])
    ax2.set_ylim([0, 2])

    plt.legend((linex, liney, linet), ('x', 'y', 'theta'))
    # ax.set_position([0.1, 0.1, 0.87, 0.88])
    plt.show()
