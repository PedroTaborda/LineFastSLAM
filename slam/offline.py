import os
import time
from matplotlib import pyplot as plt

import numpy as np
from visualization_utils.mpl_video import to_video

import slam.fastslam as fs
import sensor_data.sensor_data as sd
import usim.umap

def there_is_data(data: sd.SensorData, idx_lidar, idx_camera, idx_odometry):
    return idx_lidar < len(data.lidar) or idx_camera < len(data.camera) or idx_odometry < len(data.odometry)

def slam_sensor_data(data: sd.SensorData, slam_settings: fs.FastSLAMSettings = fs.FastSLAMSettings(), images_dir = None, realtime: bool = False):
    if slam_settings.visualize is False:
        raise ValueError('Visualization must be enabled to use slam_sensor_data.')
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)
    axes: plt.Axes
    slammer = fs.FastSLAM(slam_settings, axes)
    i = j = -1
    k = 0
    t0_ros = [data.lidar[i+1][0], data.camera[j+1][0], data.odometry[k+1][0]][np.argmin([data.lidar[i+1][0], data.camera[j+1][0], data.odometry[k+1][0]])]

    def next_times():
        tlidar = data.lidar[i+1][0] if i+1 < len(data.lidar) else np.inf
        tcamera = data.camera[j+1][0] if j+1 < len(data.camera) else np.inf
        tod = data.odometry[k+1][0] if k+1 < len(data.odometry) else np.inf
        return [tlidar-t0_ros, tcamera-t0_ros, tod-t0_ros]

    it = -1
    t0 = time.time()
    if data.sim_data is not None:
        actual_map: usim.umap.UsimMap = data.sim_data.map
        for landmark_id in actual_map.landmarks:
            axes.scatter(actual_map.landmarks[landmark_id][0], actual_map.landmarks[landmark_id][1], c='k', marker='*')

        sim_i = 0
        sim_N = len(data.sim_data.robot_pose)
        ts = data.sim_data.sampling_time

    try:
        while there_is_data(data, i+1, j+1, k+1):
            if realtime:
                last_t = t if it >= 0 else t0_ros
                it = np.argmin(next_times())
                t = next_times()[it]/1e9
                time.sleep(max(0, (t-last_t)/1e9 - (time.time() - t0)))
            else:
                it = np.argmin(next_times())
                t = next_times()[it]/1e9
                

            if data.sim_data is not None:
                sim_i = round(t/ts)
                if sim_i >= sim_N:
                    break
                actual_pose = data.sim_data.robot_pose[sim_i]
                estimated_pose = slammer.pose_estimate()

            t0 = time.time()
            if it == 0: # Lidar data incoming
                i += 1
            elif it == 1: # Camera data incoming
                j += 1
                for obs in data.camera[j][1]:
                    id, landmark = obs
                    r, theta = landmark
                    slammer.make_observation(t, (id, np.array([r, theta])))
            elif it == 2: # Odometry data incoming
                k += 1
                theta0, x0, y0 = data.odometry[k-1][1]
                theta1, x1, y1 = data.odometry[k][1]
                odom = np.array([np.sqrt((x1-x0)**2 + (y1-y0)**2), (theta1-theta0)]).squeeze()
                if data.sim_data is not None:
                    slammer.perform_action(t, odom, actual_pose)
                else:
                    slammer.perform_action(t, odom)
                slammer.resample()

                if images_dir is not None:
                    plt.savefig(os.path.join(images_dir, f"{k:06d}.png"))
    except KeyboardInterrupt:
        print('\nKeyboard interrupt. Exiting...')
    finally:
        if images_dir is not None:
            to_video(images_dir, "slam.mp4", fps=10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--not-realtime", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--file", type=str, default='sim0.xz')
    parser.add_argument("--images-dir", type=str, default='images_slam')
    
    args = parser.parse_args()

    images_dir = os.path.join('data', args.images_dir)
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    
    slam_sensor_data(
        sd.load_sensor_data(args.file),
        slam_settings=fs.FastSLAMSettings(
            num_particles=50,
            visualize=not args.no_visualize,
            trajectory_trail=True,
        ),
        realtime=not args.not_realtime,
        images_dir=images_dir
    )