import os
import time
from matplotlib import pyplot as plt

import numpy as np
from visualization_utils.mpl_video import to_video
import cv2

import slam.fastslam as fs
import sensor_data.sensor_data as sd
import usim.umap
from slam.lidar_lines import identify_lines


def there_is_data(data: sd.SensorData, idx_lidar, idx_camera, idx_odometry):
    return idx_lidar < len(data.lidar) or idx_camera < len(data.camera) or idx_odometry < len(data.odometry)


def plot_pc(pc_axes, scan: np.ndarray, pose: np.ndarray):
    ''' Adds new scan to point cloud map. 
    
        Args: 
            scan - (360,) radius of each degree from 0 to 359
            pose - (3,) robot pose, x, y and theta
    '''
    px, py, theta = pose
    good = np.where(scan > 0.01)[0]
    angles = good.astype(float)*np.pi/180
    cleaned_scan = scan[good]
    points_x = np.cos(angles + theta) * cleaned_scan + px
    points_y = np.sin(angles + theta) * cleaned_scan + py
    pc_axes.scatter(points_x, points_y, s=0.1, marker='.', c='#000000', zorder=-10)



def slam_sensor_data(data: sd.SensorData, slam_settings: fs.FastSLAMSettings = fs.FastSLAMSettings(),
                     images_dir=None, realtime: bool = False, show_images: bool = False):
    if slam_settings.visualize is False:
        raise ValueError('Visualization must be enabled to use slam_sensor_data.')

    if show_images:
        fig, (axes, cam_ax) = plt.subplots(2, 1, figsize=(10, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)

    #_, pc_ax = plt.subplots(1, 1, figsize=(10, 5))  # point cloud axes
    axes: plt.Axes
    slammer = fs.FastSLAM(slam_settings, axes)
    i = j = -1
    k = 0
    t0_ros = [data.lidar[i+1][0], data.camera[j+1][0], data.odometry[k+1][0]
              ][np.argmin([data.lidar[i+1][0], data.camera[j+1][0], data.odometry[k+1][0]])]

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
            break
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

            t0 = time.time()
            if it == 0:  # Lidar data incoming
                i += 1
                if t < 20 and data.sim_data is None:
                    continue
                scan = data.lidar[i][1]
                if data.sim_data is not None:  # TODO: re run sims and delete this code
                    scan[scan > 3.49] = 0.0
                lines = identify_lines(scan)
                for line in lines:
                    slammer.make_line_observation(t, (None, line))
                plot_pc(axes, scan, slammer.pose_estimate())

            elif it == 1:  # Camera data incoming
                j += 1
                if t < 20 and data.sim_data is None:
                    continue
                _, landmarks, CmpImg = data.camera[j]
                if CmpImg is not None and show_images:
                    cam_ax.clear()
                    Img = cv2.imdecode(CmpImg, cv2.IMREAD_COLOR)
                    cam_ax.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
                for id, z in landmarks:
                    r, theta, psi = z
                    #slammer.make_unoriented_observation(t, (id, np.array([r, theta])))
                    slammer.make_oriented_observation(t, (id, np.array([r, theta, psi])))
            elif it == 2:  # Odometry data incoming
                k += 1
                if t < 20 and data.sim_data is None:
                    continue
                theta0, x0, y0 = data.odometry[k-1][1]
                theta1, x1, y1 = data.odometry[k][1]
                diff = np.array([x1-x0, y1-y0]).flatten()
                # Rotate to frame of odom0 to use only relative info
                R = np.array([[np.cos(-theta0), -np.sin(-theta0)], [np.sin(-theta0), np.cos(-theta0)]]).squeeze()
                diff = R @ diff
                odom = np.block([diff, theta1-theta0])
                odom[2] = np.mod(odom[2] + np.pi, 2*np.pi) - np.pi

                slammer.resample()
                if data.sim_data is not None:
                    slammer.perform_action(t, odom, actual_pose)
                else:
                    slammer.perform_action(t, odom)

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
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument('--show_images', action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--file", type=str, default='sim0.xz')
    parser.add_argument("-N", type=int, default=50)
    parser.add_argument("--images-dir", type=str, default='images_slam')

    args = parser.parse_args()

    images_dir = os.path.join('data', args.images_dir)
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    slam_sensor_data(
        sd.load_sensor_data(args.file),
        slam_settings=fs.FastSLAMSettings(
            num_particles=args.N,
            visualize=not args.no_visualize,
            trajectory_trail=True,
        ),
        realtime=args.realtime,
        images_dir=images_dir,
        show_images=args.show_images
    )
