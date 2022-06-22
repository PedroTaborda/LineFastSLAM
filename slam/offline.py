import os
import shutil
import time
from matplotlib import pyplot as plt

import numpy as np
from visualization_utils.mpl_video import to_video
import cv2

import collections
import warnings

import slam.fastslam as fs
import slam.action_model as am
import sensor_data.sensor_data as sd
import usim.umap
from slam.lidar_lines import identify_lines


def there_is_data(data: sd.SensorData, idx_lidar, idx_camera, idx_odometry):
    return idx_lidar < len(data.lidar) or idx_camera < len(data.camera) or idx_odometry < len(data.odometry)


def plot_pc(pc_plot_handle, scan: np.ndarray, pose: np.ndarray):
    ''' Adds new scan to point cloud map. 

        Args: 
            scan - (360,) radius of each degree from 0 to 359
            pose - (3,) robot pose, x, y and theta
    '''
    px, py, theta = pose
    good = np.where(scan > 0.01)[0]
    angles = good.astype(float)*np.pi/180
    cleaned_scan = scan[good]
    if cleaned_scan.size > 0:
        points_x = np.cos(angles + theta) * cleaned_scan + px
        points_y = np.sin(angles + theta) * cleaned_scan + py
        X, Y = pc_plot_handle.get_data()
        pc_plot_handle.set_data(np.concatenate([X, points_x]), np.concatenate([Y, points_y]))


def slam_sensor_data(data: sd.SensorData, slam_settings: fs.FastSLAMSettings = fs.FastSLAMSettings(),
                     images_dir=None, realtime: bool = False, show_images: bool = False, stats_iter_size: int = 30,
                     save_every: int = 1, video_name: str = "slam", start_time: float = 0):
    #if slam_settings.visualize is False:
    #    raise ValueError('Visualization must be enabled to use slam_sensor_data.')

    if show_images:
        fig, (axes, cam_ax) = plt.subplots(2, 1, figsize=(10, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)

    # _, pc_ax = plt.subplots(1, 1, figsize=(10, 5))  # point cloud axes
    pc_ax = axes    # axes to draw point cloud map
    pc_plot_handle = pc_ax.plot([], [], markersize=0.1, linestyle='', marker='.', c='#000000', zorder=-10)[0]
    axes: plt.Axes
    slammer = fs.FastSLAM(slam_settings, axes)
    i = j = -1
    k = 0
    t0_ros = [data.lidar[i+1][0], data.camera[j+1][0], data.odometry[k+1][0]
              ][np.argmin([data.lidar[i+1][0], data.camera[j+1][0], data.odometry[k+1][0]])]
    total_time = ([data.lidar[-1][0], data.camera[-1][0], data.odometry[-1][0]
              ][np.argmax([data.lidar[-1][0], data.camera[-1][0], data.odometry[-1][0]])] - t0_ros)/1e9

    def next_times():
        tlidar = data.lidar[i+1][0] if i+1 < len(data.lidar) else np.inf
        tcamera = data.camera[j+1][0] if j+1 < len(data.camera) else np.inf
        tod = data.odometry[k+1][0] if k+1 < len(data.odometry) else np.inf
        return [tlidar-t0_ros, tcamera-t0_ros, tod-t0_ros]

    it = -1
    t0 = time.time()
    true_t0 = time.time()
    if data.sim_data is not None:
        actual_map: usim.umap.UsimMap = data.sim_data.map
        for landmark_id in actual_map.landmarks:
            break
            axes.scatter(actual_map.landmarks[landmark_id][0], actual_map.landmarks[landmark_id][1], c='k', marker='*')

        sim_i = 0
        sim_N = len(data.sim_data.robot_pose)
        ts = data.sim_data.sampling_time

    dt_iter = collections.deque([0.1], maxlen=stats_iter_size)
    dt_sel = collections.deque([0.1], maxlen=stats_iter_size)
    dt_lidar = collections.deque([0.1], maxlen=stats_iter_size)
    dt_camera = collections.deque([0.1], maxlen=stats_iter_size)
    dt_odometry = collections.deque([0.1], maxlen=stats_iter_size)
    dt_draw = collections.deque([0.1], maxlen=stats_iter_size)
    dt_save = collections.deque([0.1], maxlen=stats_iter_size)

    print_last_t = time.time()

    #print('\n\n')  # because print code starts by going two lines up
    try:
        while there_is_data(data, i+1, j+1, k+1):
            t0_iter = time.time()

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
            dt_sel.append(t0 - t0_iter)
            dt_lidar.append(0)
            dt_camera.append(0)
            dt_odometry.append(0)
            dt_draw.append(0)
            dt_save.append(0)
            if it == 0:  # Lidar data incoming
                i += 1
                if t < start_time:
                    continue
                scan = data.lidar[i][1]
                if data.sim_data is not None:  # TODO: re run sims and delete this code
                    scan[scan > 3.49] = 0.0
                lines = identify_lines(scan)
                for line in lines:
                    slammer.make_line_observation(t, (None, line))
                plot_pc(pc_plot_handle, scan, slammer.pose_estimate())
                dt_lidar[-1] = time.time() - t0

            elif it == 1:  # Camera data incoming
                j += 1
                if t < start_time:
                    continue
                _, landmarks, CmpImg = data.camera[j]
                if CmpImg is not None and show_images:
                    cam_ax.clear()
                    Img = cv2.imdecode(CmpImg, cv2.IMREAD_COLOR)
                    cam_ax.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
                for id, z in landmarks:
                    r, theta, psi = z
                    slammer.make_unoriented_observation(t, (id, np.array([r, theta])))
                    #slammer.make_oriented_observation(t, (id, np.array([r, theta, psi])))

                dt_camera[-1] = time.time() - t0
            elif it == 2:  # Odometry data incoming
                k += 1
                if t < start_time:
                    continue
                theta0, x0, y0 = data.odometry[k-1][1]
                theta1, x1, y1 = data.odometry[k][1]
                pos_diff = np.array([x1-x0, y1-y0]).flatten()
                # Rotate to frame of odom0 to use only relative info
                R = np.array([[np.cos(-theta0), -np.sin(-theta0)], [np.sin(-theta0), np.cos(-theta0)]]).squeeze()
                pos_diff = R @ pos_diff
                odom = np.block([pos_diff, theta1-theta0])
                odom[2] = np.mod(odom[2] + np.pi, 2*np.pi) - np.pi

                slammer.resample()
                if data.sim_data is not None:
                    slammer.perform_action(t, odom, actual_pose)
                else:
                    slammer.perform_action(t, odom)

                t1 = time.time()
                dt_odometry[-1] = t1 - t0

                #if slam_settings.visualize:
                #    slammer._draw()
                # fig.canvas.draw()
                t2 = time.time()
                dt_draw[-1] = t2 - t1

                if images_dir is not None and (k % save_every == 0):
                    plt.savefig(os.path.join(images_dir, f"{k:06d}.png"))

                t3 = time.time()
                dt_save[-1] = t3 - t2

            tf = time.time()
            dt_iter.append(tf - t0)
            # Means only of iterations where operation was executed
            t_iter_mean = np.mean(dt_iter)
            t_sel_mean = np.mean(dt_sel)
            with warnings.catch_warnings():
                # Mean of empty slices returns nan and causes these warnings
                # Replace nan with 0
                warnings.filterwarnings('ignore', r'Mean of empty slice.')
                warnings.filterwarnings('ignore', r'invalid value encountered in double_scalars')
                t_lidar_mean = np.nan_to_num(np.mean([dt for dt in dt_lidar if dt != 0]))
                t_cam_mean = np.nan_to_num(np.mean([dt for dt in dt_camera if dt != 0]))
                t_odom_mean = np.nan_to_num(np.mean([dt for dt in dt_odometry if dt != 0]))
                t_draw_mean = np.nan_to_num(np.mean([dt for dt in dt_draw if dt != 0]))
                t_save_mean = np.nan_to_num(np.mean([dt for dt in dt_save if dt != 0]))

            # Percentages counting all iterations
            t_iter_total = np.sum(dt_iter)
            t_sel_percentage = 100*np.sum(dt_sel)/t_iter_total
            t_lidar_percentage = 100*np.sum(dt_lidar)/t_iter_total
            t_cam_percentage = 100*np.sum(dt_camera)/t_iter_total
            t_odom_percentage = 100*np.sum(dt_odometry)/t_iter_total
            t_draw_percentage = 100*np.sum(dt_draw)/t_iter_total
            t_save_percentage = 100*np.sum(dt_save)/t_iter_total

            #slammer._draw_map()
            #slammer._draw_location()
            if tf - print_last_t > 0.5:
                print_last_t = tf
                # \033[<N> A move cursor N lines up; \033[K clear until end of line
                print(
                    f"Iteration {i+j+k:06d}: Averages [ms]: total:{int(1000*t_iter_mean):04d} sel:{int(1000*t_sel_mean):04d}"
                    f" lidar:{int(1000*t_lidar_mean):04d} cam:{int(1000*t_cam_mean):04d} odom:{int(1000*t_odom_mean):04d}"
                    f" draw:{int(1000*t_draw_mean):04d} save:{int(1000*t_save_mean):04d}\033[K")
                print(
                    f"Ratios: sel:{t_sel_percentage:3.1f}% lidar:{t_lidar_percentage:3.1f}% cam:{t_cam_percentage:3.1f}% odom:{t_odom_percentage:3.1f}% draw:{t_draw_percentage:3.1f}% save:{t_save_percentage:3.1f}%\033[K")
                print(
                    f"Run time: {time.time()-true_t0:.1f}s Completed: {(t-start_time)/total_time*100:3.1f}%\033[K"
                )
                print('\033[3A', end='')
    except KeyboardInterrupt:
        print('\nKeyboard interrupt. Exiting...')
    finally:
        print('\n\n')
        if video_name is not None and images_dir is not None:
            to_video(images_dir, video_name + ".mp4", fps=10)
    if slam_settings.visualize:
        slammer._draw_map()
        slammer._draw_location()
        plt.show(block=True)
    return slammer.end()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument('--show_images', action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--file", type=str, default='sim0.xz')
    parser.add_argument("-N", type=int, default=50)
    parser.add_argument("--video-name", type=str, default='slam')
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("-t0", type=float, default=0)

    args = parser.parse_args()

    images_dir = os.path.join('data', args.video_name + "_tmpimgs")
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)
    slam_settings=fs.FastSLAMSettings(
        action_model_settings=am.ActionModelSettings(
            action_type=am.ActionType.FREE,
        ),
        num_particles=args.N,
        visualize=not args.no_visualize,
        trajectory_trail=True,
    )
    images_dir = None
    res = slam_sensor_data(
        sd.load_sensor_data(args.file),
        slam_settings=slam_settings,
        realtime=args.realtime,
        images_dir=images_dir,
        show_images=args.show_images,
        save_every=args.save_every,
        video_name=args.video_name,
        start_time=args.t0
    )
    print(res)
