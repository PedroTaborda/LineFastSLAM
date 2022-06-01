import time

import numpy as np
from slam.action_model import ActionModelSettings

import slam.fastslam as fs
import sensor_data.sensor_data as sd

def there_is_data(data: sd.SensorData, idx_lidar, idx_camera, idx_odometry):
    return idx_lidar < len(data.lidar) or idx_camera < len(data.camera) or idx_odometry < len(data.odometry)

def slam_sensor_data(data: sd.SensorData, slam_settings: fs.FastSLAMSettings = fs.FastSLAMSettings(), realtime: bool = False):
    slammer = fs.FastSLAM(slam_settings)
    visualization = slam_settings.visualize

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
    while there_is_data(data, i+1, j+1, k+1):
        if realtime:
            last_t = t if it >= 0 else t0_ros
            it = np.argmin(next_times())
            t = next_times()[it]
            time.sleep(max(0, (t-last_t)/1e9 - (time.time() - t0)))
        else:
            it = np.argmin(next_times())

        t0 = time.time()
        if it == 0:
            i += 1
            print(data.lidar[i][0]/1e9, "lidar")
        elif it == 1:
            j += 1
            print(data.camera[j][0]/1e9, "camera")
            print(f"camera data: {data.camera[j][1]}")
            for obs in data.camera[j][1]:
                id, landmark = obs
                theta, r = landmark
                xr, yr, tr = slammer.get_location()
                xy_observation = np.array([xr + r*np.cos(np.deg2rad(theta + tr)),
                                       yr + r*np.sin(np.deg2rad(theta + tr))])
                print(f'Landmark position {xy_observation}')
                slammer.make_observation((id, xy_observation))
        elif it == 2:
            k += 1
            print(data.odometry[k][0]/1e9, "odometry")
            #print(f"odometry data: {data.odometry[k][1]}")
            theta0, x0, y0 = data.odometry[k-1][1]
            theta1, x1, y1 = data.odometry[k][1]
            print(f"odometry data: {x0, y0, theta0, x1, y1, theta1}")
            odom = np.array([np.sqrt((x1-x0)**2 + (y1-y0)**2), (theta1-theta0)]).squeeze()
            print(f"odometry: {odom}")
            slammer.perform_action(odom)

        if j % 10 == 0:
            slammer.resample()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--not-realtime", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--file", type=str, default='camera-unmeasured.xz')
    
    args = parser.parse_args()

    slam_sensor_data(
        sd.load_sensor_data(args.file),
        slam_settings=fs.FastSLAMSettings(
            num_particles=10,
            visualize=not args.no_visualize
        ),
        realtime=not args.not_realtime
    )