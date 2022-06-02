from __future__ import annotations
import os
import pickle
import lzma
from dataclasses import dataclass, asdict

import numpy as np
import cv2

import rosbags.rosbag1
import rosbags.serde

import usim.map


DEFAULT_SENSOR_DATA_DIR = os.path.join('data', 'sensor_data')
if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir(DEFAULT_SENSOR_DATA_DIR):
    os.mkdir(DEFAULT_SENSOR_DATA_DIR)


@dataclass
class SimulationData:
    sampling_time: float
    robot_pose: np.ndarray   # (N,3) array of robot poses (x,y,theta[rad])
    map: usim.map.Map
    def __post_init__(self):
        if type(self.map) is dict:
            self.map = usim.map.Map(**self.map)

@dataclass
class SensorData:
    odometry: list[tuple[int, np.ndarray]]        # (timestamp, [theta, x, y])
    lidar: list[tuple[int, np.ndarray]]           # (timestamp, [phi, r])
    camera: list[tuple[int, list[tuple[int, np.ndarray]]]]           # (timestamp, list[id, [phi, r]])
    comment: str = ''
    from_rosbag: bool = False
    sim_data: SimulationData = None

    def save(self, filename: str) -> None:
        """Save the sensor data to a file.

        Args:
            filename: The filename to save the data to.
        """
        save_sensor_data(self, filename)

    def __post_init__(self) -> None:
        if type(self.sim_data) is dict:
            self.sim_data = SimulationData(**self.sim_data)


def load_sensor_data(filename: str, dir: os.PathLike=DEFAULT_SENSOR_DATA_DIR) -> SensorData:
    with lzma.open(os.path.join(dir, filename), 'rb') as f:
        data_dict = pickle.load(f)
    return SensorData(**data_dict)


def save_sensor_data(sensor_data: SensorData, filename: str, dir: os.PathLike=DEFAULT_SENSOR_DATA_DIR) -> None:
    data_dict = asdict(sensor_data)
    with lzma.open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(data_dict, f)


def detect_landmarks(image: np.ndarray, camera_matrix: np.ndarray, distortion_coefficents) -> list[tuple[int, float]]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    if ids is None:
        return (image, [])

    angles = []
    distances = []
    for cornerset in corners:
        LA  = 0.083     # Physical size of the aruco markers. Should be an input parameter

        # Estimate the position of the aruco markers in world coordinates
        _, translation, _ = cv2.aruco.estimatePoseSingleMarkers(cornerset, LA, camera_matrix, distortion_coefficents)
        translation = translation.squeeze()

        # Use the position of the markers to get the distance and difference in heading to the robot
        distance = np.linalg.norm(translation)
        angle = np.rad2deg(np.arctan(-translation[0]/translation[2]))
        angles.append(angle)
        distances.append(distance)

        # Draw the aruco markers on the image
        corners = cornerset.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
		# convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

        cv2.putText(image, f'd: {distance:.3f}', (topLeft[0] - 50, topLeft[1] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        cv2.putText(image, f'angle: {angle:.1f}', (topLeft[0] - 50, topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    return (image, list(zip([id[0] for id in ids], [(distance, np.deg2rad(angle)) for angle, distance in zip(angles, distances)])))


def rosbag_to_data(rosbag_path: os.PathLike) -> SensorData:
    laser_ros = []
    odom_ros = []
    cam_ros_sim = []
    cam_ros_real = []
    camera_matrix = np.empty((3,3))
    distortion_coefficients = np.empty((5,))
    with rosbags.rosbag1.Reader(rosbag_path) as reader:
        connections_laser = []
        connections_odom = []
        connections_cam_sim = []
        connections_cam = []
        connections_cam_info = []
        for x in reader.connections:
            if x.topic == '/scan':
                connections_laser.append(x)
            elif x.topic == '/odom':
                connections_odom.append(x)
            elif x.topic == '/camera/image_raw':
                connections_cam_sim.append(x)
            elif x.topic == '/raspicam_node/image/compressed':
                connections_cam.append(x)
            elif x.topic == '/raspicam_node/camera_info':
                connections_cam_info.append(x)
        if len(connections_laser) != 0:
            for connection, timestamp, rawdata in reader.messages(connections=connections_laser):
                msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                laser_ros.append((timestamp, msg.ranges))
        if len(connections_odom) != 0:
            for connection, timestamp, rawdata in reader.messages(connections=connections_odom):
                msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                theta, x, y = msg.pose.pose.orientation.z*180.0/np.pi, msg.pose.pose.position.x, msg.pose.pose.position.y
                odom_ros.append((timestamp, np.array([theta, x, y])))
        if len(connections_cam_sim) != 0:
            raise NotImplementedError('Camera on simulation not implemented')
            for connection, timestamp, rawdata in reader.messages(connections=connections_cam_sim):
                msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                cam_ros_sim.append((detect_landmarks(msg), timestamp))
        if len(connections_cam_info) != 0:
            for connection, timestamp, rawdata in reader.messages(connections=connections_cam_info):
                msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                camera_matrix = msg.k.reshape((3,3))
                distortion_coefficients = msg.d
        if len(connections_cam) != 0:
            for connection, timestamp, rawdata in reader.messages(connections=connections_cam):
                msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                img = cv2.imdecode(msg.data, cv2.IMREAD_COLOR)
                _, landmarks = detect_landmarks(img, camera_matrix, distortion_coefficients)
                cam_ros_real.append((timestamp, landmarks))

    return SensorData(odometry=odom_ros, lidar=laser_ros, camera=cam_ros_real, comment='From rosbag', from_rosbag=True)

def rosbag_to_imgs(rosbag_path: os.PathLike) -> list[np.ndarray]:
    with rosbags.rosbag1.Reader(rosbag_path) as reader:
        connections_cam = []
        for x in reader.connections:
            if x.topic == '/raspicam_node/image/compressed':
                connections_cam.append(x)
        imgs = []
        for connection, timestamp, rawdata in reader.messages(connections=connections_cam):
            msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            img = cv2.imdecode(msg.data, cv2.IMREAD_COLOR)
            imgs.append(img)
    return imgs

def rosbag_camera_info(rosbag_path: os.PathLike) -> list[np.ndarray]:
    with rosbags.rosbag1.Reader(rosbag_path) as reader:
        connections_cam_info = []
        for x in reader.connections:
            if x.topic == '/raspicam_node/camera_info':
                connections_cam_info.append(x)
        camera_matrix = np.empty((3,3))
        distortion_coefficients = np.empty((5,))
        for connection, timestamp, rawdata in reader.messages(connections=connections_cam_info):
            msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            camera_matrix = msg.k.reshape((3,3))
            distortion_coefficients = msg.d
    return (camera_matrix, distortion_coefficients)

def list_to_data(sensor_data_lst: list[tuple[np.ndarray, list[tuple[int, float]], np.ndarray]], ts: float, comment: str = '') -> SensorData:
    odom: np.ndarray = np.array([sensor_data_lst_elem[0] for sensor_data_lst_elem in sensor_data_lst])
    lidar: np.ndarray = np.array([sensor_data_lst_elem[2] for sensor_data_lst_elem in sensor_data_lst])
    landmarks: list = [sensor_data_lst_elem[1] for sensor_data_lst_elem in sensor_data_lst]
    if not comment:
        comment = 'From list'
    else:
        comment += '\nFrom list'
    return SensorData(ts=ts, odometry=odom, lidar=lidar, camera=landmarks, comment=comment, from_rosbag=False)


def add_comment(comment: str, filename: str, dir: os.PathLike=DEFAULT_SENSOR_DATA_DIR) -> None:
    with lzma.open(os.path.join(dir, filename), 'rb') as f:
        data_dict = pickle.load(f)

    if data_dict['comment'] != '':
        data_dict['comment'] += '\n'
    else:
        data_dict['comment'] = comment

    with lzma.open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert sensor data from rosbag to sensor data')
    parser.add_argument('--rosbag', type=str, help='Path to rosbag file', required=True)

    args = parser.parse_args()

    if args.rosbag:
        sensor_data = rosbag_to_data(args.rosbag)
        lst = args.rosbag.split('.')
        lst[-1] = lst[-1].replace('bag', 'xz', 1)
        sensor_data.save('.'.join(lst))
        print('Saved to', '.'.join(lst))
