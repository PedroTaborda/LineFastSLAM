from __future__ import annotations
import os
import pickle
import lzma
from dataclasses import dataclass, asdict

import numpy as np

import rosbags.rosbag1
import rosbags.serde

DEFAULT_SENSOR_DATA_DIR = os.path.join('data', 'sensor_data')
if not os.path.isdir('data'):
    os.mkdir('data')
if not os.path.isdir(DEFAULT_SENSOR_DATA_DIR):
    os.mkdir(DEFAULT_SENSOR_DATA_DIR)


@dataclass
class SensorData:
    ts: np.ndarray
    odometry: np.ndarray
    lidar: np.ndarray
    camera: list[list[tuple[int, float]]]
    comment: str = ''
    from_rosbag: bool = False


def load_sensor_data(filename: str, dir: os.PathLike=DEFAULT_SENSOR_DATA_DIR) -> SensorData:
    with lzma.open(os.path.join(dir, filename), 'rb') as f:
        data_dict = pickle.load(f)
    return SensorData(**data_dict)


def save_sensor_data(sensor_data: SensorData, filename: str, dir: os.PathLike=DEFAULT_SENSOR_DATA_DIR) -> None:
    data_dict = asdict(sensor_data)
    with lzma.open(os.path.join(dir, filename), 'wb') as f:
        pickle.dump(data_dict, f)


def detect_landmarks(image: np.ndarray) -> np.ndarray:
    return [(0, 20.0)]


def rosbag_to_data(rosbag_path: os.PathLike) -> SensorData:
    laser_ros = []
    odom_ros = []
    cam_ros = []
    with rosbags.rosbag1.Reader(rosbag_path) as reader:
        print(list(reader.topics.keys()))
        connections_laser = []
        connections_odom = []
        connections_cam = []
        for x in reader.connections:
            if x.topic == '/scan':
                connections_laser.append(x)
            elif x.topic == '/odom':
                connections_odom.append(x)
            elif x.topic == '/camera/image_raw':
                connections_cam.append(x)
        for connection, timestamp, rawdata in reader.messages(connections=connections_laser):
            msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            laser_ros.append((msg.ranges, timestamp))
        for connection, timestamp, rawdata in reader.messages(connections=connections_odom):
            msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            odom_ros.append((msg, timestamp))
        for connection, timestamp, rawdata in reader.messages(connections=connections_cam):
            msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            cam_ros.append((detect_landmarks(msg), timestamp))
    odom = ...
    lidar = ...
    landmarks = ...
    ts = ...
    return SensorData(ts=ts, odometry=odom, lidar=lidar, camera=landmarks, comment="From rosbag", from_rosbag=True)


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
