from __future__ import annotations
import os
import pickle
import lzma
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares
import cv2

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
    odometry: np.ndarray        # [theta, x, y]
    lidar: np.ndarray
    camera: list[list[tuple[int, float]]]
    camera_matrix: np.ndarray
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


def detect_landmarks(image: np.ndarray, camera_matrix: np.ndarray, distortion_coefficents) -> list[tuple[int, float]]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    if ids is None:
        return (image, [])
    #fov = 2*62.2
    angles = []
    distances = []
    #degree_per_px = fov / image.shape[1]
    #degrees_of_px = lambda px: - degree_per_px * (px - image.shape[1] / 2)
    for cornerset in corners:
        '''cornersettemp = cornerset
        cornerset = cornerset.squeeze()
        mean_horizontal_px = np.mean(cornerset[:, 0])
        angle = degrees_of_px(mean_horizontal_px)
        angles.append(angle)'''

        LA  = 0.083
        _, translation, _ = cv2.aruco.estimatePoseSingleMarkers(cornerset, LA, camera_matrix, distortion_coefficents)
        translation = translation.squeeze()

        distance = np.linalg.norm(translation)
        angle = np.rad2deg(np.arctan(-translation[0]/translation[2]))
        angles.append(angle)
        distances.append(distance)

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

    return (image, list(zip(angles, distances, [id[0] for id in ids])))


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
                laser_ros.append((msg.ranges, timestamp))
        if len(connections_odom) != 0:
            for connection, timestamp, rawdata in reader.messages(connections=connections_odom):
                msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
                theta, x, y = msg.pose.pose.orientation.z*180.0/np.pi, msg.pose.pose.position.x, msg.pose.pose.position.y
                odom_ros.append(((theta, x, y), timestamp))
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
                image, landmarks = detect_landmarks(img, camera_matrix, distortion_coefficients)
                cam_ros_real.append((landmarks, timestamp))

    cam_ros = cam_ros_real if cam_ros_real else cam_ros_sim
    laser_times_ns: list[int] = np.array([x[1] for x in laser_ros])
    odom_times_ns: list[int] = np.array([x[1] for x in odom_ros])
    cam_ros_times_ns: list[int] = np.array([x[1] for x in cam_ros])
    times = [
        laser_times_ns,
        odom_times_ns,
        cam_ros_times_ns,
    ]
    sample_times_approx = [
        np.mean(np.diff(laser_times_ns)) if len(laser_ros) != 0 else np.inf,
        np.mean(np.diff(odom_times_ns)) if len(odom_ros) != 0 else np.inf,
        np.mean(np.diff(cam_ros_times_ns)) if len(cam_ros) != 0 else np.inf,
    ]
    ts = np.min(sample_times_approx)
    t0 = times[np.argmin(sample_times_approx)][0]
    tf = times[np.argmin(sample_times_approx)][-1]
    N = int((tf - t0) / ts)
    odom = np.zeros((N, 3))
    laser = np.zeros((N, 360))
    cam = np.zeros((N,), dtype=object)
    for i in range(N):
        t = t0 + ts*i
        i_odom = np.argmin(np.abs(odom_times_ns - t))
        i_laser = np.argmin(np.abs(laser_times_ns - t))
        i_cam = np.argmin(np.abs(cam_ros_times_ns - t))
        odom[i] = odom_ros[i_odom][0]
        laser[i] = laser_ros[i_laser][0]
        cam[i] = cam_ros[i_cam][0]
    return SensorData(ts=ts*1e-9, odometry=odom, lidar=laser, camera=cam, camera_matrix=camera_matrix, comment='From rosbag', from_rosbag=True)

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
