import rosbags.rosbag1
import rosbags.serde

import numpy as np

with rosbags.rosbag1.Reader('1.bag') as reader:
    print(list(reader.topics.keys()))
    connections_laser = []
    for x in reader.connections:
        if x.topic == '/scan':
            connections_laser.append(x)
    for connection, timestamp, rawdata in reader.messages(connections=connections_laser):
        msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)

