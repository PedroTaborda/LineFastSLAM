# Autonomous Systems IST course (P4 - 2021/2022)

This code presents an implementation of the FastSLAM algorithm.


# Packages

## ekf (Extended Kalman Filter)

An implementation of the Extended Kalman Filter (EKF) for nonlinear systems with nonlinear measurement models and gaussian noise.

### Usage:

    python -i -m ekf.test # Starts an interactive shell with the EKF package loaded, useful for testing
    

## slam (FastSLAM)
Implements the FastSLAM algorithm, along with its main components (the robot's action model, landmarks' model based on the EKF and the map).

It also features a module for running the algorithm in stored data (SensorData format).

### Usage:

    python -i -m slam.fastslam # Starts a test script for the FastSLAM algorithm
    python -m slam.offline --file <file> # Runs the FastSLAM algorithm on the data stored in the file data/sensor_data/<file>

## usim (micro simulator)

A bare-bones simulator for testing and validating the FastSLAM algorithm.

### Usage:
    
        python -m usim.usim # Starts the simulator, which can be controlled using the keyboard. At the end, the data is saved in SensorData format in the file data/sensor_data/sim<n>.xz

## sensor_data (storage and representation)

The `sensor_data` is an attempt to provide a minimalistic representation and storage solution for sensor data, which can be obtained either by an actual robot, in which case one can use this package to transform the data into this format, or by simulating with `usim`, where the data is representation is close to this format.

### Usage:

    python -m sensor_data.sensor_data --rosbag <file> # Converts the rosbag file <file> into the SensorData format, placing it in the file data/sensor_data/<file>, replacing its extension with .xz
