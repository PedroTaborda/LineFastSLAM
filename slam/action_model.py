from dataclasses import dataclass

import numpy as np

from enum import Enum


class UncertaintyType(Enum):
    POSE_ADD = 0
    ODOM_ADD = 1
    ODOM_MULT = 2


class ActionType(Enum):
    TANGENT = 0         # Moves only in theta direction
    TANGENT_CORR = 1    # Moves only in theta + delta_theta/2 direction
    FREE = 2            # Moves in any direction


@dataclass
class ActionModelSettings:
    uncertainty_type: UncertaintyType = UncertaintyType.ODOM_MULT
    action_type: ActionType = ActionType.FREE

    POSE_ADD_MU: np.ndarray = np.zeros((3,))     # adds to x, y, theta
    POSE_ADD_COV: np.ndarray = np.square(np.diag([0.01, 0.01, 0.5*np.pi/180]))

    ODOM_ADD_MU: np.ndarray = np.zeros((2,))    # adds to r, delta_theta
    ODOM_ADD_COV: np.ndarray = np.square(np.diag([0.1, 10*np.pi/180]))

    ODOM_MULT_MU: np.ndarray = np.array([1, 1])  # multiplies with r, delta_theta
    ODOM_MULT_COV: np.ndarray = np.square(np.diag([0.1, 0.1]))


def action_model(state: np.ndarray, odometry: np.ndarray,
                 settings: ActionModelSettings = ActionModelSettings()) -> np.ndarray:
    x, y, theta = state
    odom_forward, odom_left, odom_theta = odometry  # state change in robot frame

    if settings.action_type == ActionType.TANGENT:
        delta_theta = odom_theta
        distance_moved = np.linalg.norm(odometry[0:2])
        delta_pos = distance_moved * np.array([np.cos(theta), np.sin(theta)]) * np.sign(odom_forward)
    elif settings.action_type == ActionType.TANGENT_CORR:
        delta_theta = odom_theta
        distance_moved = np.linalg.norm(odometry[0:2])
        delta_pos = distance_moved * np.array([np.cos(theta + delta_theta/2),
                                              np.sin(theta + delta_theta/2)]) * np.sign(odom_forward)
    elif settings.action_type == ActionType.FREE:
        delta_pos_robot_frame = odometry[0:2]
        # Rotate to frame of world
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        delta_pos = R @ delta_pos_robot_frame
        delta_theta = odom_theta
    else:
        raise ValueError('Action model ActionType definition is invalid - is this code reachable?')

    if settings.uncertainty_type == UncertaintyType.POSE_ADD:
        new_state = np.array([x + delta_pos[0],
                              y + delta_pos[1],
                              theta + delta_theta])
        new_state += np.random.multivariate_normal(settings.POSE_ADD_MU,
                                                   settings.POSE_ADD_COV)

    elif settings.uncertainty_type == UncertaintyType.ODOM_ADD:
        r_noise, delta_theta_noise = np.random.multivariate_normal(
            settings.ODOM_ADD_MU,
            settings.ODOM_ADD_COV
        )
        delta_pos *= (1 + r_noise/np.linalg.norm(delta_pos))
        delta_theta += delta_theta_noise
        new_state = np.array([x + delta_pos[0],
                              y + delta_pos[1],
                              theta + delta_theta])
    elif settings.uncertainty_type == UncertaintyType.ODOM_MULT:
        r_factor, delta_theta_factor = np.random.multivariate_normal(
            settings.ODOM_MULT_MU,
            settings.ODOM_MULT_COV
        )
        delta_pos *= r_factor
        delta_theta *= delta_theta_factor
        new_state = np.array([x + delta_pos[0],
                              y + delta_pos[1],
                              theta + delta_theta])
    else:
        raise ValueError('Action model UncertaintyType definition is invalid - is this code reachable?')

    return new_state
