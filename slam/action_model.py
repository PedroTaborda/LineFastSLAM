from dataclasses import dataclass

import numpy as np

from enum import Enum


class UncertaintyType(Enum):
    POSE_ADD = 0
    ODOM_ADD = 1
    ODOM_MULT = 2


@dataclass
class ActionModelSettings:
    uncertainty_type: UncertaintyType = UncertaintyType.ODOM_MULT

    POSE_ADD_MU: np.ndarray = np.zeros((3,))
    POSE_ADD_COV: np.ndarray = np.square(np.diag([0.01, 0.01, 0.5*np.pi/180]))

    ODOM_ADD_MU: np.ndarray = np.zeros((2,))
    ODOM_ADD_COV: np.ndarray = np.square(np.diag([0.1, 0.3]))

    ODOM_MULT_MU: np.ndarray = np.array([1, 1])
    ODOM_MULT_COV: np.ndarray = np.square(np.diag([0.1, 0.1]))


def action_model(state: np.ndarray, odometry: np.ndarray,
                 settings: ActionModelSettings = ActionModelSettings()) -> np.ndarray:
    x, y, theta = state

    if settings.uncertainty_type == UncertaintyType.POSE_ADD:
        linear_displacement, angular_displacement = odometry
        new_state = np.array([x + linear_displacement*np.cos(theta),
                              y + linear_displacement*np.sin(theta),
                              theta + angular_displacement])
        new_state += np.random.multivariate_normal(settings.POSE_ADD_MU,
                                                   settings.POSE_ADD_COV)

    elif settings.uncertainty_type == UncertaintyType.ODOM_ADD:
        odometry += np.random.multivariate_normal(settings.ODOM_ADD_MU,
                                                  settings.ODOM_ADD_COV)
        linear_displacement, angular_displacement = odometry

        new_state = np.array([x + linear_displacement*np.cos(theta),
                              y + linear_displacement*np.sin(theta),
                              theta + angular_displacement])
    elif settings.uncertainty_type == UncertaintyType.ODOM_MULT:
        odometry *= np.random.multivariate_normal(settings.ODOM_MULT_MU,
                                                  settings.ODOM_MULT_COV)
        linear_displacement, angular_displacement = odometry

        new_state = np.array([x + linear_displacement*np.cos(theta),
                              y + linear_displacement*np.sin(theta),
                              theta + angular_displacement])
    else:
        raise ValueError('Action model type definition is invalid - is this code reachable?.')

    return new_state
