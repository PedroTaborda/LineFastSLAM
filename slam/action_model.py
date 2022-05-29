from dataclasses import dataclass

import numpy as np

@dataclass
class ActionModelSettings:
    uncertainty_type: str = 'additive'

    uncertainty_additive_mean: np.ndarray = np.zeros((3,))
    uncertainty_additive_covariance: np.ndarray = np.diag([0.005, 0.005, 0.1])

    uncertainty_multiplicative_mean: np.ndarray = np.zeros((2,))
    uncertainty_multiplicative_covariance: np.ndarray = np.diag([0.01, 0.1])

def action_model(state: np.ndarray, odometry: np.ndarray, 
                 settings: ActionModelSettings = ActionModelSettings()) -> np.ndarray:
    x, y, theta = state

    if settings.uncertainty_type == 'additive':
        linear_displacement, angular_displacement = odometry
        new_state = np.ndarray([x + linear_displacement*np.cos(np.deg2rad(theta)),
                                y + linear_displacement*np.sin(np.deg2rad(theta)),
                                theta + angular_displacement])
        new_state += np.random.multivariate_normal(settings.uncertainty_additive_mean, settings.uncertainty_additive_covariance)

    elif settings.uncertainty_type == 'multiplicative':
        odometry += np.random.multivariate_normal(settings.uncertainty_multiplicative_mean, settings.uncertainty_multiplicative_covariance)  
        linear_displacement, angular_displacement = odometry

        new_state = np.ndarray([x + linear_displacement*np.cos(np.deg2rad(theta)),
                                y + linear_displacement*np.sin(np.deg2rad(theta)),
                                theta + angular_displacement])
    else:
        raise ValueError('Action model uncertainty type must be of additive or multiplicative type.')
    
    return new_state