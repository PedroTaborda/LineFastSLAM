from abc import ABC, abstractmethod
import copy
from typing import Collection

import numpy as np


class RansacModel(ABC):
    @abstractmethod
    def fit(self, data_points: Collection):
        """Calculates model parameters based on data_points.
        Enables instance to be used for checking for inliers.
        """
        ...

    @abstractmethod
    def inliers(self, data_points: Collection) -> bool:
        """Returns the data_points that are inliers to the model.
        """
        ...


def RANSAC(data_points: np.ndarray, model: RansacModel, n: int, k: int = 1000, t: float = 0.1, d: int = 10) -> RansacModel:
    """
    Random sample consensus algorithm.

    Args:
        dataPoints: A collection of data points.
        model: A model that can be used to check for inliers.
        n: The minimum number of data points required to fit the model.
        k: The maximum number of iterations.
        t: A threshold value for determining when a model fits sufficiently well to be accepted.
        d: The number of close data points required to assert that a model fits well to data.

    Returns:
        The final (best) model.
    """
    if not isinstance(model, RansacModel):
        raise TypeError("Model must be of type RansacModel")

    # Initialize the best model and its inlier set.
    best_model = model
    best_inliers = []
    data_sz = len(data_points)

    # Iterate for k iterations.
    for iteration in range(k):
        # Randomly select d data points.
        random_sample = data_points[np.random.choice(
            data_sz, n, replace=False), :]

        # Fit a model to the random sample.
        model.fit(random_sample)

        # Find inliers with respect to the model.
        inliers = model.inliers(data_points)

        if len(inliers) > d:
            # If the number of inliers is greater than d, then add the inliers to the best model.
            model.fit(inliers)
            inliers = model.inliers(data_points)

            # If the number of inliers is greater than the number of best inliers so far, update the best model and inliers.
            if len(inliers) > len(best_inliers):
                best_model = copy.copy(model)
                best_inliers = inliers
                if len(best_inliers) > len(data_points) * t:
                    break

    # Return the best model.
    return best_model, best_inliers
