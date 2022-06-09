from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats


@dataclass
class EKFSettings:
    """
    Settings for the EKF.
    """
    mu0: np.ndarray     # Initial expected value for x
    cov0: np.ndarray    # Initial covariance matrix for x
    g: callable         # g
    get_Dgx: callable       # Jacobian of g with regard to x
    get_Dgm: callable       # Jacobian of g with regard to m

    min_cov: np.ndarray = None


class EKF:
    """ EKF implementation. Assuming a system where
        x(k+1) = g(x(k), u(k+1), m(k+1))
        z(k) = h(x(k), n(k))
        x - state (unknown)
        u - input (known)
        m - multi-normal process noise with identity covariance matrix,
                0 mean and same dimensions as x
        n - multi-normal observation noise with identity covariance matrix,
                0 mean and same dimensions as z
        z - observations (known)


        g and h - non-linear, differentiable functions      

        Set sensor model before running update or getting likelihood
    """

    def __init__(self, settings: EKFSettings) -> None:
        self.g: callable = settings.g
        self.get_Dgx: callable = settings.get_Dgx
        self.get_Dgm: callable = settings.get_Dgm
        self.mu: np.ndarray = settings.mu0
        self.cov: np.ndarray = settings.cov0
        self.h: callable = None
        self.get_Dhx: callable = None
        self.get_Dhn: callable = None
        self.min_cov = settings.min_cov

    def predict(self, u):
        zero_m = np.zeros_like(self.mu)
        # Get sensitivity to uncertainty
        Dgx = self.get_Dgx(self.mu, u)
        # Get sensitivity to process noise
        Dgm = self.get_Dgm(self.mu, u)
        # Predict expected value
        self.mu = self.g(self.mu, u, zero_m)
        # Predict uncertainty
        self.cov = Dgx @ self.cov @ Dgx.T + Dgm @ Dgm.T

        self._check_cov()

    def update(self, z, diff=lambda x, y: x - y):
        zero_m = np.zeros_like(self.mu)
        zero_n = np.zeros_like(self.h(self.mu, zero_m))
        # Get sensitivity to uncertainty
        Dhx = self.get_Dhx(self.mu)
        # Get sensitivity to measurement noise
        Dhn = self.get_Dhn(self.mu)
        S = Dhx @ self.cov @ Dhx.T + Dhn @ Dhn.T
        K = self.cov @ Dhx.T @ np.linalg.inv(S)
        # Update expected value
        self.mu = self.mu + K @ diff(z, self.h(self.mu, zero_n))
        # Predict uncertainty
        self.cov = self.cov - K @ Dhx @ self.cov

        self._check_cov()

    def get_likelihood(self, z, diff=lambda x, y: x - y):
        zero_m = np.zeros_like(self.mu)
        zero_n = np.zeros_like(self.h(self.mu, zero_m))
        # Get sensitivity to uncertainty
        Dhx = self.get_Dhx(self.mu)
        # Get sensitivity to measurement noise
        Dhn = self.get_Dhn(self.mu)
        # Variance of expected measurements
        zhat_cov = Dhx @ self.cov @ Dhx.T
        # Expected measurement
        zhat_mu = self.h(self.mu, zero_n)

        total_cov = zhat_cov + Dhn @ Dhn.T

        dist = scipy.stats.multivariate_normal(mean=zero_n, cov=total_cov)

        p = dist.pdf(diff(z, zhat_mu))
        if p == 0:
            print("[WARNING] Likelihood is 0")
            return 0.00001
        return p

    def get_mu(self) -> np.ndarray:
        return self.mu

    def get_cov(self) -> np.ndarray:
        return self.cov

    def set_sensor_model(self, h, get_Dhx, get_Dhn):
        self.h = h
        self.get_Dhx = get_Dhx
        self.get_Dhn = get_Dhn

    def _check_cov(self):
        if(self.min_cov is not None):
            if(np.linalg.det(self.cov) < np.linalg.det(self.min_cov)):
                self.cov += self.min_cov
