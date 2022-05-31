from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EKFSettings:
    """
    Settings for the EKF.
    """
    g: callable         # g
    Dgx: callable       # Jacobian of g with regard to x
    Dgm: callable       # Jacobian of g with regard to m
    h: callable         # h
    Dhx: callable       # Jacobian of h with regard to x
    Dhn: callable       # Jacobian of h with regard to n
    mu0: np.ndarray     # Initial expected value for x
    cov0: np.ndarray    # Initial covariance matrix for x


# print(f"[WARNING] EKF code does not behave as EKF.")


class EKF:
    """ EKF implementation. Assuming a system where
        x(k+1) = g(x(k), u(k+1), m(k+1))
        z(k) = h(x(k), n(k))
        x - state (unknown)
        u - input (known)
        m - multi-nomial process noise with identity covariance matrix, 0 mean
        n - multi-nomial observation noise with identity covariance matrix, 0 mean
        z - observations (known)

        g and h - non-linear, differentiable functions        
    """

    def __init__(self, settings: EKFSettings) -> None:
        self.g: callable = settings.g
        self.Dgx: callable = settings.Dgx
        self.Dgm: callable = settings.Dgm
        self.h: callable = settings.h
        self.Dhx: callable = settings.Dhx
        self.Dhn: callable = settings.Dhn
        self.mu: np.ndarray = settings.mu0
        self.cov: np.ndarray = settings.cov0

    def predict(self, u):
        zero = np.zeros_like(self.mu)
        # Get sensitivity to uncertainty
        Dgx = self.Dgx(self.mu, u,  zero)
        # Get sensitivity to noise
        Dgm = self.Dgm(self.mu, u,  zero)
        # Predict expected value
        self.mu = self.g(self.mu, u, zero)
        # Predict uncertainty
        self.cov = Dgx @ self.cov @ Dgx.T + Dgm @ Dgm.T

    def update(self, z):
        zero = np.zeros_like(self.mu)
        # Get sensitivity to uncertainty
        Dhx = self.Dhx(self.mu, zero)
        # Get sensitivity to noise
        Dhn = self.Dhn(self.mu, zero)
        S = Dhx @ self.cov @ Dhx.T + Dhn @ Dhn.T
        K = self.cov @ Dhx.T @ np.linalg.inv(S)
        # Update expected value
        self.mu = self.mu + K @ (z - self.h(self.mu, zero))
        # Predict uncertainty
        self.cov = self.cov - K @ Dhx @ self.cov

    def get_mu(self) -> np.ndarray:
        return self.mu

    def get_cov(self) -> np.ndarray:
        return self.cov
