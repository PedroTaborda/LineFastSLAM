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
    """ EKF implementation. Assumes the following mathematical model
        x(k+1) = g(x(k), u(k+1), m(k+1))
        z(k) = h(x(k), n(k))
        x - state (unknown)
        u - input (known)
        m - multi-normal process noise with identity covariance matrix,
                0 mean and same dimensions as x (unknown)
        n - multi-normal observation noise with identity covariance matrix,
                0 mean and same dimensions as z (unknown)
        z - observations (known)

        g and h - non-linear, differentiable functions  

        Callables to supply:
            g(x, u, parameters) -> x
            h(x, parameters)    -> z
            get_Dgx(x, u, parameters) -> Dgx (same style for Dgm)
            get_Dhx(x, parameters) -> Dhx (same style for Dhn)

            Dgm, for example, means jacobian of g with respect to m, at x


        Set sensor model before running update or getting likelihood        
    """

    def __init__(self, settings: EKFSettings) -> None:
        self.g: callable = settings.g
        self.get_Dgx: callable = settings.get_Dgx
        self.get_Dgm: callable = settings.get_Dgm
        self.mu: np.ndarray = settings.mu0
        self.cov: np.ndarray = settings.cov0
        self.old_mu = self.mu
        self.old_cov = self.cov
        self.min_cov = settings.min_cov
        self.parameters = []
        self.old_params = []
        
    def predict(self, u):
        # Get sensitivity to uncertainty
        Dgx = self.get_Dgx(self.mu, u)
        # Get sensitivity to process noise
        Dgm = self.get_Dgm(self.mu, u)
        # Predict expected value
        self.set_mu( self.g(self.mu, u) )
        # Predict uncertainty
        self.set_cov( Dgx @ self.cov @ Dgx.T + Dgm @ Dgm.T )
        self.update_zdist()


    def update(self, z, diff=lambda x, y: x - y):
        K = self.cov @ self.Dhx.T @ self.inv_z_cov
        # Update expected value
        self.set_mu( self.mu + K @ diff(z, self.h(self.mu, self.parameters)) )
        # Predict uncertainty
        self.set_cov( self.cov - K @ self.Dhx @ self.cov )       
        

    def get_likelihood(self, z, diff=lambda x, y: x - y, normalize=True):
        #dist = scipy.stats.multivariate_normal(mean=zero_n, cov=total_cov)
        #p = dist.pdf(diff(z, zhat_mu))

        # Replace with the pdf expression because the scipy implementation is too slow.
        p =np.exp(-1/2 * diff(z, self.zhat_mu).T @ self.inv_z_cov @ diff(z, self.zhat_mu))
        if normalize:
            p *= self.normalizing_factor
        if p == 0:
            # \033[<N>B moves cursor N lines down, in case cursor is not at end of console
            #print("\033[99B\r[WARNING] Likelihood is 0")
            return 1e-5
        return p

    def get_Mahalanobis_squared(self, z, diff=lambda x, y: x - y):
        return diff(z, self.zhat_mu).T @ self.inv_z_cov @ diff(z, self.zhat_mu)


    def get_mu(self) -> np.ndarray:
        return self.mu

    def set_mu(self, mu):
        self.old_mu = self.mu
        self.mu = mu

    def get_cov(self) -> np.ndarray:
        return self.cov

    def set_cov(self, cov):
        self.old_cov = cov
        self.cov = cov

    def set_sensor_model(self, h, get_Dhx, get_Dhn):
        self.h = h
        self.get_Dhx = get_Dhx
        self.get_Dhn = get_Dhn
        self.parameters = []        
        self.old_params = self.parameters
    
    def set_parameters(self, parameters):
        self.old_params = self.parameters
        self.parameters = parameters
        self.update_zdist()

    def update_zdist(self):
        changed_mu = (self.old_mu != self.mu).all()
        changed_cov = (self.old_cov != self.cov).all()
        changed_params = (not self.old_params and self.parameters) or not all([(self.parameters[i]==self.parameters[i]).all() for i in range(len(self.parameters))])
        if changed_mu or changed_params:
            self.Dhx = self.get_Dhx(self.mu, self.parameters)
            self.Dhn = self.get_Dhn(self.mu, self.parameters)
        if changed_cov:
            self._check_cov()            
            changed_cov = self.old_cov != self.cov
        if changed_mu or changed_params or changed_cov:
            # Variance of expected measurements without noise
            self.zhat_cov = self.Dhx @ self.cov @ self.Dhx.T
            # Expected measurement
            self.zhat_mu = self.h(self.mu, self.parameters)

            self.z_cov = self.zhat_cov + self.Dhn @ self.Dhn.T
            self.inv_z_cov = np.linalg.inv(self.z_cov)
            self.normalizing_factor = np.linalg.det(2 * np.pi * self.z_cov)**(-1/2)

    def _check_cov(self):
        if(self.min_cov is not None):
            if(np.linalg.det(self.cov) < np.linalg.det(self.min_cov)):
                self.cov += self.min_cov

    