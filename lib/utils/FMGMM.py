import numpy as np

from sklearn.mixture import GaussianMixture

class FixedMeanGMM(GaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,
                 init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None,
                 fixed_mean=None, warm_start = False):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol,
                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                         weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start = warm_start)
        self.fixed_mean = fixed_mean

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self.means_[-1] = self.fixed_mean

    def _m_step(self, X, log_resp):
        super()._m_step(X, log_resp)
        self.means_[-1] = self.fixed_mean