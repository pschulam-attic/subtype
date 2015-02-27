import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from collections import namedtuple, OrderedDict
from copy import copy
from patsy import dmatrix
from scipy.linalg import solve
from scipy.misc import logsumexp

from .logistic_regression import LogisticRegression
from .util import ConditionalPredictor, Trajectory


class SubtypeMixture:
    def __init__(self, nsubtypes, ncovariates, basis_fn, cov_fn, reg=None):
        self.nsubtypes = nsubtypes
        self.ncovariates = ncovariates
        self.basis = basis_fn
        self.cov = cov_fn
        self.reg = reg
        self._init_params()

    def __copy__(self):
        return SubtypeMixture(
            self.nsubtypes, self.ncovariates, copy(self.basis), copy(self.cov))

    def _init_params(self):
        self.subtype_marginal = LogisticRegression(
            self.nsubtypes, self.ncovariates)
        self.coef = np.zeros((self.nsubtypes, self.basis.df))

    def loglik(self, trajectory, z, X=None, C=None):
        X = self.basis(trajectory.t) if X is None else X
        C = self.cov(trajectory.t) if C is None else C
        m = X.dot(self.coef[z])
        y = trajectory.y.ravel()
        covariates = np.atleast_2d(trajectory.covariates)
        qz = self.subtype_marginal.predict_prob(covariates).ravel()

        lpz = np.log(qz[z])
        lpy = stats.multivariate_normal.logpdf(y, m, C)

        return lpz + lpy

    def obs_loglik(self, trajectory, X=None, C=None):
        X = self.basis(trajectory.t) if X is None else X
        C = self.cov(trajectory.t) if C is None else C
        ll = [self.loglik(trajectory, z, X, C) for z in range(self.nsubtypes)]
        return logsumexp(ll)

    def posterior(self, trajectory):
        all_loglik = [self.loglik(trajectory, z) for z in range(self.nsubtypes)]
        marg_loglik = logsumexp(all_loglik)
        qz = np.exp(all_loglik - marg_loglik)
        return qz

    def predict(self, t_new, trajectory):
        X_new = self.basis(t_new)
        X_obs = self.basis(trajectory.t)
        y_obs = trajectory.y
        C1 = self.cov(t_new, trajectory.t)
        C2 = self.cov(trajectory.t)
        qz = self.posterior(trajectory)
        y_new = np.zeros_like(t_new)

        for z, w in enumerate(qz):
            m1 = X_new.dot(self.coef[z]).ravel()
            m2 = X_obs.dot(self.coef[z]).ravel()
            y_new += w * (m1 + C1.dot(solve(C2, y_obs - m2)).ravel())

        return y_new

    def conditional(self, trajectory):
        return ConditionalPredictor(self, trajectory)

    TrajectoryData = namedtuple(
        'TrajectoryData', ['X', 'C', 'cov_xx', 'cov_xy'])

    def fit(self, trajectories, devset=None, max_iter=100, tol=1e-5):

        ## Cache computations needed for estimation and store
        ## covariates of all trajectories in a single feature matrix.

        cache = OrderedDict()
        covariates = []

        for trj in trajectories:
            X = self.basis(trj.t)
            C = self.cov(trj.t)

            cov_xx = X.T.dot(solve(C, X))
            cov_xy = X.T.dot(solve(C, trj.y)).ravel()

            cache[trj.key] = self.TrajectoryData(X, C, cov_xx, cov_xy)
            
            covariates.append(trj.covariates)

        covariates = np.vstack(covariates)
        qz = stats.dirichlet.rvs(self.nsubtypes * [0.1], len(trajectories))

        dev_cache = OrderedDict()

        for trj in devset:
            X = self.basis(trj.t)
            C = self.cov(trj.t)
            dev_cache[trj.key] = self.TrajectoryData(X, C, None, None)

        ## EM iterations start here.

        iteration = 0
        logl = np.zeros(max_iter + 1)
        logl[iteration] = -float('inf')

        while True:
            iteration += 1

            ## M-step (1): Estimate subtype probabilities.

            self.subtype_marginal.fit(covariates, qz)

            ## M-step (2): Estimate subtype coefficients.

            cov_xx = np.zeros((self.nsubtypes, self.basis.df, self.basis.df))
            cov_xy = np.zeros((self.nsubtypes, self.basis.df))

            if self.reg is not None:
                cov_xx += self.reg * self.basis.penalty

            for i, trj in enumerate(trajectories):
                for z, w in enumerate(qz[i]):
                    cov_xx[z] += w * cache[trj.key].cov_xx
                    cov_xy[z] += w * cache[trj.key].cov_xy

            for z in range(self.nsubtypes):
                self.coef[z] = solve(cov_xx[z], cov_xy[z])

            ## E-step: Recompute log-likelihood and posteriors

            for i, trj in enumerate(trajectories):
                X = cache[trj.key].X
                C = cache[trj.key].C
                stypes = range(self.nsubtypes)
                all_loglik = [self.loglik(trj, z, X, C) for z in stypes]
                marg_loglik = logsumexp(all_loglik)
                qz[i] = np.exp(all_loglik - marg_loglik)
                logl[iteration] += marg_loglik

            delta = logl[iteration] - logl[iteration - 1]
            abs_delta = delta / abs(logl[iteration])

            if devset is not None:
                dev_ll = 0.0
                
                for trj in devset:
                    X = dev_cache[trj.key].X
                    C = dev_cache[trj.key].C
                    dev_ll += self.obs_loglik(trj, X, C)

            else:
                dev_ll = -float('inf')

            msg = 'Iter={:03d}, LL={:.02f}, DevLL={:.02f}, Convergence={:.06f}'
            msg = msg.format(iteration, logl[iteration], dev_ll, abs_delta)
            logging.info(msg)

            if iteration >= max_iter or abs_delta < tol:
                break

        return self

    def plot(self):
        fig, ax = plt.subplots()
        x = np.linspace(self.basis.lower, self.basis.upper, 100)
        X = self.basis(x)
        y = X.dot(self.coef.T)
        ax.plot(x, y)
        return fig

    def plot_trajectories(self, trajectories=None):
        ncol = int(np.ceil(np.sqrt(4.0 / 3.0 * self.nsubtypes)))
        nrow = int(np.ceil(self.nsubtypes / float(ncol)))
        fig, axes = plt.subplots(nrow, ncol)
        x = np.linspace(self.basis.lower, self.basis.upper, 100)
        X = self.basis(x)
        y = X.dot(self.coef.T)

        if trajectories is not None:
            classified = [[] for _ in range(self.nsubtypes)]
            
            for trj in trajectories:
                qz = self.posterior(trj)
                z = np.argmax(qz)
                classified[z].append(trj)

        for i in range(self.nsubtypes):
            ax = axes[np.unravel_index(i, (nrow, ncol))]
            ax.plot(x, y[:, i])

            if trajectories is not None:
                for trj in classified[i]:
                    ax.plot(trj.t, trj.y, '-k')

            ax.set_ylim(0, 120)

        return fig

