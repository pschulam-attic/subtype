import logging

import numpy as np
import scipy as sp
import scipy.stats as stats

from collections import namedtuple, OrderedDict
from scipy.linalg import inv, solve
from scipy.misc import logsumexp


class DirichletMultinomial:
    def __init__(self, K, alpha):
        self.K = K
        self.alpha = alpha
        self.counts = np.zeros(K)

    def increment(self, k):
        self.counts[k] += 1

    def decrement(self, k):
        self.counts[k] -= 1

    def sample(self):
        p = self.counts + self.alpha
        p /= p.sum()
        return np.random.choice(self.K, p=p)

    def predictive_logpdf(self, k):
        p = self.counts + self.alpha
        p /= p.sum()
        return np.log(p[k])


class BayesianRegression:
    def __init__(self, mean, cov):
        self.prior_mean = mean.copy()
        self.prior_cov = cov.copy()

        self.precision = inv(cov)
        self.unweighted_mean = solve(cov, mean)
        self.n = 0

    def increment(self, cov_xx, cov_xy):
        self.precision += cov_xx
        self.unweighted_mean += cov_xy
        self.n += 1

    def decrement(self, cov_xx, cov_xy):
        self.precision -= cov_xx
        self.unweighted_mean -= cov_xy
        self.n -= 1

    @property
    def mean(self):
        m = solve(self.precision, self.unweighted_mean).ravel()
        return m

    @property
    def cov(self):
        c = inv(self.precision)
        return c

    def predictive_mean(self, X):
        m = X.dot(self.mean)
        return m

    def predictive_cov(self, X, C):
        cov = C + X.dot(self.cov.dot(X.T))
        return cov

    def predictive_logpdf(self, X, y, C):
        post_m = self.predictive_mean(X)
        post_C = C + X.dot(self.cov).dot(X.T)
        ll = stats.multivariate_normal.logpdf(y, post_m, post_C)
        return ll


class MarginalizedSubtypeMixture:
    def __init__(self, nsubtypes, alpha, basis_fn, cov_fn,
                 prior_mean, prior_cov):

        self.nsubtypes = nsubtypes
        self.subtype_marginal = DirichletMultinomial(nsubtypes, alpha)

        self.basis = basis_fn
        self.cov = cov_fn

        new_br = lambda: BayesianRegression(prior_mean, prior_cov)
        self.subtype_likelihoods = [new_br() for _ in range(nsubtypes)]

    TrajectoryData = namedtuple(
        'TrajectoryData', ['X', 'C', 'cov_xx', 'cov_xy'])

    def fit(self, trajectories, nsamples=1000, nburn=25):

        ## Precompute and store sufficient statistics.

        cache = OrderedDict()

        for trj in trajectories:
            X = self.basis(trj.t)
            C = self.cov(trj.t)

            cov_xx = X.T.dot(solve(C, X))
            cov_xy = X.T.dot(solve(C, trj.y)).ravel()

            cache[trj.key] = self.TrajectoryData(X, C, cov_xx, cov_xy)

        ## Initialize subtype marginal and likelihoods.

        Z = np.zeros((nsamples + 1, len(trajectories)), dtype=np.int64)

        for i, trj in enumerate(trajectories):
            z_i = self.subtype_marginal.sample()
            self.subtype_marginal.increment(z_i)
            Z[0, i] = z_i

            cov_xx = cache[trj.key].cov_xx
            cov_xy = cache[trj.key].cov_xy
            self.subtype_likelihoods[z_i].increment(cov_xx, cov_xy)

        ## Sample from the posterior over assignments.

        for iter_ in range(1, nsamples + 1):
            logging.info('Starting iteration {}'.format(iter_))
            Z[iter_] = Z[iter_ - 1]
            
            for burn_ in range(nburn):
                logging.info('Burning {}:{}'.format(iter_, burn_))
                
                for i, trj in enumerate(trajectories):
                    z_i = Z[iter_, i]
                    cov_xx = cache[trj.key].cov_xx
                    cov_xy = cache[trj.key].cov_xy

                    self.subtype_marginal.decrement(z_i)
                    self.subtype_likelihoods[z_i].decrement(cov_xx, cov_xy)

                    X = cache[trj.key].X
                    y = trj.y
                    C = cache[trj.key].C
                    lp = np.zeros(self.nsubtypes)

                    for k in range(self.nsubtypes):
                        lp[k] += self.subtype_marginal.predictive_logpdf(k)
                        lp[k] += self.subtype_likelihoods[k].\
                                 predictive_logpdf(X, y, C)

                    lp -= logsumexp(lp)
                    z_i = np.random.choice(self.nsubtypes, p=np.exp(lp))

                    self.subtype_marginal.increment(z_i)
                    self.subtype_likelihoods[z_i].increment(cov_xx, cov_xy)
                    Z[iter_, i] = z_i

        self.Z = Z
        return self

