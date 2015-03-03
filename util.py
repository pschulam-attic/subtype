import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

from collections import namedtuple
from numpy.linalg import cholesky, inv, slogdet, solve
from patsy import dmatrix


Trajectory = namedtuple('Trajectory', ['key', 't', 'y', 'covariates'])


def make_trajectories(key, time, marker, dataframe):
    trajectories = []

    for k, data in dataframe.groupby(key):
        t = data[time].values.ravel()
        y = data[marker].values.ravel()
        covariates = np.ones(1)

        idx = np.argsort(t)
        trj = Trajectory(k, t[idx], y[idx], covariates)
        trajectories.append(trj)

    return trajectories


def add_covariates(trajectories, key, formula, dataframe):
    cov_matrix = dmatrix(formula, dataframe)
    row_keys = dataframe[key]
    covariates = {k:v for k, v in zip(row_keys, cov_matrix)}

    with_covariates = []

    for trj in trajectories:
        trj_copy = Trajectory(
            trj.key, trj.t, trj.y, covariates[trj.key])
        with_covariates.append(trj_copy)

    return with_covariates


def truncate_trajectory(trajectory, num_obs=None, censoring_time=None):
    if num_obs is not None:
        t = trajectory.t[:num_obs]
        y = trajectory.y[:num_obs]

    elif censoring_time is not None:
        keep = trajectory.t < censoring_time
        t = trajectory.t[keep]
        y = trajectory.y[keep]

    else:
        raise RuntimeError('You must specify num_obs or censoring_time.')

    new_trj = Trajectory(trajectory.key, t, y, trajectory.covariates)

    return new_trj


def predictive_contexts(trajectory):
    n = len(trajectory.t)

    for num_obs in range(1, n):
        obs_trj = truncate_trajectory(trajectory, num_obs)
        t_new = trajectory.t[num_obs:]
        y_new = trajectory.y[num_obs:]
        
        yield obs_trj, t_new, y_new

        
class ConditionalPredictor:
    def __init__(self, model, trajectory):
        self.model = model
        self.trajectory = trajectory

    def predict(self, t_new):
        y_new = self.model.predict(t_new, self.trajectory)
        return y_new

    def confidence_bands(self, t_new, nsim=1000):
        samples = np.zeros((nsim, len(t_new)))
        qz = self.model.posterior(self.trajectory)

        C11 = self.model.cov(t_new)
        C12 = self.model.cov(t_new, self.trajectory.t)
        C22 = self.model.cov(self.trajectory.t)
        C = C11 - C12.dot(solve(C22, C12.T))

        s = stats.multivariate_normal.rvs(cov=C, size=nsim)
        l, u = np.percentile(s, [2.5, 97.5], axis=0)
        
        return l, u

    def plot(self, ax=None, *args, **kwargs):
        lower = self.model.basis.lower
        upper = self.model.basis.upper
        t_grid = np.linspace(lower, upper, 100)
        y_grid = self.predict(t_grid)
        l_grid, u_grid = y_grid + self.confidence_bands(t_grid)

        if ax is None:
            fig, ax = plt.subplots(*args, **kwargs)

        else:
            fig = ax.figure
            
        ax.plot(self.trajectory.t, self.trajectory.y, 'xb', label='Observed')
        ax.plot(t_grid, y_grid, '-r', label='Predicted')
        ax.plot(t_grid, l_grid, '-.r', label='95% Lower')
        ax.plot(t_grid, u_grid, '-.r', label='95% Upper')

        return fig, ax


_LOG_2PI = np.log(2 * np.pi)

def mvnlogpdf(x, mean, cov):
    rank = len(x)
    prec_U = cholesky(inv(cov))
    log_det_cov = slogdet(cov)[1]
    dev = x - mean
    maha = np.sum(np.square(np.dot(dev, prec_U)))
    return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)


def capture_bw(y_obs, y_hat, frac):
    n = len(y_obs)
    d = np.sort(np.abs(y_obs - y_hat))
    p = (np.arange(n, dtype=np.float_) + 1) / n
    return d[np.searchsorted(p, frac)]


def confusion_matrix(model, trajectories, censoring_time):
    true_subtype = np.zeros(len(trajectories))
    pred_subtype = np.zeros(len(trajectories))
    cmat = np.zeros((model.nsubtypes, model.nsubtypes))    

    for i, trj in enumerate(trajectories):
        trunc = truncate_trajectory(trj, censoring_time=censoring_time)
        if len(trunc.t) < 1:
            continue
        true_z = np.argmax(model.posterior(trj))
        pred_z = np.argmax(model.posterior(trunc))
        cmat[true_z, pred_z] += 1

    return cmat / np.atleast_2d(cmat.sum(axis=1)).T


def make_all_predictions(model, trajectories):
    all_predictions = {}
    for trj in trajectories:
        for obs, tnew, ynew in predictive_contexts(trj):
            yhat = model.conditional(obs).predict(tnew)
            nobs = len(obs.t)
            all_predictions[(trj.key, nobs)] = (obs, tnew, ynew, yhat)

    return all_predictions


def prediction_summary(all_predictions):
    def bucket_coverage(x):
        m = max(np.ceil(np.max(x)), 1)
        b = np.linspace(0, m, 2 * m + 1)
        h, _ = np.histogram(x, bins=b)
        return np.mean(h > 0)

    summary = []
    
    for obs, tnew, ynew, yhat in all_predictions.values():
        frontier = max(1, np.ceil(obs.t[-1]))
        coverage = np.round(bucket_coverage(obs.t), 1)
        year_of_care = np.ceil(tnew)

        for yoc, yn, yh in zip(year_of_care, ynew, yhat):
            summary.append((frontier, yoc, coverage, yn, yh, yn-yh, np.abs(yn-yh)))

    s = pd.DataFrame(list(zip(*summary))).T
    s.columns = ['frontier', 'year_of_care', 'coverage', 'ynew', 'yhat', 'residual', 'error']
    return s

