import numpy as np

from scipy.optimize import fmin_bfgs

class LogisticRegression:
    def __init__(self, noutcomes, npredictors):
        self.noutcomes = noutcomes
        self.npredictors = npredictors
        self._init_params()

    def _init_params(self):
        k, p = self.noutcomes, self.npredictors
        self.weights = np.random.normal(size=(k - 1, p))

    def fit(self, X, y):
        p = self.npredictors
        f = lambda w: -logistic_loglik(w.reshape((-1, p)), X, y)
        g = lambda w: -logistic_loglik_grad(w.reshape((-1, p)), X, y).ravel()

        solution = fmin_bfgs(f, self.weights.ravel().copy(), g, disp=False)
        self.weights = solution.reshape((-1, p))

        return self

    def predict_prob(self, X):
        prob = np.zeros((X.shape[0], self.noutcomes))

        for i, x_i in enumerate(X):
            prob[i] = logistic_predict_prob(self.weights, x_i)

        return prob
        
    def predict_outcome(self, X):
        prob = self.predict_prob(X)
        yhat = np.argmax(prob, axis=1)
        return yhat


def onehot_encode(x, k):
    n = x.size
    X = np.zeros((n, k))

    for i, j in enumerate(x):
        X[i, j] = 1

    return X


def logistic_predict_prob(W, x):
    k = W.shape[0] + 1

    p = np.ones(k)
    p[:-1] = np.exp(W.dot(x.ravel()))
    p /= p.sum()

    return p


def logistic_loglik(W, X, y):
    k = W.shape[0] + 1
    n, p = X.shape

    if y.ndim == 1:
        y = onehot_encode(y, k)

    ll = 0.0

    for i, x_i in enumerate(X):
        p_i = logistic_predict_prob(W, x_i)
        ll += y[i].dot(np.log(p_i))

    return ll


def logistic_loglik_grad(W, X, y):
    k = W.shape[0] + 1
    n, p = X.shape

    if y.ndim == 1:
        y = onehot_encode(y, k)

    g = np.zeros_like(W)

    for i, x_i in enumerate(X):
        p_i = logistic_predict_prob(W, x_i)

        for j, y_ij in enumerate(y[i]):
            if j >= k - 1:
                continue

            g[j] += (y_ij - p_i[j]) * x_i

    return g
