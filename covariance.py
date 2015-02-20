import numpy as np

from copy import copy


def row_vec(x):
    x = x.ravel()
    return np.atleast_2d(x)


def col_vec(x):
    return row_vec(x).T


class Covariance:
    def __copy__(self):
        param = self.__dict__
        new_cov = type(self)(**param)
        return new_cov


class DiagonalCovariance(Covariance):
    def __init__(self, variance):
        self.variance = variance

    def __call__(self, x1, x2=None):
        if x2 is None:
            return self.variance * np.eye(len(x1))
        else:
            return np.zeros((len(x1), len(x2)))


class RandomInterceptCovariance(Covariance):
    def __init__(self, variance):
        self.variance = variance

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        return self.variance * np.ones((len(x1), len(x2)))


class SquaredExpCovariance(Covariance):
    def __init__(self, amplitude, lengthscale):
        self.amplitude = amplitude
        self.lengthscale = lengthscale

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        r = col_vec(x1) - row_vec(x2)
        k = -0.5 * (r / self.lengthscale) ** 2
        return self.amplitude * np.exp(k)


class Matern32Covariance(Covariance):
    def __init__(self, lengthscale):
        self.lengthscale = lengthscale

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        r = np.abs(col_vec(x1) - row_vec(x2))
        l = self.lengthscale
        return ((1 + np.sqrt(3) * r / l) *
                np.exp(- np.sqrt(3) * r / l))


class Matern52Covariance(Covariance):
    def __init__(self, lengthscale):
        self.lengthscale = lengthscale

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        r = np.abs(col_vec(x1) - row_vec(x2))
        l = self.lengthscale
        return ((1 + np.sqrt(5) * r / l + 5 * r ** 2 / 3 / (l ** 2)) *
                np.exp(- np.sqrt(5) * r / l))


class CompositeCovariance(Covariance):
    def __init__(self, *components):
        self.components = components

    def __call__(self, x1, x2=None):
        all_C = [cov_fn(x1, x2) for cov_fn in self.components]
        return sum(all_C)

    def __copy__(self):
        copied_components = [copy(c) for c in self.components]
        new_comp = CompositeCovariance(*copied_components)
        return new_comp
