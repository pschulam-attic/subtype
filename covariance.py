import numpy as np

class DiagonalCovariance:
    def __init__(self, variance):
        self.variance = variance

    def __call__(self, x1, x2=None):
        if x2 is None:
            return self.variance * np.eye(len(x1))
        else:
            return np.zeros((len(x1), len(x2)))


class RandomInterceptCovariance:
    def __init__(self, variance):
        self.variance = variance

    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        return self.variance * np.ones((len(x1), len(x2)))


class CompositeCovariance:
    def __init__(self, *components):
        self.components = components

    def __call__(self, x1, x2=None):
        all_C = [cov_fn(x1, x2) for cov_fn in self.components]
        return sum(all_C)
