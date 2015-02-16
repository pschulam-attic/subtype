from patsy import dmatrix

class Basis:
    def __init__(self, lower, upper, knots, degree):
        self.lower = lower
        self.upper = upper
        self.knots = knots
        self.degree = degree

    @property
    def df(self):
        df = self.degree * (len(self.knots) + 1)
        df -= (self.degree - 1) * len(self.knots)
        return df + 1  # Include intercept term.

    @property
    def formula(self):
        knots = ', '.join(str(k) for k in self.knots)
        f = 'bs(x, knots=[{}], degree={}, lower_bound={}, upper_bound={})'
        return f.format(knots, self.degree, self.lower, self.upper)

    def __call__(self, x):
        return dmatrix(self.formula)

    def __copy__(self):
        return Basis(self.lower, self.upper, self.knots[:], self.degree)
