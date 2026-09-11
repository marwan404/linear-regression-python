import numpy as np


class NotFittedError(Exception):
    """Raised when predict() before learn() is called."""


class LinearRegression:
    def __init__(self, x_values, y_values):
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)

        if x_values.shape != y_values.shape:
            raise ValueError(
                f"x_values and y_values must be the same shape, "
                f"got {x_values.shape} and {y_values.shape}"
            )
        if x_values.ndim != 1:
            raise ValueError("x_values and y_values must be 1D arrays")
        if x_values.shape[0] < 2:
            raise ValueError("need at least 2 data points to fit a line")

        self.x_values = x_values
        self.y_values = y_values
        self.n = x_values.shape[0]

        self.a = None          # intercept
        self.b = None          # slope
        self.r = None          # correlation coefficient (absolute value)
        self.direction = None  # "positive" or "negative"
        self._fitted = False

    def _calculate_sums(self):
        x, y = self.x_values, self.y_values
        self.sigma_x = np.sum(x)
        self.sigma_y = np.sum(y)
        self.sigma_xy = np.sum(x * y)
        self.sigma_xsqr = np.sum(x ** 2)
        self.sigma_ysqr = np.sum(y ** 2)

    def _calculate_summary_statistics(self):
        n = self.n
        self.Sxx = self.sigma_xsqr - (self.sigma_x ** 2) / n
        self.Syy = self.sigma_ysqr - (self.sigma_y ** 2) / n
        self.Sxy = self.sigma_xy - (self.sigma_x * self.sigma_y) / n

        if self.Sxx == 0:
            raise ValueError(
                "all x_values are identical; cannot fit a line (undefined slope)"
            )

    def _fit(self):
        n = self.n
        self.b = self.Sxy / self.Sxx
        self.a = (self.sigma_y / n) - self.b * (self.sigma_x / n)

    def _calculate_correlation(self):
        if self.Sxx * self.Syy == 0:
            self.r = 0.0
            self.direction = "undefined"
            return
        r = self.Sxy / ((self.Sxx * self.Syy) ** 0.5)
        self.direction = "positive" if r >= 0 else "negative"
        self.r = abs(r)

    def learn(self):
        self._calculate_sums()
        self._calculate_summary_statistics()
        self._calculate_correlation()
        self._fit()
        self._fitted = True
        return self

    def predict(self, x):
        if not self._fitted:
            raise NotFittedError("call learn() before predict()")
        x = np.asarray(x, dtype=np.float64)
        return self.a + self.b * x
    