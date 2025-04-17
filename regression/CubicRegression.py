import numpy as np
from common import HISTORY


class CubicRegression:
    def __init__(self, points):
        if len(points) != HISTORY:
            raise ValueError("Exactly 7 points are required.")

        self.x = np.array([p[0] for p in points])
        self.y = np.array([p[1] for p in points])

        self.coefficients = self._fit()

    def _fit(self):
        # Fit a cubic (degree 3) polynomial
        return np.polyfit(self.x, self.y, 3)

    def predict(self, x_value):
        y_value = np.polyval(self.coefficients, x_value)
        return y_value
