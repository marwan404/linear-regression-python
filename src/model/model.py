import numpy as np

class LinearRegression():
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def calculateSums(self):
        x = self.x_values
        y = self.y_values

        sigma_x = np.sum(x)
        sigma_y = np.sum(y)
        sigma_xy = np.sum(x * y)

        sigma_xsqr = np.sum(x ** 2)
        sigma_ysqr = np.sum(y ** 2)

        return sigma_x, sigma_y, sigma_xy, sigma_xsqr, sigma_ysqr

    def calculateSummaryStatistics(self, sigma_x, sigma_y, sigma_xy, sigma_xsqr, sigma_ysqr):
        n = self.x_values.shape[0]

        Sxx = sigma_xsqr - (sigma_x ** 2)/n
        Syy = sigma_ysqr - (sigma_y ** 2)/n
        Sxy = sigma_xy - (sigma_x * sigma_y)/n

        return Sxx, Syy, Sxy

    def fit(self, Sxx, Sxy):
        n = self.x_values.shape[0]
        sigma_x = np.sum(self.x_values)
        sigma_y = np.sum(self.y_values)

        b = Sxy / Sxx
        a = (sigma_y / n) - b * (sigma_x / n)

        return a, b

    def calculateCorelation(self, Sxx, Syy,Sxy):

        r = Sxy/((Sxx * Syy) ** 0.5)
        direction = "positive" if r >=0 else "negative"

        return abs(r), direction

    def learn(self):
        sigma_x, sigma_y, sigma_xy, sigma_xsqr, sigma_ysqr = self.calculateSums()
        Sxx, Syy, Sxy = self.calculateSummaryStatistics(sigma_x, sigma_y, sigma_xy, sigma_xsqr, sigma_ysqr)
        r, direction = self.calculateCorellation(Sxx, Syy,Sxy)
        a, b = self.fit(Sxx, Sxy)

        return a, b, r, direction
