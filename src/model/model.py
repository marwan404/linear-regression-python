import numpy as np

class LinearRegression():
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def calculateSums(self):
        x = self.x_values
        y = self.y_values

        self.sigma_x = np.sum(x)
        self.sigma_y = np.sum(y)
        self.sigma_xy = np.sum(x * y)

        self.sigma_xsqr = np.sum(x ** 2)
        self.sigma_ysqr = np.sum(y ** 2)

    def calculateSummaryStatistics(self):
        n = self.x_values.shape[0]

        self.Sxx = self.sigma_xsqr - (self.sigma_x ** 2)/n
        self.Syy = self.sigma_ysqr - (self.sigma_y ** 2)/n
        self.Sxy = self.sigma_xy - (self.sigma_x * self.sigma_y)/n

    def fit(self):
        n = self.x_values.shape[0]
        sigma_x = np.sum(self.x_values)
        sigma_y = np.sum(self.y_values)

        self.b = self.Sxy / self.Sxx
        self.a = (sigma_y / n) - self.b * (sigma_x / n)


    def calculateCorelation(self):

        r = self.Sxy/((self.Sxx * self.Syy) ** 0.5)
        direction = "positive" if r >=0 else "negative"

        self.r = abs(r)
        self.direction = direction

    def learn(self):
        self.calculateSums()
        self.calculateSummaryStatistics()
        self.calculateCorelation()
        self.fit()

    def predict(self, x):
        return self.a + self.b * x
