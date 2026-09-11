import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import LinearRegression

df = pd.read_csv("data.csv")
x_values = df["mpg"].to_numpy()
y_values = df["acceleration"].to_numpy()

regression = LinearRegression(x_values, y_values)
regression.learn()

x_sorted = np.sort(x_values)
new_y = regression.predict(x_sorted)

plt.figure("mpg vs acc")
plt.scatter(x_values, y_values, marker="x", c="#97ED69EC")
plt.plot(x_sorted, new_y, color='red')
plt.title("mpg vs acceleration")
plt.xlabel("mpg")
plt.ylabel("acceleration")
plt.show()

x = np.float64(input("enter mpg: "))

print(f"estimated acceleration: {round(regression.predict(x), 2)}")
print(f"correlation = {round(regression.r * 100, 2)}% in the {regression.direction} direction")
