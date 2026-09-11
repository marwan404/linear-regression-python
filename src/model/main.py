import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import LinearRegression
# from generate_random_csv import generate_csv

# generate_csv()

df = pd.read_csv("data.csv")

# this is not needed for all datasets, this is just for the one i got from kaggle
# so depending on the csv file itself you might have to write some code to handel the edge cases

x_values = df["mpg"].to_numpy()
y_values = df["acceleration"].to_numpy()

regression = LinearRegression(x_values, y_values)
regression.learn()

x_sorted = np.sort(x_values)
new_y = regression.predict(x_sorted)

plt.figure("mpg vs acc")
plt.scatter(x_values, y_values)
plt.plot(x_sorted, new_y, color='red')
# change the title and labels as you wish
plt.title("mpg vs acceleration")
plt.xlabel("mpg")
plt.ylabel("acceleration")
plt.show()

x = np.float64(input("enter mpg: "))

print(f"estimated acceleration: {round(regression.predict(x), 2)}")

print(f"correlation = {round(regression.r * 100, 2)}% in the {regression.direction} direction")