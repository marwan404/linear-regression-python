import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import LinearRegression
# from generate_random_csv import generate_csv

# generate_csv()

df = pd.read_csv("data.csv")

# this is not needed for all datasets, this is just for the one i got from kaggle
# so depending on the csv file itself you might have to write some code to handel the edge cases
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower"])

x_values = df["cylinders"].to_numpy()
y_values = df["horsepower"].to_numpy()

regression = LinearRegression(x_values, y_values)
a, b, r, direction = regression.learn()

x_sorted = np.sort(x_values)
new_y = a + b * x_sorted

plt.scatter(x_values, y_values)
plt.plot(x_sorted, new_y, color='red')
# change the title and labels as you wish
plt.title("cylinders vs hp")
plt.xlabel("cylinders")
plt.ylabel("hp")
plt.show()

x = np.float64(input("enter n.o cylenders: "))

print(f"estimated hp: {round(a + (b * x), 2)}")

print(f"correlation = {round(r * 100, 2)}% in the {direction} direction")