import numpy as np
import pandas as pd


def generate_csv():
    n = 100

    player_count = np.random.uniform(100, 0.5, n)
    noise = np.random.normal(0, 8, n)

    price = 0.5 * player_count + 20 + noise
    price = np.clip(price, 0, 70)

    df = pd.DataFrame({
        "price (USD)": price,
        "player count (millions)": player_count
    })

    df.to_csv("data.csv", index=False)
    print("done, saved to data.csv")