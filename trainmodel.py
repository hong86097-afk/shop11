import pandas as pd
from sklearn.linear_model import LogisticRegression 
import joblib
import numpy as np

np.save("training_costs.npy", np.array([0.7 - 0.00035*i for i in range(1000)]))  # Simulated cost data
def train_model():
    df = pd.read_csv("shopping_data.csv")
    x =df[["price","discount","brand","rating"]]
    y = df["buy"]

    model = LogisticRegression()
    model.fit(x,y)

    joblib.dump(model, "model.pkl")
    print("Model trained and saved as model.pkl")

    # Plotting the results
   

if __name__ == "__main__":
    train_model()

