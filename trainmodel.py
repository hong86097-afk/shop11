import pandas as pd
from sklearn.linear_model import LogisticRegression 
import joblib


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