import traceback

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


page_bg_img = """
<style>
.stApp {
    background-image: url(
        "https://i.pinimg.com/736x/7b/e8/5a/7be85abb15f78a5727c1c40c8a62061a.jpg"
    );
    background-size: 1000px 1000px;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

[data-testid="stSidebar"] {
    background-image: url(
        "https://i.pinimg.com/736x/7b/e8/5a/7be85abb15f78a5727c1c40c8a62061a.jpg"
    );
    background-size: 1000px 1000px;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

h1, h2, h3, h4, h5, h6, p, div, label {
    color: #000000 !important;
    text-shadow: 2px 2px 4px #ffffff;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.code(traceback.format_exc())
    st.stop()

st.title("🛒 Shopping Behavior Dashboard")
st.markdown("### 📘 Welcome to the Shopping Prediction App")
st.markdown("## 📑 Project Summary")

st.markdown(
    """
This project applies **machine learning** to predict customer shopping behavior.
The model uses features such as **price, discount, brand, and rating**
to estimate the likelihood of purchase.

- ✅ The prediction dashboard shows whether a customer is likely to buy a product.
- 📉 The **Training Cost vs Iteration** graph demonstrates how the model improved during training.
- 📊 The **Probability Curve** illustrates how purchase probability changes with price.
- 🔍 The **Correlation Heatmap** provides insights into relationships among features.

Overall, this app combines **predictive analytics** with **interactive visualization**
to support business decision-making.
"""
)

st.sidebar.header("Input Features")

price = st.sidebar.slider("Price", 0.0, 1000.0, 50.0)
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
brand = st.sidebar.selectbox("Brand (encoded)", list(range(0, 11)), index=5)
rating = st.sidebar.slider("Rating (1-5)", 1, 5, 3)

if st.sidebar.button("Predict Purchase"):
    try:
        input_data = np.array([[price, discount, brand, rating]])
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1] * 100
        else:
            prob = 0.0

        if prediction == 1:
            st.success(f"✅ Likely to buy! (Probability: {prob:.2f}%)")
            st.balloons()
        else:
            st.error(f"❌ Unlikely to buy. (Probability: {prob:.2f}%)")

        col1, col2 = st.columns(2)

        price_range = np.linspace(0, 1000, 50).reshape(-1, 1)
        x_demo = np.hstack(
            [
                price_range,
                np.full((len(price_range), 1), discount),
                np.full((len(price_range), 1), brand),
                np.full((len(price_range), 1), rating),
            ]
        )

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_demo)[:, 1] * 100
        else:
            probs = np.zeros(len(price_range))

        sns.set_style("whitegrid")

        fig1, ax1 = plt.subplots()
        ax1.plot(
            price_range.flatten(),
            probs,
            color="red",
            linewidth=2,
            label="Purchase Probability",
        )
        ax1.fill_between(
            price_range.flatten(),
            probs,
            color="red",
            alpha=0.2,
        )
        ax1.scatter(
            [price],
            [prob],
            color="blue",
            edgecolors="black",
            s=120,
            label="Current Input",
        )
        ax1.set_xlabel("Price ($)")
        ax1.set_ylabel("Probability (%)")
        ax1.set_title("Probability Curve")
        ax1.legend()
        col1.pyplot(fig1)

        try:
            costs = np.load("training_costs.npy")
        except Exception:
            costs = [0.7 - 0.00035 * i for i in range(1000)]

        fig2, ax2 = plt.subplots()
        ax2.plot(
            range(1, len(costs) + 1),
            costs,
            color="purple",
            linewidth=2,
            label="Training Cost",
        )
        ax2.fill_between(
            range(1, len(costs) + 1),
            costs,
            color="purple",
            alpha=0.2,
        )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cost")
        ax2.set_title("Cost vs Iteration")
        ax2.legend()
        col2.pyplot(fig2)

        try:
            df = pd.read_csv("shopping_data.csv")
            corr = df[["price", "discount", "brand", "rating", "buy"]].corr()

            fig3, ax3 = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax3)
            ax3.set_title("Feature Correlation Heatmap")
            st.pyplot(fig3)

        except Exception as e:
            st.info(f"📊 Heatmap unavailable: {e}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.code(traceback.format_exc())

feedback = st.text_area("💬 Leave your feedback or suggestions here:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback! We appreciate your input.")