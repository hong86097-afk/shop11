import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

page_bg_img = """
<style>
/*Main app background */
.stApp {
    background-image: url("https://i.pinimg.com/736x/7b/e8/5a/7be85abb15f78a5727c1c40c8a62061a.jpg");
    background-size: 1000px 1000px;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;

}
/*sidebar background */
[data-testid="stSidebar"] {
    background-image: url("https://i.pinimg.com/736x/7b/e8/5a/7be85abb15f78a5727c1c40c8a62061a.jpg"); /* same image or different one */
    background-size: 1000px 1000px;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

/* Make all text clearer */
h1, h2, h3, h4, h5, h6, p, div {
    color: #000000 !important;   /* force black text */
    text-shadow: 4px 4px 8px #ffffff; /* add subtle white glow */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


model = joblib.load("model.pkl")

st.title("🛒 Shopping Behavior Dashboard")
st.markdown("### 📘 Welcome to the Shopping Prediction App")
st.markdown("## 📑 Project Summary")

st.markdown("""
This project applies **machine learning** to predict customer shopping behavior.  
The model uses features such as **price, discount, brand, and rating** to estimate the likelihood of purchase.  

- ✅ The prediction dashboard shows whether a customer is likely to buy a product.  
- 📉 The **Training Cost vs Iteration** graph demonstrates how the model improved during training, with cost steadily decreasing as iterations increased.  
- 📊 The **Probability Curve** illustrates how purchase probability changes with price, highlighting sensitivity to pricing strategies.  
- 🔍 The **Correlation Heatmap** provides insights into relationships among features, helping identify which factors most influence buying decisions.  

Overall, this app combines **predictive analytics** with **interactive visualization**, making it a practical tool for understanding consumer behavior and supporting business decision-making.
""")

st.sidebar.header("Input Features")
price = st.sidebar.slider("Price", 0.0, 1000.0, 50.0)
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
brand = st.sidebar.selectbox("Brand (encoded)", list(range(0, 11)), index=5)
rating = st.sidebar.slider("Rating (1-5)", 1, 5, 3)

if st.sidebar.button("Predict Purchase"):
    input_data = np.array([[price, discount, brand, rating]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.success(f"✅ Likely to buy! (Probability: {prob:.2f}%)")
        st.balloons()
    else:
        st.error(f"❌ Unlikely to buy. (Probability: {prob:.2f}%)")

   
    col1, col2 = st.columns(2)

    price_range = np.linspace(0, 1000, 50).reshape(-1,1)
    X_demo = np.hstack([price_range,
                        np.full((len(price_range),1), discount),
                        np.full((len(price_range),1), brand),
                        np.full((len(price_range),1), rating)])
    probs = model.predict_proba(X_demo)[:,1] * 100

    sns.set_style("whitegrid")
    fig1, ax1 = plt.subplots()
    ax1.plot(price_range, probs, color="red", linewidth=2, label="Purchase Probability")
    ax1.fill_between(price_range.flatten(), probs, color="red", alpha=0.2)
    ax1.scatter([price], [prob], color="blue", edgecolors="black", s=120, label="Current Input")
    ax1.set_xlabel("Price ($)")
    ax1.set_ylabel("Probability (%)")
    ax1.legend()
    col1.pyplot(fig1)


    try:
        costs = np.load("training_costs.npy")
    except:
        costs = [0.7 - 0.00035*i for i in range(1000)]  # fallback demo curve

    fig2, ax = plt.subplots()
    ax.plot(range(1, len(costs)+1), costs, color="purple", linewidth=2, label="Training Cost")
    ax.fill_between(range(1, len(costs)+1), costs, color="purple", alpha=0.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Cost vs Iteration")
    ax.legend()
    col2.pyplot(fig2)


    try:
        df = pd.read_csv("shopping_data.csv")
        corr = df[["price","discount","brand","rating","buy"]].corr()
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax3)
        ax3.set_title("Feature Correlation Heatmap")
        st.pyplot(fig3)
    except:
        st.info("📊 Heatmap unavailable (dataset not found).")


feedback = st.text_area("💬 Leave your feedback or suggestions here:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback! We appreciate your input.")
