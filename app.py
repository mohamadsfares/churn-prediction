import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")
st.markdown(
    "<div style='background:linear-gradient(90deg,#1f3a5f,#3b6ea5);padding:14px 20px;border-radius:10px;color:white;font-size:15px'>"
    "📉 <b>Customer Churn Radar</b> &nbsp;|&nbsp; spot customers who are about to stop buying, before they do"
    "</div>",
    unsafe_allow_html=True,
)
bundle = joblib.load("models/churn_model.joblib")
model = bundle["model"]
options = {"Shorter list (fewer, more certain)": 0.5, "Balanced (recommended)": 0.3, "Catch as many as possible": 0.2}
choice = st.sidebar.select_slider("How many customers should we flag?", options=list(options), value="Balanced (recommended)", help="A longer list catches more customers who are about to leave, but also includes more who would have stayed.")
threshold = options[choice]
features = bundle["features"]
data = pd.read_csv("data/model_data.csv", index_col=0)
data["risk"] = model.predict_proba(data[features])[:, 1]
data["high_risk"] = data["risk"] > threshold
typical = data[features].median()
new = typical.copy()
st.title("Which customers are about to leave?")
st.write("Estimates which regular customers (5+ orders) may stop buying in the next 90 days.")
k1, k2, k3 = st.columns(3)
k1.metric("Customers scored", len(data))
k2.metric("Flagged high risk", int(data["high_risk"].sum()))
k3.metric("Share flagged", f"{data['high_risk'].mean():.0%}")
st.sidebar.subheader("Pick a customer")
customer_id = st.sidebar.selectbox("Customer ID (click to choose)", data.index)
row= data.loc[customer_id]
st.sidebar.divider()
st.sidebar.subheader("About this model")
st.sidebar.metric("Real churners it catches", "3 out of 4")
st.sidebar.metric("Flagged customers who really leave", "about 1 in 2")
with st.sidebar.expander("Technical details"):
    st.write("Random forest. ROC-AUC 0.75, recall 0.75, precision 0.47 on held-out customers at the Balanced setting. Risk is the estimated chance of buying nothing in the next 90 days.")
bar_color = "#e5484d" if row["high_risk"] else "#30a46c"

with st.container(border=True):
    st.subheader(f"Customer {customer_id}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn risk", f"{row['risk']:.0%}")
        st.markdown(
            f"<div style='background:#eee;border-radius:8px;height:12px'>"
            f"<div style='width:{row['risk']:.0%};background:{bar_color};height:12px;border-radius:8px'></div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        if row["high_risk"]:
            st.error("High risk: consider a retention offer.")
        else:
            st.success("Low risk: no action needed now.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Days since last order", int(row["recency"]))
    m2.metric("Usual days between orders", f"{row['usual_gap']:.0f}")
    m3.metric("Spend in last 90 days", f"{row['recent_spend']:,.0f}")


st.subheader("Top 10 customers to contact first")
top = data.sort_values("risk", ascending=False).head(10)
st.dataframe(
    top[["risk", "recency", "usual_gap", "recent_spend"]],
    column_config={
        "risk": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="percent"),
        "recency": st.column_config.NumberColumn("Days since last order"),
        "usual_gap": st.column_config.NumberColumn("Usual days between orders", format="%.0f"),
        "recent_spend": st.column_config.NumberColumn("Spend in last 90 days", format="%.0f"),
    },
)
flagged = data[data["high_risk"]].sort_values("risk", ascending=False)                       # NEW
st.download_button("Download flagged customers (CSV)",                                      # NEW
                   flagged[["risk", "recency", "usual_gap", "recent_spend"]].to_csv(),
                   "flagged_customers.csv", "text/csv")
st.caption("Demo on historical data. The model reached ROC-AUC 0.75 on held-out customers; risk scores here are for customers it was trained on, so they look slightly better than for new customers.")
st.subheader("Score a new customer")
new_recency = st.number_input("Days since last order", min_value=0, max_value=800, value=int(typical["recency"]))
new_frequency = st.number_input("Number of orders", min_value=5, max_value=100, value=int(typical["frequency"]))
new_variety = st.number_input("Different products bought", min_value=1, max_value=500, value=int(typical["variety"]))
new_recent_spend = st.number_input("Spend in the last 90 days", min_value=0.0, max_value=5000.0, value=float(typical["recent_spend"]))
new = typical.copy()
new["recency"] = new_recency
new["frequency"] = new_frequency
new["variety"] = new_variety
new["recent_spend"] = new_recent_spend
new["recent_orders"] = 0 if new_recent_spend == 0 else typical["recent_orders"]
new["recent_share"] = 0 if new_recent_spend == 0 else typical["recent_share"]
new_row = pd.DataFrame([new], columns=features)
new_risk = model.predict_proba(new_row)[:, 1][0]
st.metric("Risk for this customer", f"{new_risk:.0%}")
st.caption("Other values use the typical customer, so unusual combinations give rough estimates.")