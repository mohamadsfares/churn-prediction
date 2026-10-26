# Churn Prediction: Which Customers Are About to Leave?

**Live demo:** https://churn-radar-mohamad.streamlit.app/

## Problem
A retailer loses regular customers silently. If we can spot who is likely to stop buying, the team can contact them before they leave.

## Aim
Predict which regular customers (5+ orders) will make **no purchase in the next 90 days**.

## Data
UCI Online Retail II (about 1M transactions, Dec 2009 to Dec 2011), cleaned in Project 1.

## Method
1. **Label without leakage:** cutoff date 2011-09-09. Features use only history before the cutoff. Churned = 1 if no purchase in the 90 days after it. 539 of 1,798 regular customers churned (30%).
2. **Features (10):** recency, frequency, monetary, avg_order, variety, usual_gap, recent_orders, recent_share, recent_spend, spend_share.
3. **Models:** dummy baseline, logistic regression, random forest. Compared with cross-validation, tuned with GridSearchCV.
4. **Final test used once** on 20% held-out customers (stratified split).
5. **Threshold 0.3**, chosen on out-of-fold training predictions, because missing a churner costs more than a false alarm.

## Results (held-out test set, 360 customers)
| Metric | Value |
|---|---|
| ROC-AUC | 0.754 |
| Recall (churners found) | 0.75 (3 of 4) |
| Precision (flagged who really leave) | 0.47 (about 1 in 2) |

The model is a **prioritisation tool, not a crystal ball**: it ranks customers well enough to focus retention effort, but about half of the flagged customers would have stayed anyway. Accuracy is not used, because a model that predicts "nobody leaves" already gets 70%.

Top drivers: days since last order, recent spend, usual gap between orders.

## The app
Streamlit app with a flagging setting (shorter list, balanced, catch as many as possible), a customer risk card, the top 10 customers to contact, a CSV download of flagged customers, and a form to score a new customer.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
- `notebooks/01_load_and_label.ipynb`: load data, cutoff, label, features
- `notebooks/02_model_baseline.ipynb`: baseline, models, tuning, final test, saving the model
- `models/churn_model.joblib`: saved model, feature list and threshold
- `app.py`: Streamlit app
- `decisions.md`: every decision and why

## Limitations
- The demo scores customers from the same historical data the model learned from, so it shows how the tool works, not future performance.
- One retailer, one time window. A real deployment would retrain regularly.
- The churn label is a simple 90-day rule. A stricter rule (based on each customer's usual gap) is a possible next step.

Built by Mohamad Fares