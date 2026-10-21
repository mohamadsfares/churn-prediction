# Churn Prediction

## Problem
Regular customers stop buying, and the business only notices months later, when it is too late to react.

## Why it matters
Keeping a customer costs less than finding a new one. If we know who is at risk early, the team can call them or offer a deal.

## Aim
Predict, for each regular customer (5+ orders), the chance they will churn in the next 90 days, then deploy the model so anyone can get a risk score.

## Data
Online Retail II (UCI), Dec 2009 - Dec 2011, cleaned in Project 1 (`sales_clean.csv`).

## Method note
Features are built only from data before a cutoff date. The label comes from behaviour after it, to avoid leakage.

## Success
Only ~17% churn, so accuracy is misleading. We judge by recall, precision and ROC-AUC on churners.

## Plan
1. Load data
2. Define cutoff and label
3. Build features
4. Train a baseline
5. Train better models
6. Evaluate
7. Deploy
8. Document