# Decisions: Project 2, Churn Prediction

## Problem and label
1. Goal: predict which regular customers will stop buying in the next 90 days. Regular = 5 or more distinct invoices before the cutoff (`nunique`, not `len`).
2. Project 1's churn rule stays unchanged. Project 2 redefines the label with a cutoff date, so features come only from before it and the label from after it. This avoids leakage.
3. Cutoff date 2011-09-09. History: InvoiceDate < cutoff. Window: >= cutoff and < cutoff + 90 days.
4. Label option A (simple rule): churned = 1 if no purchase in the window. Result: 539 of 1,798 regular customers churned (30%, looser than Project 1's 17%). Churners stay in the data.
5. Option B (stricter rule using the customer's usual gap, computed from pre-cutoff data only) is kept as a possible later comparison.

## Features
6. Ten features built from pre-cutoff history: recency, frequency, monetary, avg_order, variety, usual_gap, recent_orders, recent_share, recent_spend, spend_share. Saved to `data/model_data.csv` (1,798 rows, index Customer ID).

## Modeling (entries 14 to 27)
14. Judge models by recall, precision and ROC-AUC, not accuracy.
15. 80/20 stratified split, random_state=42, so train and test keep the same churn share.
16. Dummy baseline: accuracy 0.70, recall 0. Any real model must beat this.
17. Logistic regression (scaled): recall 0.30 at threshold 0.5, AUC 0.758.
18. Random forest at defaults overfits: train 1.0, test 0.721.
19. Cross-validated AUC: forest 0.773, logistic pipeline 0.763.
20. GridSearchCV best: max_depth=8, min_samples_leaf=20 (CV AUC 0.773, no gain over default).
21. Top feature importances: recency 0.20, recent_spend 0.16, usual_gap 0.15.
22. Dropping the 3 weakest features gave the same AUC. Kept all 10.
23. Final test set used exactly once: forest AUC 0.754.
24. Threshold 0.3, chosen on out-of-fold training probabilities (recall 0.76, precision 0.47), because missing a churner costs more than a false alarm.
25. Test result at 0.3: recall 0.75, precision 0.47, recall for staying customers 0.63.
26. Saved the model with joblib as one bundle: model, threshold and the 10 feature names in order.
27. The saved bundle is what the app loads.

## Deployment
28. Verified the saved bundle (`models/churn_model.joblib`): the reloaded model, feature list and threshold reproduce the final test exactly (ROC-AUC 0.754, recall 0.75, precision 0.47 at t=0.3).
29. Deployment target: a Streamlit app on Streamlit Community Cloud, because it is free, public and takes the code straight from GitHub.
30. The app offers three plain-language flagging levels instead of a raw threshold: Shorter list (0.5), Balanced (0.3, recommended), Catch as many as possible (0.2). Each is a threshold I tested. Technical numbers sit in an expander.
31. Allowed only `data/model_data.csv` into git (`data/*` plus `!data/model_data.csv`). It holds only customer IDs and numeric features, so no personal data. The raw Excel and `sales_clean.csv` stay private.
32. `requirements.txt` lists only what the app needs and pins `scikit-learn==1.9.1`, the version that saved the model, so loading it in the cloud does not break.
33. Live app: https://churn-radar-mohamad.streamlit.app/ (repo `mohamadsfares/churn-prediction`, branch main, file `app.py`). Checked online: same results as local. No feedback form (needs storage, little portfolio value); feedback goes through GitHub issues.

