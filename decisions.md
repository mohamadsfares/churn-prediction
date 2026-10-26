# Decisions

1. Project 2 uses a cutoff-based label; Project 1 unchanged. Reason: avoid leakage.
2. Dropped rows without Customer ID (22.7%): cannot build a per-customer history. 779,423 rows and 5,878 customers remain.
3. Cutoff = 2011-09-09. History = orders before the cutoff. Window = the 90 days from the cutoff (until 2011-12-08, excluded).
4. Regular customers = 5+ distinct invoices (nunique, not row count) before the cutoff: 1,798 customers.
5. Label rule A: churned = 1 if no purchase in the window. 539 of 1,798 churned (30%). Later: try the stricter Project 1 rule (B) and compare models.
6. Metrics: recall, precision, ROC-AUC. Accuracy is misleading (a "nobody churns" model scores ~70%).
7. Feature plan (history only): recency, frequency (+ recent vs earlier), monetary (total, average order, + recent vs earlier), variety (distinct StockCodes), usual gap between orders.
8. avg_order (total spend / number of orders) is right-skewed (median 316, max 6,819). Keep as is for now; revisit if we use linear models.
9. Added usual_gap (mean days between consecutive orders, from pre-cutoff orders only, one row per Invoice). Median 51 days, range 1.7 to 157.5, no nulls for the 1,798 regular customers.
10. recent_orders = distinct invoices in the 90 days before the cutoff (2011-06-11 to 2011-09-09). Customers with none get 0 (fillna), not NaN. 635 of 1,798 have 0.
11. recent_share = recent_orders / frequency (0 to 1). Measures how much of a customer's order history is recent; low values mean cooling down.
12. Added recent_spend (revenue in the 90 days before the cutoff, 0 if none) and spend_share = recent_spend / monetary (0 to 1). Final feature table: 1,798 rows x 10 columns, no nulls, history-only (no leakage).
13. Joined features and labels into data (1,798 rows x 11 columns: 10 features + churned, no nulls, churn rate 30%). Next: save data, then train/test split and baseline model.
14. Train/test split: 80/20, stratified on churned, random_state=42. Test set (~360 customers) is untouched until final evaluation; all fitting happens on train only.
15. Baseline: DummyClassifier(most_frequent) on the 360-customer test set: accuracy 0.70, churn recall 0.00, precision 0.00. Any real model must beat 0 recall and ROC-AUC 0.5; accuracy is not used to judge models.
16. Scaling: StandardScaler fitted on X_train only, then applied to X_test (no leakage). Used for logistic regression; tree models do not need it.
17. Logistic regression (scaled features, default settings): churn recall 0.30, precision 0.55, accuracy 0.72 on test. Beats the baseline but misses 70% of churners; next: probabilities and threshold, ROC-AUC, then non-linear models.
18. Threshold: lowering logistic regression's cut-off from 0.5 to 0.3 raised churn recall 0.30 -> 0.71 and lowered precision 0.55 -> 0.49. Chosen because missing a churner costs more than a wasted retention offer. Final threshold to be tuned on validation data, not test.
19. Logistic regression ROC-AUC on test: 0.758. This is the number other models must beat. Models will be compared by ROC-AUC (threshold-free), then the threshold is tuned for recall.
21. Random forest with max_depth=5, min_samples_leaf=10: ROC-AUC train 0.85 / test 0.760 (default settings: 1.00 / 0.721, overfit). Roughly ties logistic regression (0.758). Models will be compared with 5-fold cross-validation on train only.
22. 5-fold CV ROC-AUC on train: random forest (max_depth=5, min_samples_leaf=10) 0.773 vs logistic regression pipeline 0.763. Forest has a small edge; proceed to tune it with GridSearchCV.
23. GridSearchCV (max_depth [3,5,8] x min_samples_leaf [5,10,20], 5-fold, roc_auc): best max_depth=8, min_samples_leaf=20, CV AUC 0.773, no gain over the hand-picked forest. Next: feature importance and feature selection.
25. FINAL TEST (used once): tuned random forest (max_depth=8, min_samples_leaf=20, 200 trees, 10 features) ROC-AUC 0.754 on the 360 held-out customers (CV 0.773). Chosen over logistic regression (CV 0.763).
26. Threshold chosen on out-of-fold train probabilities (not test): t = 0.3 gives recall 0.76 / precision 0.47. Chosen because retention offers are cheap compared with losing a regular customer, so catching 3 of 4 churners is worth contacting some loyal customers. The threshold is a business setting and can be changed.
27. FINAL RESULT (test, threshold 0.3): churn recall 0.75, precision 0.47, ROC-AUC 0.754, versus baseline recall 0 / precision 0. Model flags ~3 of 4 churners; about half the flagged customers are false alarms.
28. Verified saved bundle (models/churn_model.joblib): reloaded model, feature list and threshold reproduce the final test exactly (ROC-AUC 0.754, recall 0.75, precision 0.47 at t=0.3).

