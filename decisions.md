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
