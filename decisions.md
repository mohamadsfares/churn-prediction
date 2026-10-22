# Decisions

1. Project 2 uses a cutoff-based label; Project 1 unchanged. Reason: avoid leakage.
2. Dropped rows without Customer ID (22.7%): cannot build a per-customer history. 779,423 rows and 5,878 customers remain.
3. Cutoff = 2011-09-09. History = orders before the cutoff. Window = the 90 days from the cutoff (until 2011-12-08, excluded).
4. Regular customers = 5+ distinct invoices (nunique, not row count) before the cutoff: 1,798 customers.
5. Label rule A: churned = 1 if no purchase in the window. 539 of 1,798 churned (30%). Later: try the stricter Project 1 rule (B) and compare models.
6. Metrics: recall, precision, ROC-AUC. Accuracy is misleading (a "nobody churns" model scores ~70%).
7. Feature plan (history only): recency, frequency (+ recent vs earlier), monetary (total, average order, + recent vs earlier), variety (distinct StockCodes), usual gap between orders.
8. avg_order (total spend / number of orders) is right-skewed (median 316, max 6,819). Keep as is for now; revisit if we use linear models.

