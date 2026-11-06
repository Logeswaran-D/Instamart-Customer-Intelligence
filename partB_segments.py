"""Part B (tasks 1-3) - RFM table, K-means segmentation, segment profiles and PCA.
Run after:  python partA_clean.py
Writes:     outputs/customer_segments.csv
"""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = o[o["status"] == "completed"]

# ------------------------------------------------------------- 1. RFM table
rfm = done.groupby("customer_id").agg(last_order=("order_ts", "max"),
                                      frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum"))
rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()

print("=" * 70, "\n1. RFM TABLE\n" + "=" * 70)
print("customers:", len(rfm), "(ref 1175)")
print(f"recency   median {rfm.recency.median():.0f}  min {rfm.recency.min()}  max {rfm.recency.max()}"
      "   (ref 19 / 0 / 180)")
print(f"frequency mean {rfm.frequency.mean():.2f}  max {rfm.frequency.max()}   (ref 9.46 / 33)")
print(f"monetary  mean {rfm.monetary.mean():.2f}  max {rfm.monetary.max():.2f}   (ref 2552.06 / 14766.55)")

# --------------------------------------------------------- 2. K-means, K=2..8
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
ktab = pd.DataFrame(rows)
ktab["inertia_drop"] = (-ktab["inertia"].diff()).round(1)
print("\n" + "=" * 70, "\n2. K-MEANS INERTIA AND SILHOUETTE\n" + "=" * 70)
print(ktab.to_string(index=False))
print("chosen K:", FINAL_K)

# ------------------------------------------------ 3. profile, name, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     recency=("recency", "mean"),
                                     frequency=("frequency", "mean"),
                                     monetary=("monetary", "mean")).round(2)

# Names are assigned from the profile, not from the cluster number
# (cluster numbers are arbitrary labels with no order).
premium_id = prof["monetary"].idxmax()
rest = prof.drop(premium_id)
loyal_id = rest["frequency"].idxmax()
rest = rest.drop(loyal_id)
lapsing_id = rest["recency"].idxmax()
occasional_id = rest.drop(lapsing_id).index[0]
NAMES = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars",
         occasional_id: "Occasional Shoppers", lapsing_id: "Drifting Away"}
ACTIONS = {premium_id: "Priority support, early access to new electronics, premium loyalty perks",
           loyal_id: "Subscription / basket-builder offers to grow order value",
           occasional_id: "Personalised reminders and category cross-sell coupons",
           lapsing_id: "Win-back campaign with a time-limited discount"}
prof["segment_name"] = prof.index.map(NAMES)
prof["is_premium"] = prof.index == premium_id
prof["action"] = prof.index.map(ACTIONS)
print("\n" + "=" * 70, "\n3. SEGMENT PROFILE (K=%d)\n" % FINAL_K + "=" * 70)
print(prof.to_string())
print("Note: segment_id values are only labels; 0..3 carry no ordering.")

pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: {evr[0]:.3f}, {evr[1]:.3f}  total {evr.sum():.3f}"
      "  (ref 0.634 / 0.218 / 0.852)")

rfm["segment_name"] = rfm["segment_id"].map(NAMES)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)
print("\nWrote outputs/customer_segments.csv | premium customers:", int(rfm["is_premium"].sum()))
