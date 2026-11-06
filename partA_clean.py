"""Part A - audit, clean and store the Instamart data.
Run after:  python generate_data.py
"""
import os
import sqlite3

import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw", "data/clean"
os.makedirs(CLEAN, exist_ok=True)

cust = pd.read_csv(f"{RAW}/customers.csv")
prod = pd.read_csv(f"{RAW}/products.csv")
orders = pd.read_csv(f"{RAW}/orders.csv")
tickets = pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False)

# ------------------------------------------------------------------ 2. AUDIT
print("=" * 70, "\nAUDIT\n" + "=" * 70)
for name, df in [("customers", cust), ("products", prod), ("orders", orders), ("tickets", tickets)]:
    print(f"\n--- {name}: shape {df.shape}")
    print(pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    }))

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "customers: blank age": int(cust["age"].isna().sum()),
    "customers: age outside 10-100": int(((cust["age"] < 10) | (cust["age"] > 100)).sum()),
    "customers: raw city spellings": int(cust["city"].nunique()),
    "customers: raw gender spellings (non-blank)": int(cust["gender"].dropna().nunique()),
    "customers: blank gender": int(cust["gender"].isna().sum()),
}
print("\nAUDIT FIGURES")
for k, v in audit.items():
    print(f"  {k:45s} {v}")

# ------------------------------------------------------------ 3. CLEAN CUSTOMERS
c = cust.copy()
c["city"] = c["city"].str.strip().str.title()
g = c["gender"].fillna("").str.strip().str.lower()
c["gender"] = g.map({"male": "male", "m": "male", "female": "female", "f": "female"}).fillna("unknown")
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c["age"] = c["age"].where(c["age"].between(10, 100))          # out of range -> missing
n_age_fill = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])

print("\n" + "=" * 70, "\nCLEAN CUSTOMERS\n" + "=" * 70)
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())
print("missing ages:", int(c["age"].isna().sum()), "| age range:", c["age"].min(), "-", c["age"].max())
print(f"ages set to missing (out of range): {n_bad_age}; ages filled with city median: {n_age_fill}")

# --------------------------------------------------------------- 4. CLEAN ORDERS
o = orders.copy()
log = {}

# i. datetime
o["order_ts"] = pd.to_datetime(o["order_ts"])

# ii. exact duplicates
before = len(o)
o = o.drop_duplicates()
log["ii. exact duplicate rows removed"] = before - len(o)

# iii. keep latest record per order_id
before = len(o)
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")
log["iii. earlier records of same order_id removed"] = before - len(o)

# iv. amounts
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o["order_amount"].isna()
calc = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
o.loc[miss, "order_amount"] = calc[miss]
log["iv. missing order_amount filled from price"] = int(miss.sum())

# v. delivery minutes
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = np.nan
log["v. negative delivery_minutes set to missing"] = int(neg.sum())
comp = o["status"] == "completed"
cat_median = o[comp].groupby("category")["delivery_minutes"].transform("median")
need = comp & o["delivery_minutes"].isna()
o.loc[comp, "delivery_minutes"] = o.loc[comp, "delivery_minutes"].fillna(cat_median)
log["v. completed delivery_minutes filled (category median)"] = int(need.sum())

o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

print("\n" + "=" * 70, "\nCLEAN ORDERS\n" + "=" * 70)
for k, v in log.items():
    print(f"  {k:60s} {v}")
print("rows:", len(o), "| status counts:", o["status"].value_counts().to_dict())
print("missing order_amount:", int(o["order_amount"].isna().sum()))
print("missing delivery (completed):", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()),
      "| (cancelled):", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
print("completed revenue INR:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ------------------------------------------------------------- 5. STORE
o_out = o.copy()
o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = c.copy()
c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")

c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
prod.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)

if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
c_out.to_sql("customers", con, index=False)
prod.to_sql("products", con, index=False)
o_out.to_sql("orders", con, index=False)
con.close()
print("\nWrote data/clean/*.csv and data/instamart.db")
