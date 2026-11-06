"""Validation of Part A reference checkpoints."""
import sqlite3

import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
o = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")

for t, n in [("customers", 1200), ("products", 40), ("orders", 12309)]:
    got = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {t}", con)["n"][0]
    assert got == n, f"{t} rows {got} != {n}"

assert len(o) == 12309 and o["order_id"].is_unique
vc = o["status"].value_counts()
assert vc["completed"] == 11110 and vc["cancelled"] == 1199
assert "pending" not in vc.index
assert o["order_amount"].isna().sum() == 0
done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]
assert done["delivery_minutes"].isna().sum() == 0
assert canc["delivery_minutes"].isna().sum() == 1199
assert abs(done["order_amount"].sum() - 2998668.80) < 0.01

assert c["city"].value_counts().to_dict() == {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c["age"].isna().sum() == 0 and c["age"].between(18, 60).all()

print("ALL CHECKPOINTS PASSED")
