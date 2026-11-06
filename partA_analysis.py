"""Part A - run the 7 SQL queries and the pandas cross-check / EDA."""
import re
import sqlite3

import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d)[^\n]*\n", text, flags=re.M)
queries = dict(zip(parts[1::2], parts[2::2]))
res = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

print("Q1 customers:", len(res["Q1"]), "(ref 30)")
q2 = res["Q2"]
print("Q2 rows:", len(q2), "(ref 30) | total:", round(q2.revenue.sum(), 2), "(ref 2998668.80)")
print("   largest cell:", q2.iloc[0].to_dict(), "(ref Bengaluru/Electronics 233909.45)")
print("Q3 products:", len(res["Q3"]), "(ref 11)")
print("Q4 customers:", len(res["Q4"]), "(ref 25)")
q5 = res["Q5"]
print("Q5 customers:", len(q5), "(ref 550) | average:", q5.avg_total.iloc[0], "(ref 2552.06)")
q6 = res["Q6"]
print("Q6 top per city:\n", q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("   rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()), "(ref 8)")
q7 = res["Q7"]
print("Q7 rows:", len(q7), "(ref 183) | last cumulative:", q7.cumulative_revenue.iloc[-1], "(ref 2998668.80)")

# ---------------------------------------------------------- pandas cross-check
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
o = pd.read_csv("data/clean/orders.csv")
done = o[o.status == "completed"]
m = done.merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")

pv = pd.pivot_table(m, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\nPivot grand total:", round(pv.loc["All", "All"], 2), "| SQL total:", round(q2.revenue.sum(), 2))

city_stats = m.groupby("city").agg(total=("order_amount", "sum"),
                                   mean=("order_amount", "mean"),
                                   count=("order_amount", "count")).round(2)
print(city_stats)

# ------------------------------------------------------------------------- EDA
a = done["order_amount"]
q1, q3 = a.quantile(0.25), a.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {a.mean():.2f} | median {a.median():.2f} | skew {a.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | fences [{lo:.2f}, {hi:.2f}]")
print(f"outliers below: {(a < lo).sum()} | above: {(a > hi).sum()}")
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3))
