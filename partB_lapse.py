"""Part B (tasks 4-8) - lapse prediction without leakage.
Run after:  python partB_segments.py
Writes:     outputs/lapse_model.joblib, outputs/retention_list.csv
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id", how="left")

# ------------------------------------------------------- 4. define the problem
before = o[o["order_ts"] < CUTOFF]                       # ONLY pre-cutoff orders feed the features
done_b = before[before["status"] == "completed"]
sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < END) & (o["status"] == "completed")]

cohort = sorted(done_b["customer_id"].unique())
f = pd.DataFrame(index=pd.Index(cohort, name="customer_id"))
g = done_b.groupby("customer_id")
f["recency"] = (CUTOFF - g["order_ts"].max()).dt.days
f["frequency"] = g["order_id"].nunique()
f["monetary"] = g["order_amount"].sum()
f["avg_discount"] = g["discount_pct"].mean()
f["avg_delivery"] = g["delivery_minutes"].mean()
all_b = before.groupby("customer_id")
f["cancel_rate"] = all_b["status"].apply(lambda s: (s == "cancelled").mean()).reindex(f.index)
fav = (done_b.groupby(["customer_id", "category"]).size().rename("n").reset_index()
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"])
f["fav_category"] = fav
cc = c.set_index("customer_id")
f["city"] = cc["city"]
f["age"] = cc["age"]
f["gender"] = cc["gender"]
f["tenure_days"] = (CUTOFF - cc["signup_date"]).dt.days
f["lapsed"] = (~f.index.isin(sept["customer_id"])).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = f[NUM + CAT], f["lapsed"]
assert X.isna().sum().sum() == 0, "missing values in features"

print("=" * 70, "\n4. LAPSE PROBLEM\n" + "=" * 70)
print(f"cohort {len(f)} (ref 1156) | lapsed {y.sum()} ({y.mean()*100:.2f}%) (ref 443, 38.32%)")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(X_tr)} | test {len(X_te)} (ref 924 / 232)")
print("no missing values in any feature column: True")

base = np.zeros(len(y_te), dtype=int)
print(f"baseline 'never lapsed': accuracy {accuracy_score(y_te, base):.3f} | "
      f"lapsed F1 {f1_score(y_te, base, zero_division=0):.3f} (ref 0.616 / 0)")

# ------------------------------------------------- 5. leakage-free preprocessing
def make_prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

prep_check = make_prep().fit(X_tr)                        # fitted on training rows only
fi = NUM.index("frequency")
print("\n" + "=" * 70, "\n5. PREPROCESSING\n" + "=" * 70)
print(f"scaler mean of frequency (train only): {prep_check.named_transformers_['num'].mean_[fi]:.4f} | "
      f"full-cohort mean: {X['frequency'].mean():.4f}")
print("They differ because the scaler learned only from the 924 training rows; the test rows were never seen.")

# ----------------------------------------------------- 6. train, tune, evaluate
def report(name, y_true, pred):
    cm = confusion_matrix(y_true, pred)
    r = {"model": name,
         "precision": round(precision_score(y_true, pred, zero_division=0), 3),
         "recall": round(recall_score(y_true, pred, zero_division=0), 3),
         "f1": round(f1_score(y_true, pred, zero_division=0), 3)}
    print(f"{name}: confusion matrix [[TN FP] [FN TP]] = {cm.tolist()} | "
          f"precision {r['precision']} recall {r['recall']} F1 {r['f1']}")
    return r

lr_gs = GridSearchCV(Pipeline([("prep", make_prep()), ("clf", LogisticRegression(max_iter=1000))]),
                     {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
rf_gs = GridSearchCV(Pipeline([("prep", make_prep()),
                               ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_tr, y_tr)
print("\n" + "=" * 70, "\n6. TUNING AND TEST EVALUATION\n" + "=" * 70)
print("LogReg best params:", lr_gs.best_params_, "| best CV F1:", round(lr_gs.best_score_, 3))
print("RandomForest best params:", rf_gs.best_params_, "| best CV F1:", round(rf_gs.best_score_, 3),
      f"| fits: {len(rf_gs.cv_results_['params']) * 5}")
lr, rf = lr_gs.best_estimator_, rf_gs.best_estimator_
res_lr = report("LogisticRegression", y_te, lr.predict(X_te))
res_rf = report("RandomForest      ", y_te, rf.predict(X_te))
best = rf if res_rf["f1"] >= res_lr["f1"] else lr
best_name = "RandomForest" if best is rf else "LogisticRegression"
print("Selected model:", best_name)

# ------------------------------------------------------- 7. class balance
tr = X_tr.assign(lapsed=y_tr)
maj, mino = tr[tr.lapsed == 0], tr[tr.lapsed == 1]
extra = mino.sample(n=len(maj) - len(mino), replace=True, random_state=42)
bal = pd.concat([tr, extra])                               # training rows only; test untouched
print("\n" + "=" * 70, "\n7. CLASS-BALANCE EXPERIMENT\n" + "=" * 70)
print("balanced training counts:", bal.lapsed.value_counts().to_dict(), "(ref 570 each)")
lr_bal = Pipeline([("prep", make_prep()),
                   ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))])
lr_bal.fit(bal[NUM + CAT], bal["lapsed"])
cmp = pd.DataFrame([report("LogReg un-balanced", y_te, lr.predict(X_te)),
                    report("LogReg over-sampled", y_te, lr_bal.predict(X_te))])
print(cmp.to_string(index=False))

# ------------------------------------------------ 8. interpret, persist, use
print("\n" + "=" * 70, "\n8. INTERPRETATION\n" + "=" * 70)
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("Top 5 random-forest features:\n", imp.head(5).round(3).to_string())
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["prep"].get_feature_names_out())
print("LogReg coefficients (selected):\n",
      coef[["num__recency", "num__frequency", "num__cancel_rate", "num__monetary"]].round(3).to_string())

joblib.dump(best, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = (reloaded.predict(X_te) == best.predict(X_te)).all()
print("\nSaved outputs/lapse_model.joblib | reloaded predictions identical:", bool(same))

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
te = X_te.copy()
te["lapse_probability"] = best.predict_proba(X_te)[:, 1].round(4)
te["actually_lapsed"] = y_te
prem = te[te.index.isin(seg.index[seg["is_premium"]])]
ret = prem.sort_values("lapse_probability", ascending=False).head(10).reset_index()
ret[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv("outputs/retention_list.csv", index=False)
print(f"Premium customers in test set: {len(prem)} | retention list rows: {len(ret)} | "
      f"truly lapsed in list: {int(ret.actually_lapsed.sum())}")
print(ret[["customer_id", "lapse_probability", "actually_lapsed"]].to_string(index=False))
print("Note: the Premium flag is a post-hoc whole-window label (it includes September), "
      "so this list is illustrative, not a leakage-free evaluation.")
