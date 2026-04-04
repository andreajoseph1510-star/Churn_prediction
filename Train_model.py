import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("telco_churn.csv")

# ── 2. Clean data ─────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. Encode categorical columns ─────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ── 4. Split features and target ──────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Handle class imbalance with SMOTE ──────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ── 6. Scale features ─────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_sc, y_train_res)

print("✅ Model trained!")

# ── 8. Save everything ────────────────────────────────────────
artifacts = {
    "model": model,
    "scaler": scaler,
    "le_dict": le_dict,
    "cat_cols": cat_cols,
    "feature_names": list(X.columns)
}
with open("churn_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

# ── 9. Export cleaned data for Power BI ───────────────────────
df["ChurnLabel"] = df["Churn"].map({1: "Yes", 0: "No"})
df.to_csv("powerbi_export.csv", index=False)

print("✅ Model saved → churn_model.pkl")
print("✅ Power BI data exported → powerbi_export.csv")
from sklearn.metrics import classification_report, roc_auc_score

preds = model.predict(X_test_sc)
proba = model.predict_proba(X_test_sc)[:,1]

print("\n📊 Model Performance on Test Data:")
print(f"AUC Score : {roc_auc_score(y_test, proba):.4f}")
print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))