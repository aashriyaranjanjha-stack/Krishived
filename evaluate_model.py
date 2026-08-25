import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Crop_recommendation.csv")

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)

dt_scores = cross_val_score(
    dt_model, X, y, cv=kf, scoring="accuracy"
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_scores = cross_val_score(
    rf_model, X, y, cv=kf, scoring="accuracy"
)

print("\n========================================")
print("5-FOLD CROSS-VALIDATION COMPARISON")
print("========================================")

print("\nDECISION TREE")
print("----------------------------------------")

for i, score in enumerate(dt_scores, 1):
    print(f"Fold {i}: {score:.4f} ({score * 100:.2f}%)")

print(f"\nMean Accuracy: {dt_scores.mean():.4f} ({dt_scores.mean() * 100:.2f}%)")
print(f"Standard Deviation: {dt_scores.std():.4f} ({dt_scores.std() * 100:.2f} percentage points)")

print("\nRANDOM FOREST")
print("----------------------------------------")

for i, score in enumerate(rf_scores, 1):
    print(f"Fold {i}: {score:.4f} ({score * 100:.2f}%)")

print(f"\nMean Accuracy: {rf_scores.mean():.4f} ({rf_scores.mean() * 100:.2f}%)")
print(f"Standard Deviation: {rf_scores.std():.4f} ({rf_scores.std() * 100:.2f} percentage points)")

print("\n========================================")
print("FINAL COMPARISON")
print("========================================")

print(f"Decision Tree Mean Accuracy : {dt_scores.mean() * 100:.2f}%")
print(f"Random Forest Mean Accuracy : {rf_scores.mean() * 100:.2f}%")

difference = (rf_scores.mean() - dt_scores.mean()) * 100

print(f"\nRandom Forest improvement: {difference:.2f} percentage points")

if rf_scores.mean() > dt_scores.mean():
    print("\nWINNER: Random Forest")
elif dt_scores.mean() > rf_scores.mean():
    print("\nWINNER: Decision Tree")
else:
    print("\nRESULT: Both models performed equally")
