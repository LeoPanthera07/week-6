import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("suv_data.csv")

print("Part A - Data Loading & Exploration")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

df.columns = [c.strip().replace(" ", "") for c in df.columns]

if "UserID" in df.columns:
    df = df.drop(columns=["UserID"])

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

print("\nPart A - Preprocessing")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Target values:", y.unique())

results = []

for test_size in [0.20, 0.25, 0.30]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "test_size": test_size,
        "accuracy": acc,
        "confusion_matrix": cm,
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test
    })

    print(f"\nTest size: {test_size}")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:")
    print(cm)

best_result = max(results, key=lambda x: x["accuracy"])

print("\nPart B - Best Split Result")
print("Best test size:", best_result["test_size"])
print("Best accuracy:", round(best_result["accuracy"], 4))

best_model = best_result["model"]
best_scaler = best_result["scaler"]
X_test_best = best_result["X_test"]
y_test_best = best_result["y_test"]

x1_min, x1_max = X["Age"].min() - 5, X["Age"].max() + 5
x2_min, x2_max = X["EstimatedSalary"].min() - 10000, X["EstimatedSalary"].max() + 10000

xx1, xx2 = np.meshgrid(
    np.arange(x1_min, x1_max, 1),
    np.arange(x2_min, x2_max, 500)
)

grid = np.c_[xx1.ravel(), xx2.ravel()]
grid_scaled = best_scaler.transform(grid)
Z = best_model.predict(grid_scaled).reshape(xx1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(
    X_test_best["Age"],
    X_test_best["EstimatedSalary"],
    c=y_test_best,
    cmap="coolwarm",
    edgecolor="black"
)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Logistic Regression Decision Boundary")
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
plt.show()

comparison = pd.DataFrame(
    {
        "Test Size": [r["test_size"] for r in results],
        "Accuracy": [round(r["accuracy"], 4) for r in results]
    }
)

print("\nAccuracy Comparison")
print(comparison)

print("\nInterpretation")
print("The model uses Age and EstimatedSalary to classify whether a customer purchased the SUV.")
print("Higher accuracy indicates a better split for this dataset.")
print("The confusion matrix shows correct and incorrect predictions for both classes.")

print("\nPart C")
print("Q1: Logistic Regression is a classification algorithm.")
print("It predicts the probability of belonging to a class such as 0 or 1.")
print("It is called regression because it models log-odds, but it is used for classification.")

print("\nQ2 - Train-test split and scaling code")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

print("Train shape:", X_train_c.shape)
print("Test shape:", X_test_c.shape)
print("Scaled train sample:")
print(X_train_c_scaled[:5])

print("\nQ3: Confusion Matrix")
print("A confusion matrix compares actual class labels with predicted class labels.")
print("It shows True Positives, True Negatives, False Positives, and False Negatives.")
print("It helps us understand what kinds of mistakes the classifier is making.")

print("\nPart D")
prompt = "Explain Logistic Regression with Python example using sklearn on SUV dataset."
ai_output = """
Logistic Regression is a supervised machine learning algorithm used for binary classification.
Using the SUV dataset:
1. Load the dataset with Pandas.
2. Select Age and EstimatedSalary as input features.
3. Select Purchased as the target variable.
4. Split the data into training and testing sets.
5. Scale the features using StandardScaler.
6. Train LogisticRegression from sklearn.
7. Predict on test data.
8. Evaluate using accuracy score and confusion matrix.
"""

print("Prompt:", prompt)
print("AI Output:")
print(ai_output)

print("Evaluation:")
print("The code structure is correct for a basic sklearn pipeline.")
print("The steps are complete because they include loading, preprocessing, splitting, scaling, training, prediction, and evaluation.")
print("The explanation is meaningful and suitable for beginners.")
print("A stronger answer would also explain decision boundary interpretation and why scaling matters for this dataset.")