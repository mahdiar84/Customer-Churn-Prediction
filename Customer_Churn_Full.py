import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------
def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop ID column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Replace blanks with NaN and handle missing values
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    # Scale numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Encode categorical features
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if df[col].nunique() < 10:  # only encode small categorical features
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df

# -------------------------------
# 2. Train & evaluate model
# -------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    end = time.time()

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    print(f"Execution Time: {end - start:.2f} seconds")

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return acc

# -------------------------------
# 3. Main workflow
# -------------------------------
if __name__ == "__main__":
    # Load dataset
    df = load_and_preprocess(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Features & Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Engineering
    X_train["AverageMonthlyCharges"] = X_train["MonthlyCharges"] / (X_train["tenure"] + 1)
    X_test["AverageMonthlyCharges"] = X_test["MonthlyCharges"] / (X_test["tenure"] + 1)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    # Accuracy dictionary
    accuracy_scores = {}

    # Train & evaluate
    for name, model in models.items():
        accuracy_scores[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test)

    # Accuracy Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("customer_churn_model_accuracy_comparison.png", dpi=300)
    plt.show()