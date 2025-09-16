import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


# Load the dataset
data = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Datasets\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.DataFrame(data)

# Drop customerID if it exists
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Replace empty strings or spaces with NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Handle missing values (e.g., fill with mode or drop rows)
df.fillna(df.mode().iloc[0], inplace=True)

# Scale numerical features
sc = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_features] = sc.fit_transform(df[numerical_features])

# Encode categorical features
categorical_features = df.select_dtypes(include=['object']).columns
for col in categorical_features:
    if df[col].nunique() < 10:  # Only encode if the number of unique values is less than 10
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Split the dataset into features and target variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create new features based on existing ones
X_train["AverageMonthlyCharges"] = X_train["MonthlyCharges"] / X_train["tenure"]
X_test["AverageMonthlyCharges"] = X_test["MonthlyCharges"] / X_test["tenure"]

# Display the first few rows of the processed DataFrame
#print(X_train.head())

# Models
Models = {
    "LogisticRegression" : LogisticRegression(),
    "RandomForest" : RandomForestClassifier(),
    "GradientBoosting" : GradientBoostingClassifier(),
    
}

accuracy_scores = {}
for name, model in Models.items():
    # Hyperparameter tuning using RandomizedSearchCV
    if name == "RandomForest":
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_features": ["auto", "sqrt"],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
        rf_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        rf_random.fit(X_train, y_train)
        model = rf_random.best_estimator_
    elif name == "GradientBoosting":
        param_dist = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        gb_random = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                       n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        gb_random.fit(X_train, y_train)
        model = gb_random.best_estimator_
    # Train the model
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    # Evaluation
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred)
    ras = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    accuracy_scores[name] = acc
    
    print(f"Model: {name}")
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{cr}")
    print(f"ROC AUC Score: {ras}\n")
    
    # Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
# Accuracy comparison bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("customer_churn_model_accuracy_comparison.png", dpi=300)
plt.show()