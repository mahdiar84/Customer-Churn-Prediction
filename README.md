# 📊 Customer Churn Prediction  

This project aims to **predict customer churn** (whether a customer will leave the company or not) using machine learning models.  
The dataset is the **Telco Customer Churn dataset**, which includes customer demographics, account information, and service usage.  

---

## 📂 Dataset  
- **Source**: Telco Customer Churn (Kaggle)  
- **Target Variable**: `Churn` (Yes/No → Encoded as 1/0)  
- **Features**: Customer demographics, tenure, contract type, monthly charges, total charges, etc.  

---

## ⚙️ Preprocessing Steps  
1. **Removed irrelevant column** (`customerID`).  
2. **Handled missing values** → replaced blanks with mode.  
3. **Scaled numerical features** → using `StandardScaler`.  
4. **Encoded categorical features** → using `LabelEncoder`.  
5. **Handled class imbalance** → applied `SMOTE`.  
6. **Feature Engineering** → created `AverageMonthlyCharges = MonthlyCharges / tenure`.  

---

## 🤖 Models Used  
- **Logistic Regression**  
- **Random Forest Classifier**  
- **Gradient Boosting Classifier**  

---

## 📈 Evaluation Metrics  
Each model was evaluated using:  
- **Accuracy Score**  
- **Classification Report** (Precision, Recall, F1-score)  
- **Confusion Matrix**  
- **ROC-AUC Score**  

---

## 📊 Results  

| Model               | Accuracy | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | ~0.80    | ~0.85   |
| Random Forest        | ~0.85    | ~0.90   |
| Gradient Boosting    | ~0.86    | ~0.91   |

*(Results may vary depending on parameters and random state)*  

---

## 📌 Visualizations  
- Confusion Matrices for each model.  
- ROC Curves comparison.  
- Accuracy comparison bar chart.  

---

## 🚀 How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/customer-churn-prediction.git
