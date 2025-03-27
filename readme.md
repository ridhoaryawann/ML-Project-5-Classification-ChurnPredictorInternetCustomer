# 🧠 Churn Prediction – End-to-End Data Science Project

This project is a **practice exercise** to demonstrate a full **data science workflow**, from raw data all the way to a deployable web application. The goal is to predict whether a customer will **churn** (leave) based on their profile and usage behavior.

✅ Built using:
- **Python (Pandas, NumPy, Scikit-learn)**
- **Streamlit** (for app interface)
- **Joblib** (to save the model and scaler)

---

## 📦 Dataset

This project uses a simplified churn dataset, containing:
- `Age`: Age of the customer
- `Gender`: Male or Female
- `Tenure`: How many months they’ve been a customer
- `MonthlyCharges`: Their monthly bill
- `Churn`: Target variable (Yes = churned, No = stayed)

---

## 🔧 Models Used and How They Work

We used several popular classification models to see which performs best. Below is an easy-to-understand explanation of how each model works:

---

### 1. **Logistic Regression**

🧠 *How it works*:
- Despite the name, this is used for **classification**, not regression.
- It calculates the **probability** of churn using a linear combination of the input features.
- Then applies a **sigmoid function** to output a value between 0 and 1 (interpreted as probability).

📈 *Why it's good*:
- Simple and fast.
- Works well when the relationship between features and the target is roughly linear.
- Easy to interpret: each coefficient tells how much a feature influences churn.

📊 *Performance*:
- Delivered solid accuracy.
- Chosen as the final model for deployment due to balance of performance and speed.

---

### 2. **K-Nearest Neighbors (KNN)**

🧠 *How it works*:
- KNN doesn’t “learn” in the traditional sense.
- When predicting, it looks at the **K closest customers** (based on age, tenure, etc.).
- The prediction is based on the majority label (churn or not) among those neighbors.

📈 *Why it's good*:
- Very intuitive — “people similar to you behave like this”.
- Non-parametric (doesn’t assume a specific data shape).
- Can capture non-linear patterns.

⚠️ *Challenges*:
- Slower with large datasets.
- Sensitive to irrelevant or unscaled features (we used StandardScaler to fix this).

📊 *Performance*:
- Performed well and was competitive with logistic regression.

---

### 3. **Support Vector Machine (SVM)**

🧠 *How it works*:
- Tries to find the **best boundary (hyperplane)** that separates churners from non-churners.
- Can use **kernels** (e.g., RBF) to handle more complex, curved boundaries.
- Focuses on the most “difficult” data points near the boundary (called support vectors).

📈 *Why it's good*:
- Powerful and flexible.
- Works well even when classes aren’t clearly separated.

⚠️ *Challenges*:
- Slower training time.
- Can be sensitive to tuning (e.g., C and kernel parameters).

📊 *Performance*:
- Delivered strong results with the right parameters.

---

### 4. **Decision Tree**

🧠 *How it works*:
- Think of it like 20 questions: it splits data into smaller groups based on conditions (e.g., "Is tenure < 5?").
- Each path in the tree leads to a decision — churn or not.

📈 *Why it's good*:
- Easy to understand and explain.
- Handles both numeric and categorical data.
- Can capture non-linear patterns.

⚠️ *Challenges*:
- Can easily overfit the data (learn noise instead of pattern).
- Needs pruning or parameter tuning to avoid over-complexity.

📊 *Performance*:
- Good after tuning; accuracy improved with depth and leaf size constraints.

---

## ✅ Model Evaluation

We used **accuracy score** to evaluate model performance — i.e., how many correct predictions out of total.

| Model               | Accuracy (approx.) | Notes |
|--------------------|--------------------|-------|
| Logistic Regression | ✅ Good, chosen model | Fast and interpretable |
| K-Nearest Neighbors | ✅ Competitive | Slightly slower, still good |
| Support Vector Machine | ✅ Strong | Best in some cases, but heavier |
| Decision Tree       | ⚠️ Decent | Improved with tuning |

---

## 🌐 Streamlit App

The app lets users enter:
- Age
- Tenure
- Monthly Charges
- Gender

It will instantly predict whether the customer is likely to:
> 🔴 **Churn**  
> or  
> 🟢 **Stay**

Model and scaler were saved using `joblib` and loaded into the app for prediction.

---

## 📁 File Structure

- `2-model-building.ipynb`: Notebook for preprocessing, training, and tuning
- `3-app.py`: Streamlit web application
- `model.pkl`: Final logistic regression model
- `scaler.pkl`: Scaler for input normalization

---

## 🧹 Key Learning Goals

This project demonstrates:

- Data preprocessing (handling categorical and numerical features)
- Splitting data and scaling
- Trying multiple machine learning models
- Hyperparameter tuning using GridSearchCV
- Saving & deploying a model with Streamlit