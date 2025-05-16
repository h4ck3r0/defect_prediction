
# 📊 Defects Prediction System

This project builds a machine learning pipeline to **predict the number of product defects** in a manufacturing setup using features such as product ID, type, shift, and operator experience. The model is trained using a **Random Forest Regressor** and provides performance metrics and insightful visualizations.

---

## 📁 Dataset

Path: `./datasets/aggressive_defects_dataset.csv`

### 📌 Columns:
- `Date`: Date of production
- `Product_ID`: Identifier for the product
- `Product_Type`: Category/type of the product
- `Shift`: Production shift (e.g., Morning, Evening)
- `Operator_Experience_Level`: Operator’s experience level
- `Defects`: Number of defects observed (target variable)

---

## 🛠 Features

✅ Label encoding for categorical features  
✅ Model training using `RandomForestRegressor`  
✅ Evaluation using R² Score and RMSE  
✅ Visualizations for feature importance, prediction accuracy, and defect trends  
✅ Model & encoders saved using `joblib`

---

## 📦 Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
````

---

## 🚀 How to Run

1. Place your dataset at:

   ```
   ./datasets/aggressive_defects_dataset.csv
   ```

2. Run the script:

   ```bash
   python defects_predictor.py
   ```

3. The script will:

   * Train and evaluate a model
   * Save `defect_regressor.pkl` and `label_encoders.pkl`
   * Show feature importance and prediction plots
   * Plot defects trend over time

---

## 📈 Visualizations

### 🔹 Top 10 Feature Importances

Shows the most influential features for defect prediction.

### 🔹 Actual vs Predicted Defects

Compares model predictions with real defect counts.

### 🔹 Defects Over Time

Line plot to monitor defect trends chronologically.

---

## 🧪 Model Files

* `defect_regressor.pkl`: Trained Random Forest model
* `label_encoders.pkl`: Saved encoders for categorical variables

---

## 🔍 Predict from CLI

You can later load the model and encoders to predict from new data:

```python
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("defect_regressor.pkl")
encoders = joblib.load("label_encoders.pkl")

# Sample input (replace with your values)
input_dict = {
    "Product_ID": "P123",
    "Product_Type": "A",
    "Shift": "Night",
    "Operator_Experience_Level":
"Intermediate",
"Machine_usage_hour": 15
}

# Encode input
for col in input_dict:
    le = encoders[col]
    input_dict[col] = le.transform([input_dict[col]])[0]

# Predict
X_input = pd.DataFrame([input_dict])
predicted_defects = model.predict(X_input)[0]
print(f"Predicted Defects: {predicted_defects:.2f}")
```

---

## 📤 Future Improvements

* Add a web dashboard using Flask or Streamlit
* Hyperparameter tuning with GridSearchCV
* Integration with real-time factory data sources
* Support for more advanced models (XGBoost, CatBoost)

---

## 👨‍💻 Author

**Raj Aryan**
🎓 B.Tech | RNSIT
🔗 [LinkedIn](https://www.linkedin.com/in/h4ck3r0)
🔗 [GitHub](https://github.com/h4ck3r0)

---

## 📝 License

This project is open-source and free to use under the [MIT License](LICENSE).



