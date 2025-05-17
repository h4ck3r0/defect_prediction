import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

df = pd.read_csv("./datasets/aggressive_defects_dataset.csv")


label_encoders = {}
categorical_cols = ["Product_ID", "Product_Type", "Shift", "Operator_Experience_Level"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop(columns=["Date", "Defects"])
y = df["Defects"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.4f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

joblib.dump(model, 'defect_regressor.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl') 
 


plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.title("Actual vs Predicted Defects")
plt.xlabel("Actual Defects")
plt.ylabel("Predicted Defects")
plt.tight_layout()
plt.show()


df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values('Date')

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_sorted, x='Date', y='Defects')
plt.title("Defect Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Defects")
plt.tight_layout()
plt.show()
