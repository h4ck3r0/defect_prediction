from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('./models/defect_regressor.pkl')
label_encoders = joblib.load('./models/label_encoders.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    input_data = {
        'Product_ID': request.form['Product_ID'],
        'Product_Type': request.form['Product_Type'],
        'Production_Volume': float(request.form['Production_Volume']),
        'Shift': request.form['Shift'],
        'Operator_Experience_Level': request.form['Operator_Experience_Level'],
        'Machine_Usage_Hours': float(request.form['Machine_Usage_Hours']),
        'Temperature': float(request.form['Temperature']),
        'Humidity': float(request.form['Humidity']),
        'Previous_Day_Defects': float(request.form['Previous_Day_Defects'])
    }

    df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    # Predict
    prediction = model.predict(df)
    predicted_defects = int(prediction[0])

    return render_template('result.html', defects=predicted_defects)

if __name__ == '__main__':
    app.run(debug=True)



