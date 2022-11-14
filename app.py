from flask import Flask, render_template, request
import pandas as pd
import numpy as np

import pickle as pkl

model = pkl.load(open('KNN_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_class():
    Glucose= request.form['glucose']
    BloodPressure= request.form['bp']
    SkinThickness= request.form['skin_thk']
    Insulin= request.form['insulin']
    BMI= request.form['bmi']
    DiabetesPedigreeFunction= request.form['dpf']
    Age= request.form['age']

    arr = np.array([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    arr2 = np.array(arr, dtype=float)
    pred = str(model.predict(arr2)[0])
    if pred == '1':
       return 'yes'
    if pred == '0':
        return 'no'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070, debug=True)