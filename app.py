import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import os

app = Flask(__name__)

# Load model, scaler, and threshold
MODEL_PATH = 'heart_disease_triage_model.pkl'
SCALER_PATH = 'scaler.pkl'
THRESHOLD_PATH = 'best_threshold.txt'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read().strip())

# Features expected by the model in EXACT order (from your training)
MODEL_FEATURES = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
    'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
    'SkinCancer', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other',
    'Race_White', 'Age_BMI', 'Chronic_Count', 'Inactive_Obese', 'PoorSleep_Mental'
]

# ---------- Helper mapping functions ----------
def map_binary(value):
    return 1 if value == 'Yes' else 0

def map_sex(value):
    return 1 if value == 'Male' else 0

def map_age(value):
    age_map = {
        '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4,
        '45-49': 5, '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9,
        '70-74': 10, '75-79': 11, '80 or older': 12
    }
    return age_map.get(value, 0)

def map_gen_health(value):
    gen_map = {'Excellent': 4, 'Very good': 3, 'Good': 2, 'Fair': 1, 'Poor': 0}
    return gen_map.get(value, 2)

def map_diabetic(value):
    dia_map = {'No': 0, 'Borderline': 1, 'Yes': 2}
    return dia_map.get(value, 0)

def map_race(value):
    # One-hot encode exactly like training (5 columns)
    race_dict = {'Asian': 0, 'Black': 0, 'Hispanic': 0, 'Other': 0, 'White': 0}
    if value in race_dict:
        race_dict[value] = 1
    return race_dict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get raw user-friendly inputs from form
        raw = {
            'BMI': float(request.form.get('BMI')),
            'Smoking': request.form.get('Smoking'),
            'AlcoholDrinking': request.form.get('AlcoholDrinking'),
            'Stroke': request.form.get('Stroke'),
            'PhysicalHealth': float(request.form.get('PhysicalHealth')),
            'MentalHealth': float(request.form.get('MentalHealth')),
            'DiffWalking': request.form.get('DiffWalking'),
            'Sex': request.form.get('Sex'),
            'AgeCategory': request.form.get('AgeCategory'),
            'Diabetic': request.form.get('Diabetic'),
            'PhysicalActivity': request.form.get('PhysicalActivity'),
            'GenHealth': request.form.get('GenHealth'),
            'SleepTime': float(request.form.get('SleepTime')),
            'Asthma': request.form.get('Asthma'),
            'KidneyDisease': request.form.get('KidneyDisease'),
            'SkinCancer': request.form.get('SkinCancer'),
            'Race': request.form.get('Race')
        }

        # 2. Convert to numeric values used by the model
        bmi = raw['BMI']
        age_num = map_age(raw['AgeCategory'])
        diabetic_num = map_diabetic(raw['Diabetic'])
        
        processed = {
            'BMI': bmi,
            'Smoking': map_binary(raw['Smoking']),
            'AlcoholDrinking': map_binary(raw['AlcoholDrinking']),
            'Stroke': map_binary(raw['Stroke']),
            'PhysicalHealth': raw['PhysicalHealth'],
            'MentalHealth': raw['MentalHealth'],
            'DiffWalking': map_binary(raw['DiffWalking']),
            'Sex': map_sex(raw['Sex']),
            'AgeCategory': age_num,
            'Diabetic': diabetic_num,
            'PhysicalActivity': map_binary(raw['PhysicalActivity']),
            'GenHealth': map_gen_health(raw['GenHealth']),
            'SleepTime': raw['SleepTime'],
            'Asthma': map_binary(raw['Asthma']),
            'KidneyDisease': map_binary(raw['KidneyDisease']),
            'SkinCancer': map_binary(raw['SkinCancer']),
        }
        
        # 3. One-hot encode Race
        race_oh = map_race(raw['Race'])
        processed['Race_Asian'] = race_oh['Asian']
        processed['Race_Black'] = race_oh['Black']
        processed['Race_Hispanic'] = race_oh['Hispanic']
        processed['Race_Other'] = race_oh['Other']
        processed['Race_White'] = race_oh['White']

        # 4. Recreate the engineered features (exactly as in training)
        processed['Age_BMI'] = age_num * bmi
        
        # Chronic count: sum of (Diabetic, Asthma, Kidney, SkinCancer, Stroke)
        processed['Chronic_Count'] = (processed['Diabetic'] + processed['Asthma'] + 
                                      processed['KidneyDisease'] + processed['SkinCancer'] + 
                                      processed['Stroke'])
        
        # Inactive_Obese: 1 if NO physical activity AND BMI >= 30
        processed['Inactive_Obese'] = 1 if (processed['PhysicalActivity'] == 0 and bmi >= 30) else 0
        
        # PoorSleep_Mental: SleepTime * MentalHealth days
        processed['PoorSleep_Mental'] = raw['SleepTime'] * raw['MentalHealth']

        # 5. Create DataFrame in the exact model order and scale
        df_input = pd.DataFrame([processed])[MODEL_FEATURES]
        scaled = scaler.transform(df_input)
        
        # 6. Predict
        prob = model.predict_proba(scaled)[0, 1]  # probability of heart disease
        
        is_high = prob >= threshold
        risk_label = "High Risk" if is_high else "Low Risk"
        
        return render_template('index.html',
                               result=True,
                               probability=f"{prob*100:.2f}%",
                               risk_label=risk_label,
                               risk_class="high" if is_high else "low",
                               raw=raw)  # pass raw input to show summary
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                               result=True,
                               error=str(e))

if __name__ == '__main__':
    # For production, Render will use Gunicorn.
    # This line is only for when you run the script locally.
    app.run(host='0.0.0.0', port=5000, debug=False)