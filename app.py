import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import shap
import io
import base64
from flask import Flask, request, render_template

app = Flask(__name__)

# ---------- Load Model, Scaler, Threshold ----------
with open('heart_disease_triage_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('best_threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

# ---------- Load SHAP Explainer (once at startup) ----------
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

# ---------- Feature Names (exact order used in training) ----------
MODEL_FEATURES = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
    'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
    'SkinCancer', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other',
    'Race_White', 'Age_BMI', 'Chronic_Count', 'Inactive_Obese', 'PoorSleep_Mental'
]

# ---------- Mapping Helpers ----------
def map_binary(value): return 1 if value == 'Yes' else 0
def map_sex(value): return 1 if value == 'Male' else 0

def map_age(value):
    age_map = {'18-24':0,'25-29':1,'30-34':2,'35-39':3,'40-44':4,
               '45-49':5,'50-54':6,'55-59':7,'60-64':8,'65-69':9,
               '70-74':10,'75-79':11,'80 or older':12}
    return age_map.get(value, 0)

def map_gen_health(value):
    gen_map = {'Excellent':4,'Very good':3,'Good':2,'Fair':1,'Poor':0}
    return gen_map.get(value, 2)

def map_diabetic(value):
    dia_map = {'No':0,'Borderline':1,'Yes':2}
    return dia_map.get(value, 0)

def map_race(value):
    race_dict = {'Asian':0,'Black':0,'Hispanic':0,'Other':0,'White':0}
    if value in race_dict: race_dict[value] = 1
    return race_dict

# ---------- Generate SHAP Waterfall Plot as Base64 ----------
def generate_shap_plot(df_input, processed_values, shap_values, base_value):
    """Create a waterfall plot and return it as a base64-encoded PNG string."""
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=processed_values,
            feature_names=MODEL_FEATURES
        ),
        show=False,
        max_display=10
    )
    plt.tight_layout()
    
    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    # Encode as base64 for embedding in HTML
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get raw inputs
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

        # 2. Convert to model values
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

        # 4. Engineered features
        processed['Age_BMI'] = age_num * bmi
        processed['Chronic_Count'] = (processed['Diabetic'] + processed['Asthma'] +
                                      processed['KidneyDisease'] + processed['SkinCancer'] +
                                      processed['Stroke'])
        processed['Inactive_Obese'] = 1 if (processed['PhysicalActivity'] == 0 and bmi >= 30) else 0
        processed['PoorSleep_Mental'] = raw['SleepTime'] * raw['MentalHealth']

        # 5. Prepare for model
        df_input = pd.DataFrame([processed])[MODEL_FEATURES]
        scaled = scaler.transform(df_input)
        scaled_df = pd.DataFrame(scaled, columns=MODEL_FEATURES)

        # 6. Predict
        prob = model.predict_proba(scaled)[0, 1]
        is_high = prob >= threshold
        risk_label = "High Risk" if is_high else "Low Risk"

        # 7. Generate SHAP explanation
        shap_vals = explainer.shap_values(scaled_df)
        # TreeExplainer for XGBoost returns 1 array; grab the first row
        shap_row = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        # Use raw feature values for display (more interpretable than scaled)
        display_values = df_input.iloc[0].values
        shap_img = generate_shap_plot(df_input, display_values, shap_row, base_value)

        # 8. Extract Top 5 contributing features for text summary
        feature_contributions = sorted(
            zip(MODEL_FEATURES, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        top_features = []
        for feat, val in feature_contributions:
            direction = "increases" if val > 0 else "decreases"
            top_features.append({
                'name': feat,
                'value': round(float(val), 4),
                'direction': direction
            })

        return render_template('index.html',
                               result=True,
                               probability=f"{prob*100:.2f}%",
                               risk_label=risk_label,
                               risk_class="high" if is_high else "low",
                               raw=raw,
                               shap_image=shap_img,
                               top_features=top_features)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', result=True, error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
