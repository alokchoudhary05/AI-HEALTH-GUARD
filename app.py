from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
import json

# Flask app
app = Flask(__name__)

# Load datasets
precautions = pd.read_csv("datasets/Symptoms-Disease Datasets/precautions_df.csv")
workout = pd.read_csv("datasets/Symptoms-Disease Datasets/workout_df.csv")
description = pd.read_csv("datasets/Symptoms-Disease Datasets/description.csv", encoding='latin-1')
medications = pd.read_csv('datasets/Symptoms-Disease Datasets/medications.csv')
diets = pd.read_csv("datasets/Symptoms-Disease Datasets/diets.csv")

# Load the unique symptoms data
unique_symptoms = pd.read_csv("datasets/Symptoms-Disease Datasets/unique_symptoms.csv")

# Load model
with open('datasets/Symptoms-Disease Datasets/NaiveBayes.pkl', 'rb') as model_file:
    svc = pickle.load(model_file)

# Load label encoder
with open('datasets/Symptoms-Disease Datasets/label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)



#Advance Load models
pregnancy_model = joblib.load(open("datasets/advance/models/pregnancy_model.pkl", 'rb'))
heart_model = pickle.load(open("datasets/advance/models/Heart.sav", 'rb'))
diabetic_model = pickle.load(open("datasets/advance/models/Diabetes.sav", 'rb'))

# Normalize column names and data to handle inconsistencies
workout.rename(columns={'disease': 'Disease'}, inplace=True)

def normalize_column(df, column_name):
    df[column_name] = df[column_name].str.strip().str.lower()

for df in [description, precautions, medications, workout, diets]:
    normalize_column(df, 'Disease')

# Function to predict disease based on symptoms
def predict_disease(symptoms):
    symptoms_dict = {symptom: 0 for symptom in svc.feature_names_in_}
    for symptom in symptoms:
        symptom = symptom.strip().lower()
        if symptom in symptoms_dict:
            symptoms_dict[symptom] = 1
    
    # Debugging: Print symptoms dictionary
    print(f"Symptoms dictionary: {symptoms_dict}")
    
    input_data = pd.DataFrame([symptoms_dict])
    
    # Debugging: Print input data for the model
    print(f"Input data for model: {input_data}")
    
    predicted_disease = svc.predict(input_data)
    disease_name = le.inverse_transform(predicted_disease)[0].strip().lower()
    
    # Debugging: Print predicted disease name
    print(f"Predicted disease name: {disease_name}")
    
    return disease_name




# Helper function to fetch recommendations
def helper(disease):
    # Initialize variables to store recommendations
    desc = 'No description available'
    pre = ['No precautions available']
    med = ['No medications available']
    die = ['No diet information available']
    wrkout = ['No workout information available']

    # Fetch description if disease exists in description dataset
    if disease in description['Disease'].values:
        desc = description[description['Disease'] == disease]['Description'].values[0]

    # Fetch precautions if disease exists in precautions dataset
    if disease in precautions['Disease'].values:
        pre = []
        precaution_columns = [col for col in precautions.columns if 'Precaution_' in col]
        precautions_list = precautions[precautions['Disease'] == disease][precaution_columns].values[0]
        for precaution in precautions_list:
            if pd.notna(precaution):
                pre.append(precaution)

    # Fetch medications if disease exists in medications dataset
    if disease in medications['Disease'].values:
        med = []
        medication_columns = [col for col in medications.columns if 'Medication_' in col]
        medications_list = medications[medications['Disease'] == disease][medication_columns].values[0]
        for medication in medications_list:
            if pd.notna(medication):
                med.append(medication)

    # Fetch diets if disease exists in diets dataset
    if disease in diets['Disease'].values:
        diet_columns = [col for col in diets.columns if 'Diet_' in col]
        diets_list = diets[diets['Disease'] == disease][diet_columns].values[0]
        die = []
        for diet in diets_list:
            if pd.notna(diet):
                die.append(diet)

    # Fetch workouts if disease exists in workout dataset
    if disease in workout['Disease'].values:
        workout_columns = [col for col in workout.columns if 'workout_' in col]
        workouts_list = workout[workout['Disease'] == disease][workout_columns].values[0]
        wrkout = []
        for workout_item in workouts_list:
            if pd.notna(workout_item):
                wrkout.append(workout_item)

    return desc, pre, med, die, wrkout


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route("/symptoms")
def get_symptoms():
    # Convert the unique symptoms data to a list
    symptoms_list = unique_symptoms['symptom'].tolist()
    return jsonify(symptoms_list)



@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    # Debugging: Print received symptoms
    print(f"Received symptoms: {symptoms}")
    
    disease = predict_disease(symptoms)
    
    # Debugging: Print predicted disease
    print(f"Predicted disease: {disease}")
    
    desc, pre, med, die, wrkout = helper(disease)
    result = {
        'disease': disease.capitalize(),
        'description': desc,
        'precautions': list(pre),
        'medications': med,
        'diet': die,
        'workout': wrkout
    }
    
    # Ensure JSON serialization is correct
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


###############################################################


@app.route('/pregnancy', methods=['GET', 'POST'])
def pregnancy():
    if request.method == 'POST':
        data = request.json
        age = data['age']
        diastolicBP = data['diastolicBP']
        BS = data['BS']
        bodyTemp = data['bodyTemp']
        heartRate = data['heartRate']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted_risk = pregnancy_model.predict([[age, diastolicBP, BS, bodyTemp, heartRate]])

        if predicted_risk[0] == 0:
            risk_level = "Low Risk"
            color = "green"
        elif predicted_risk[0] == 1:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"

        result = {
            'risk_level': risk_level,
            'color': color
        }

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return render_template('pregnancy.html')




@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        data = request.json
        sex_dict = {'Male': 0, 'Female': 1}
        sex = sex_dict[data['sex']]
        cp_dict = {'Low pain': 0, 'Mild pain': 1, 'Moderate pain': 2, 'Extreme pain': 3}
        cp = cp_dict[data['cp']]
        fbs_dict = {'Yes': 1, 'No': 0}
        fbs = fbs_dict[data['fbs']]
        exang_dict = {'Yes': 1, 'No': 0}
        exang = exang_dict[data['exang']]
        thal_dict = {
            'Normal (No Thalassemia)': 0,
            'Fixed Defect (Beta-thalassemia minor)': 1,
            'Reversible Defect (Beta-thalassemia intermedia)': 2,
            'Serious Defect (Beta-thalassemia major)': 3
        }
        thal = thal_dict[data['thal']]

        input_data = [
            data['age'],
            sex,
            cp,
            data['trestbps'],
            data['chol'],
            fbs,
            data['restecg'],
            data['thalach'],
            exang,
            data['oldpeak'],
            data['slope'],
            data['ca'],
            thal
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heart_prediction = heart_model.predict([input_data])
        
        if heart_prediction[0] == 1:
            prediction_text = 'The person is having heart disease'
            color = "red"
        else:
            prediction_text = 'The person does not have any heart disease'
            color = "green"

        result = {
            'prediction_text': prediction_text,
            'color': color
        }

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return render_template('heart.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        data = request.json
        input_data = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = diabetic_model.predict([input_data])

        if prediction[0] == 1:
            prediction_text = 'The person is diabetic'
            color = "red"
        else:
            prediction_text = 'The person is not diabetic'
            color = "green"

        result = {
            'prediction_text': prediction_text,
            'color': color
        }

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return render_template('diabetes.html')



if __name__ == '__main__':
    app.run(debug=True)