import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
model = load_model("healthcare_diagnosis_model.h5")
le = pickle.load(open("label_encoder.pkl", "rb"))
st.title("AI-Powered Smart Healthcare Diagnosis System")
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
specialist_dict = {

    "Fungal infection": "Dermatologist",

    "Allergy": "Allergist",

    "GERD": "Gastroenterologist",

    "Chronic cholestasis": "Hepatologist",

    "Drug Reaction": "General Physician",

    "Peptic ulcer diseae": "Gastroenterologist",

    "AIDS": "Infectious Disease Specialist",

    "Diabetes ": "Endocrinologist",

    "Bronchial Asthma": "Pulmonologist"
}
def recommend_specialist(disease):

    return specialist_dict.get(disease, "General Physician")


def predict_risk(prediction):

    confidence = prediction.max()

    if confidence > 0.8:
        return "High Risk"

    elif confidence > 0.5:
        return "Moderate Risk"

    else:
        return "Low Risk"
selected_symptoms = st.multiselect(
    "Select Your Symptoms",
    symptoms
)
if st.button("Predict Disease"):

    input_data = np.zeros(len(symptoms))

    for symptom in selected_symptoms:
        index = symptoms.index(symptom)
        input_data[index] = 1

    input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)

    predicted_class = np.argmax(prediction, axis=1)

    disease = le.inverse_transform(predicted_class)[0]

    doctor = recommend_specialist(disease)

    risk = predict_risk(prediction)

    st.success(f"Predicted Disease: {disease}")

    st.info(f"Recommended Specialist: {doctor}")

    st.warning(f"Risk Level: {risk}")