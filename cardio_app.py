import pandas as pd
import streamlit as st
from pycaret.classification import load_model,predict_model

st.set_page_config(page_title="Cardiovascular disease prediction App")
#@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("cardio_pred")

def predict(model,data):
    prediction = predict_model(model, data=data)
    return prediction['prediction_label'][0]

model = get_model()
st.title("Cardiovascular disease prediction App")

form = st.form('CVD')

Age = form.number_input('Age', min_value=1, max_value=100, value=35)
Gender = form.radio('Gender', ['Male', 'Female'])
Height = form.number_input('Height', min_value=40.0, max_value=250.0, value=164.0)
Weight = form.number_input('Weight', min_value=10.0, max_value=200.0, value=74.0)
Systolic = form.slider('Systolic', min_value=30, max_value=200, value=120)
Diastolic = form.slider('Diastolic', min_value=50, max_value=180, value=80)
Cholesterol_list = ['normal','high','very high']
Cholesterol = form.selectbox('Cholesterol', Cholesterol_list)
Glucose_list = ['normal','high','very high']
Glucose = form.selectbox('Glucose', Glucose_list)
Smoke = form.radio('Smoke',['No','Yes'])
Alcohol = form.radio('Alcohol', ['no','yes'])
Physical_Activity = form.radio('Physical_Activity', ['yes','no'])

predict_button = form.form_submit_button('Predict')

input_dict = {'Age':Age,
              'Gender':Gender,
              'Height':Height,
              'Weight':Weight,
              'Systolic':Systolic,
              'Diastolic':Diastolic,
              'Cholesterol':Cholesterol,
              'Glucose':Glucose,
              'Smoke':Smoke,
              'Alcohol':Alcohol,
              'Physical_Activity':Physical_Activity}

input_df = pd.DataFrame([input_dict])

if predict_button:
    output = predict(model, input_df)
    st.success("Your predicted Cardiovascular disease status is {}".format(output))

