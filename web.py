import numpy as np
import pickle
import streamlit as st

#loading the saved model
load_model = pickle.load(open('C:/Users/Hemant/jupyter_codes/ML Project 1/Diabetes detection/Trained_model.sav', 'rb'))

# creating a function for prediction
def diabetes_prediction(input_data):

    #convert the input to numpy array
    input_array = np.asarray(input_data)
    
    #reshape the array as we are predicting only on one instance
    reshaped_array = input_array.reshape(1, -1)
    
    #prediction
    prediction = load_model.predict(reshaped_array)
    print(prediction)
    if prediction == 0:
        return 'THE PERSON IS NON-DIABETIC'
    else:
        return 'THE PERSON IS DIABETIC'


def main():
    
    # giving a title
    st.title('DIABETES PREDICTION WEB APP')
    
    # getting the input data
    
    Pregnancies = st.text_input('No. of Pregnancies :')
    Glucose = st.text_input('Glucose level :')
    BloodPressur = st.text_input('Blood Pressure level :')
    SkinThickness = st.text_input('Skin Thickness value :')
    Insulin = st.text_input('Insulin level :')
    BMI = st.text_input('Body Mass Index value :')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value :')
    Age = st.text_input('Age :')
    
    input_data = [Pregnancies, Glucose, BloodPressur, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(input_data = input_data)
        
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    