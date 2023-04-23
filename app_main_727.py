
import streamlit as st 
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


model = pickle.load(open('model.heart.sav','rb'))

Sex_encoder = pickle.load(open('encoder.Sex.sav','rb'))
ChestPainType_encoder = pickle.load(open('encoder.ChestPainType.sav','rb'))
RestingECG_encoder = pickle.load(open('encoder.RestingECG.sav','rb'))
ExerciseAngina_encoder = pickle.load(open('encoder.ExerciseAngina.sav','rb'))
ST_Slope_encoder = pickle.load(open('encoder.ST_Slope.sav','rb'))

evaluations = pickle.load(open('evals.all.sav','rb'))

st.title('Heart Disease Prection')

tab1, tab2 = st.tabs(["Prediction", "Model Evaluations"])


with tab1:
    
    #'Age'
    x1 = st.slider('Age', 0, 100, 30)

    #'Sex'
    x2 = st.radio('Select Sex', Sex_encoder.classes_)
    x2 = Sex_encoder.transform([x2])[0]

    #'ChestPainType'
    x3 = st.radio('Select ChestPainType', ChestPainType_encoder.classes_)
    x3 = ChestPainType_encoder.transform([x3])[0]

    #'RestingBP'
    x4 = st.slider('RestingBP', 0, 210, 130)

    #'Cholesterol'
    x5 = st.slider('Cholesterol', 0, 700, 180)

    #'FastingBS'
    x6 = st.slider('FastingBS', 0, 1, 0)

    #'RestingECG'
    x7 = st.radio('Select RestingECG', RestingECG_encoder.classes_)
    x7 = RestingECG_encoder.transform([x7])[0]

    #'MaxHR'
    x8 = float(input('Enter MaxHR'))

    #'ExerciseAngina'
    x9 = st.radio('Select ExerciseAngina', ExerciseAngina_encoder.classes_)
    x9 = ExerciseAngina_encoder.transform([x9])[0]

    #'Oldpeak'
    x10 = float(input('Enter Oldpeak'))

    #'ST_Slope'
    x11 = st.radio('Select ST_Slope', ST_Slope_encoder.classes_)
    x11 = ST_Slope_encoder.transform([x11])[0]

    x_new = pd.DataFrame(data=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]).reshape(1,-1), 
                 columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                           'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

    pred = model.predict(x_new)

    st.header('Predicted Result: ' , )

    if pred ==0 : 
        st.subheader('Normal')
    else :
        st.subheader('Heart Disease')

    #HeartDisease: output class [1: heart disease, 0: Normal]
    
with tab2:
    import plotly.graph_objects as px
    
    evaluations = pickle.load(open('evals.all.sav','rb'))

    x = evaluations.columns
    fig = px.Figure(data=[
        px.Bar(name = 'Decision Tree',
               x = x,
               y = evaluations.loc['Decision Tress']),
        px.Bar(name = 'Random Forest',
               x = x,
               y =  evaluations.loc['Random Forest']),
        px.Bar(name = 'KNN',
               x = x,
               y =  evaluations.loc['KNN']),
        px.Bar(name = 'AdaBoost',
               x = x,
               y =  evaluations.loc['AdaBoost']),
        px.Bar(name = 'XGBoost',
               x = x,
               y =  evaluations.loc['XGBoost']),
        
    ])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(evaluations)
    
