import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
from sklearn.compose import ColumnTransformer

import streamlit as st


def predict(testing_arr):

    with st.spinner('Wait for it...'):
            #Import the Cardiovascular Disease dataset
        data = pd.read_csv('./archive/cardio_train.csv', sep=";")

        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            data = data.drop_duplicates() #remove duplicates if applicable


        #data transformation, removing outliers, and interesting findings
        data['years'] = (data['age'] / 365).round().astype('int') #in order to have a more appropriate format for age
        data['cardio'].value_counts(normalize=True) #to balance the target variable
        data['bmi'] = data['weight'] / (data['height']/100)**2 #include the BMI variable
        data.drop('id', axis=1, inplace=True) #id column is not vital for modeling
        data.drop('age', axis=1, inplace=True) #not needed after calculating the age


        data.drop(data[(data['height'] > data['height'].quantile(0.975)) | (data['height'] < data['height'].quantile(0.025))].index,inplace=True) #remove outliers => remove heights that fall below 2.5% or above 97.5% of a given range
        data.drop(data[(data['weight'] > data['weight'].quantile(0.975)) | (data['weight'] < data['weight'].quantile(0.025))].index,inplace=True) #remove outliers => remove weights that fall below 2.5% or above 97.5% of a given range
        data.drop(data[(data['ap_hi'] > data['ap_hi'].quantile(0.975)) | (data['ap_hi'] < data['ap_hi'].quantile(0.025))].index,inplace=True) #remove outliers => remove systolic blood pressure values that fall below 2.5% or above 97.5% of a given range
        data.drop(data[(data['ap_lo'] > data['ap_lo'].quantile(0.975)) | (data['ap_lo'] < data['ap_lo'].quantile(0.025))].index,inplace=True) #remove outliers => remove diastolic blood pressure values that fall below 2.5% or above 97.5% of a given range


        data.groupby('gender')['height'].mean() #on average, men are taller than women
        data.groupby('gender')['alco'].sum() #on average, men drink more than women

        #Features and Target
        X = data[['years', 'height', 'weight', 'ap_hi', 'ap_lo', 'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']]
        y = data['cardio']

        #Ensure that y is a 1D array
        y = y.values.ravel()

        #Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

        #Initialize the Random Forest Classifier
        #RFmodel = RandomForestClassifier(n_estimators = 100, criterion='entropy', max_depth=10, random_state = 42)
        #RFmodel = RandomForestClassifier(n_estimators = 150, criterion='entropy',  max_depth=20, random_state = 42)
        RFmodel = RandomForestClassifier(n_estimators=150, max_depth=6, max_features=None, random_state=42)

        #Fit the model
        RFmodel.fit(X_train, y_train)

        test_data = RFmodel.predict(testing_arr)

    if(test_data[0] == 0):
        st.success("You are safe. But you can always be healthier :)")
    else:
        st.warning("You are at risk of having Cardio Vascular Disease!")

st.set_page_config(page_title="CVD Predictor")
st.markdown("# CVD Predcitor")

years = st.slider("Enter age", min_value=0, max_value=100)
height = st.number_input("Enter height (in cm)",  min_value=120, max_value=200)
weight = st.number_input("Enter weight (in kg)",  min_value=30, max_value=200)
gender = st.selectbox(
    'Select gender',
    ('Male', 'Female'),
    index=None,
    placeholder="Select gender...",    
)
cholesterol = st.selectbox(
    'Select cholesterol level',
    ('normal', 'above normal', 'well above normal'),
    index=None,
    placeholder="Select cholesterol level...",    
)
glucose = st.selectbox(
    'Select glucose level',
    ('normal', 'above normal', 'well above normal'),
    index=None,
    placeholder="Select glucose level...",    
)
sys_bp = st.slider("Enter systolic bp", min_value=90, max_value=200)
dia_bp = st.slider("Enter diastolic bp", min_value=35, max_value=110)
smoke = st.toggle('Do you smoke?')
alc = st.toggle('Do you consume alcohol?')
active = st.toggle('Are you active?')

gender_var = 1
cholesterol_var = 1
glucose_var = 1
smoke_var = 0
alc_var = 0
active_var = 0

if gender == 'Male':
    gender_var = 1
else:
    gender_var = 2

if cholesterol == "normal":
    cholesterol_var = 1
elif cholesterol == "above normal":
    cholesterol_var = 2
else:
    cholesterol_var = 3

if glucose == "normal":
    glucose_var = 1
elif glucose == "above normal":
    glucose_var = 2
else:
    glucose_var = 3

if smoke:
    smoke_var = 1
else:
    smoke_var = 0

if alc:
    alc_var = 1
else:
    alc_var = 0

if active:
    active_var = 1
else:
    active_var = 0

bmi = weight /(( height/100)**2)

arr = [[years, height, weight, sys_bp, dia_bp, gender_var, cholesterol_var, glucose_var, smoke_var, alc_var, active_var, bmi]]

predcit_btn = st.button("Predict", type="primary", on_click=predict, kwargs={"testing_arr": arr})