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

#Import the Cardiovascular Disease dataset
data = pd.read_csv('./archive/cardio_train.csv', sep=";")


#Quick overview of the data and important stats
data.head() #to visualize the first 5 rows of the dataset
data.info() #to check the data type for each variable; all features are numerical, 11 of integer type and 1 of decimal type
data.describe() #to identify important statistics for each variable (e.g. mean, standard deviation, min, max, etc.); one can spot the outliers in height, weight, ap_hi, api_lo


#Are there null values?What about duplicates?
print(data.isnull().sum()) #check for null values

print(f"Number of duplicate rows: {data.duplicated().sum()}") #display the number of duplicate values
duplicate_count = data.duplicated().sum()
if duplicate_count > 0:
    data = data.drop_duplicates() #remove duplicates if applicable


#data transformation, removing outliers, and interesting findings
data['years'] = (data['age'] / 365).round().astype('int') #in order to have a more appropriate format for age
data['cardio'].value_counts(normalize=True) #to balance the target variable
data['bmi'] = data['weight'] / (data['height']/100)**2 #include the BMI variable
data.drop('id', axis=1, inplace=True) #id column is not vital for modeling
data.drop('age', axis=1, inplace=True) #not needed after calculating the age
data.describe()
data.info()
len(data)


data.drop(data[(data['height'] > data['height'].quantile(0.975)) | (data['height'] < data['height'].quantile(0.025))].index,inplace=True) #remove outliers => remove heights that fall below 2.5% or above 97.5% of a given range
data.drop(data[(data['weight'] > data['weight'].quantile(0.975)) | (data['weight'] < data['weight'].quantile(0.025))].index,inplace=True) #remove outliers => remove weights that fall below 2.5% or above 97.5% of a given range
data.drop(data[(data['ap_hi'] > data['ap_hi'].quantile(0.975)) | (data['ap_hi'] < data['ap_hi'].quantile(0.025))].index,inplace=True) #remove outliers => remove systolic blood pressure values that fall below 2.5% or above 97.5% of a given range
data.drop(data[(data['ap_lo'] > data['ap_lo'].quantile(0.975)) | (data['ap_lo'] < data['ap_lo'].quantile(0.025))].index,inplace=True) #remove outliers => remove diastolic blood pressure values that fall below 2.5% or above 97.5% of a given range
len(data)

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

