
**Name: Atta Ur Rehman**

#*   **Heart Disease Analysis And Prediction**
#*   **Objective: This Project aims to first analyze the dataset & Develop a prediction model for heart disease based on patient data.**

**First Need to import libraries and dataset**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

"""**Upload Dataset to Google Colab**"""

df = pd.read_csv('heart_disease.csv')

"""#**EDA(Exploratory Data Analysis)**"""

df.head()

df.isnull().sum()

df['Age'].fillna(df['Age'].median(), inplace =True)
df['Gender'].fillna('male',inplace = True)
df['Blood Pressure'].fillna(df['Blood Pressure'].median(), inplace =True)
df['Smoking'].replace('Yes',1,inplace = True)
df['Smoking'].replace('No',0,inplace = True)
df['Cholesterol Level'].fillna(df['Cholesterol Level'].median(), inplace =True)
df['Exercise Habits'].fillna('Medium', inplace =True)
df['Smoking'].fillna(df['Smoking'].median(), inplace =True)
df['Family Heart Disease'].replace('Yes',1,inplace = True)
df['Family Heart Disease'].replace('No',0,inplace = True)
df['Family Heart Disease'].fillna(df['Family Heart Disease'].median(), inplace =True)
df['Diabetes'].replace('Yes',1,inplace = True)
df['Diabetes'].replace('No',0,inplace = True)
df['Diabetes'].fillna(df['Diabetes'].median(), inplace =True)
df['BMI'].fillna(df['BMI'].median(), inplace =True)
df['High Blood Pressure'].replace('Yes',1,inplace = True)
df['High Blood Pressure'].replace('No',0,inplace = True)
df['High Blood Pressure'].fillna(df['High Blood Pressure'].median(), inplace =True)
df['Low HDL Cholesterol'].replace('Yes',1,inplace = True)
df['Low HDL Cholesterol'].replace('No',0,inplace = True)
df['Low HDL Cholesterol'].fillna(df['Low HDL Cholesterol'].median(), inplace =True)
df['High LDL Cholesterol'].replace('Yes',1,inplace = True)
df['High LDL Cholesterol'].replace('No',0,inplace = True)
df['High LDL Cholesterol'].fillna(df['High LDL Cholesterol'].median(), inplace =True)
df['Alcohol Consumption'].fillna('Medium', inplace =True)
df['Stress Level'].fillna('Medium', inplace =True)
df['Sleep Hours'].fillna(df['Sleep Hours'].median(), inplace =True)
df['Sugar Consumption'].fillna('Medium', inplace =True)
df['Triglyceride Level'].fillna(df['Triglyceride Level'].median(), inplace =True)
df['Fasting Blood Sugar'].fillna(df['Fasting Blood Sugar'].median(), inplace =True)
df['CRP Level'].fillna(df['CRP Level'].median(), inplace =True)
df['Homocysteine Level'].fillna(df['Homocysteine Level'].median(), inplace =True)
df['Heart Disease Status'].replace('Yes',1,inplace = True)
df['Heart Disease Status'].replace('No',0,inplace = True)

df.info()

df.describe()

"""**How Much of Patients Are Effected by Heart Disease and how much aren't**"""

sns.countplot(x="Heart Disease Status", data=df, palette="Oranges")
plt.title("Heart Disease")
plt.xlabel("Heart Disease Status (0 = No, 1 = Yes)")
plt.ylabel("Cases")
plt.show()

"""**time to see distribution of values**"""

features = ['Age', 'Blood Pressure', 'Diabetes', 'BMI', 'Smoking','Cholesterol Level','Family Heart Disease','High Blood Pressure','Low HDL Cholesterol','High LDL Cholesterol','Sugar Consumption','Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level','Gender', 'Exercise Habits', 'Alcohol Consumption', 'Stress Level', 'Sleep Hours']

for f in features:
    sns.histplot(df[f], bins=20, color='blue')
    plt.title(f"Distribution of {f}")
    plt.xlabel(f)
    plt.ylabel("Frequency")
    plt.show()

"""#**Data Analysis**

**First Need to see which age group has most Heart Diseases**
"""

df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 70, 100],
                           labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"])

sns.countplot(x='age_group', hue='Heart Disease Status', data=df, palette='Oranges')
plt.title("Heart Disease Across Age Groups")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Heart Disease", labels=["No", "Yes"])
plt.show()

"""**People with less then age 30 has Most heart diseases**

**Then need to see Blood Pressure relation with heart Disease, whether increase in Blood Pressure increases heart diseases or not**
"""

plt.figure(figsize=(10, 6))
sns.boxplot(x='Heart Disease Status', y='Blood Pressure', data=df)
plt.title('Blood Pressure Levels in Individuals With and Without Heart Disease')
plt.xlabel('Heart Disease Status')
plt.ylabel('Blood Pressure')

plt.show()

"""**Blood Pressure Doesn't Have Significant effect on heart disease**

**Then we need to see whether Exercise decreases the heart diseases or not.**
"""

sns.countplot(x = 'Exercise Habits',hue ='Heart Disease Status', data=df, palette='Greens')
plt.title("Exercise Effect On Heart Disease")
plt.xlabel("Exercise")
plt.ylabel("Count")
plt.legend(title="Heart Disease", labels=["No", "Yes"])
plt.show()

"""**So Doing more Exercise decrease Heart Disease and We need to do Exercise to prevent Heart Disease.**

**Then need to see Stress Level Relation With Heart Disease**
"""

sns.countplot(x = 'Stress Level',hue ='Heart Disease Status', data=df, palette='Reds')
plt.title("Stress Level And Heart Disease Status")
plt.xlabel("Stress Level")
plt.ylabel("Heart Disease Number")
plt.legend(title="Heart Disease", labels=["No", "Yes"])
plt.show()

"""**So Higher Stress Level Increase Heart Disease And Low Stress Level Decrease heart Disease So A person shouldn't take much Stress.**

**Next, We Need To see Alcohol and Smoking Effect On Heart Disease.**
"""

for i in ['Smoking', 'Alcohol Consumption']:
    sns.barplot(x=i, y='Heart Disease Status', data=df, ci=None)
    plt.title(f"Heart Disease vs {i.capitalize()}")
    plt.xlabel(f"{i.capitalize()} (0 = No, 1 = Yes)")
    plt.ylabel("Average Heart Disease Rate")
    plt.show()

"""**So Smoking and Alcohol Does Increase Heart Disease Chance and A Person Should Prevent These For A better Life.**

**Lets see Diabetes Relation With Heart Disease**
"""

sns.countplot(x = 'Diabetes',hue ='Heart Disease Status', data=df, palette='Reds')
plt.title("Diabetes And Heart Disease")
plt.xlabel("Diabetes 0 = No 1 = Yes")
plt.ylabel("Heart Disease Number")
plt.legend(title="Heart Disease", labels=["No", "Yes"])
plt.show()

"""#**Prediction Model**

**First need to drop extra and non-numerical Columns**
"""

ctd = ['Gender', 'Exercise Habits' ,'Alcohol Consumption','BMI','High Blood Pressure','age_group',	'Stress Level', 'Sugar Consumption']
df1 = df.drop(columns=ctd,axis = 1)

"""**Then take Heart Disease Status As Our Output and All other columns as our input for Prediction Model.**"""

X = df1.drop('Heart Disease Status', axis=1)
y = df1['Heart Disease Status']

"""**The Dataset isn't balanced so need to balance this by Synthetic Minority Over-sampling Technique**"""

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

"""**Then split the dataset into 2 parts with propotion of 80:20 to train and test the model. 80% will be use for training while 20% will be used to testing**"""

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

"""**Then import the random forest model. Use standerscaler to scale the input to make it better for our model. Then used the model and used 100 random forests and fit it. In the end, saw the classificaton report**"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))

"""**Time to ready our input to test the model**"""

new_data = pd.DataFrame({
    'Age': [60],
    'Blood Pressure': [256],
    'Cholesterol Level': [242],
    'Smoking' : [1],
    'Family Heart Disease': [1],
    'Diabetes' : [1],
    'Low HDL Cholesterol': [1],
    'High LDL Cholesterol': [1],
    'Sleep Hours' : [3],
    'Triglyceride Level' : [362],
    'Fasting Blood Sugar' : [254],
    'CRP Level' : [10.3],
    'Homocysteine Level' : [8]


})

"""**Predict the new input we made using the model we trained.**"""

rf_model.predict(new_data)

"""**Time To Make A Beautifull Interface**"""

df1.columns = df1.columns.str.replace(' ', '_')
df1.head()

pip install joblib

import joblib
joblib.dump(rf_model, 'heart_disease_model.pkl')

pip install gradio

import gradio as gr
import joblib

model = joblib.load('heart_disease_model.pkl')


def predict_heart_disease(Age,Blood_Pressure, Cholesterol_Level, Smoking, Family_Heart_Disease, Diabetes,
    Low_Hdl_Cholesterol, High_Ldl_Cholesterol, Sleep_Hours, Triglyceride_Level,
    Fasting_Blood_Sugar, Crp_Level, Homocysteine_Level):
    Smoking = 1 if Smoking == "Yes" else 0
    Family_Heart_Disease = 1 if Family_Heart_Disease == "Yes" else 0
    Diabetes = 1 if Diabetes == "Yes" else 0
    Low_Hdl_Cholesterol = 1 if Low_Hdl_Cholesterol == "Yes" else 0
    High_Ldl_Cholesterol = 1 if High_Ldl_Cholesterol == "Yes" else 0
    features = [Age,Blood_Pressure, Cholesterol_Level, Smoking, Family_Heart_Disease, Diabetes,
    Low_Hdl_Cholesterol, High_Ldl_Cholesterol, Sleep_Hours, Triglyceride_Level,
    Fasting_Blood_Sugar, Crp_Level, Homocysteine_Level]
    prediction = model.predict([features])
    return "High Risk" if prediction[0] == 1 else "Low Risk"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[gr.Number(label="Age"), gr.Number(label="Blood_Pressure"),gr.Number(label="Cholesterol_Level"),gr.Radio(choices=["Yes", "No"], label="Smoking"),gr.Radio(choices=["Yes", "No"], label="Family History of Heart Disease"),gr.Radio(choices=["Yes", "No"], label="Diabetes"),
        gr.Radio(choices=["Yes", "No"], label="Low HDL Cholesterol"),
        gr.Radio(choices=["Yes", "No"], label="High LDL Cholesterol"),gr.Number(label="Sleep_Hours"),gr.Number(label="Triglyceride_Level"),gr.Number(label=
    "Fasting_Blood_Sugar"),gr.Number(label="Crp_Level"),gr.Number(label="Homocysteine_Level")],
    outputs="text",
    title="Heart Disease Prediction Chatbot"
)

interface.launch()

