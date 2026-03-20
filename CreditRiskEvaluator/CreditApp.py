import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train the model
df = pd.read_csv('credit_risk_dataset.csv')
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df['person_home_ownership'] = df['person_home_ownership'].map({'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3})
df['loan_intent'] = df['loan_intent'].map({'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5})
df['loan_grade'] = df['loan_grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'N': 0, 'Y': 1})

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Web app
st.title('CreditRiskEvaluator')
st.write('Enter borrower details to predict loan default risk')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
income = st.number_input('Annual Income', min_value=0, value=50000)
emp_length = st.number_input('Employment Length (years)', min_value=0, value=5)
loan_amount = st.number_input('Loan Amount', min_value=0, value=10000)
int_rate = st.number_input('Interest Rate (%)', min_value=0.0, value=10.0)
percent_income = st.number_input('Loan as % of Income', min_value=0.0, max_value=1.0, value=0.2)
home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
default_on_file = st.selectbox('Previous Default on File', ['N', 'Y'])
cred_hist = st.number_input('Credit History Length (years)', min_value=0, value=3)

home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
intent_map = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_map = {'N': 0, 'Y': 1}

if st.button('Predict'):
    input_data = [[age, income, home_map[home_ownership], emp_length, intent_map[loan_intent], grade_map[loan_grade], loan_amount, int_rate, percent_income, default_map[default_on_file], cred_hist]]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error('High Risk — Likely to Default')
    else:
        st.success('Low Risk — Likely to Repay')