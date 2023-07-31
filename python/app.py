# app.py

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained RandomForestClassifier model using pickle
model = pickle.load(open('rdf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read user inputs from the form in home.html
    gender = int(request.form['Gender'])
    married = int(request.form['Married'])
    dependents = int(request.form['Dependents'])
    education = int(request.form['Education'])
    self_employed = int(request.form['Self_Employed'])
    applicant_income = int(request.form['ApplicantIncome'])
    coapplicant_income = int(request.form['CoapplicantIncome'])
    loan_amount = int(request.form['LoanAmount'])
    loan_amount_term = int(request.form['Loan_Amount_Term'])
    credit_history = int(request.form['Credit_History'])
    property_area = int(request.form['Property_Area'])
    
    # Create a DataFrame with user inputs
    user_df = pd.DataFrame([[gender, married, dependents, education, self_employed,
                             applicant_income, coapplicant_income, loan_amount,
                             loan_amount_term, credit_history, property_area]],
                           columns=['Gender', 'Married', 'Dependents', 'Education',
                                    'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                    'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    
    # Make prediction using the loaded model
    prediction = model.predict(user_df)
    
    if prediction[0] == 0:
        result = "Loan will Not be Approved"
    else:
        result = "Loan will be Approved"
    
    return render_template('output.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
