from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
import pickle
import os

app = Flask(__name__)
model=pickle.load(open(r'rdf.pkl', 'rb'))

@app.route('/') # rendering the html template 
def home():
    return render_template('home.html')

@app.route('/submit',methods=["POST", "GET"])# route to show the predictions in a web UI 
def submit():
    #reading the inputs given by the user 
    input_feature=[int(x) for x in request.form.values() ]
    #input_feature = np. transpose (input_feature)
    input_feature=[np.array(input_feature)]
    print(input_feature)
    names = [ 'Gender','Married','Dependents','Education','Self_Employed','Applicant Income',
    'Coapplicant Income', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History','Credit History','Property Area']
    data = pd.DataFrame(input_feature,columns=names)
    print(data)


    #data scaled scale.fit_transform(data)
    #data = pandas. DataFrame (, columns=names)
    # predictions using the loaded model file
    prediction=model.predict(data)
    print(prediction)
    prediction=int(prediction)
    print(type(prediction))
    if(prediction == 0):
        return render_template("output1 .html", result = "Loan wiil Not be Approved'")
    else:
        return render_template("output2.html",result ="Loan will be Approved")
    # showing the prediction results in a UI
    

if __name__=="__main__":
    # app.run(host= '0.0.0.0', port=8000, debug=True)
    port=int(os.environ.get("PORT " ,5000))
    app.run(debug=False)