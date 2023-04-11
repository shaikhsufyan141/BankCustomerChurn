from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
import pickle
import joblib
import numpy as np
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField, SelectField, TextField,
                     TextAreaField, SubmitField)
from wtforms.validators import DataRequired

app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class InfoForm(FlaskForm):
    '''
    This general class to accept form data
    '''

    Name = StringField('Enter Your Full Name', validators=[DataRequired()])

    CreditScore = StringField('Enter Your CreditScore:', validators=[DataRequired()])

    Geography = RadioField('Country:', choices=[('0', 'France'), ('1', 'Germany'),("2","Spain")])

    Gender = RadioField('Please choose your Gender :', choices=[('0', 'Female'), ('1', 'Male')])

    Age = StringField('Please Enter your Age :', validators=[DataRequired()])

    Balance = StringField('Please enter remaining Loan amount', validators=[DataRequired()])

    NumOfProducts = StringField('Please enter number of products ',validators=[DataRequired()])

    HasCrCard = RadioField('Do you have Credit Card ?', choices=[('0', 'No'), ('1', 'Yes')])

    IsActiveMember = RadioField('Are you an active member? :', choices=[('1', 'Yes'), ('0', 'No')])

    EstimatedSalary = StringField('Your Estimated salary :', validators=[DataRequired()])


    feedback = TextAreaField()

    submit = SubmitField('Submit')


def return_prediction(model, scaler, sample_json):
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

    credscore = sample_json['CreditScore']
    geo = sample_json['Geography']
    gen = sample_json['Gender']
    age = sample_json['Age']
    bal = sample_json['Balance']
    prod = sample_json['NumOfProducts']
    card = sample_json['HasCrCard']
    act = sample_json['IsActiveMember']
    sal = sample_json['EstimatedSalary']

    person = [[credscore,geo,gen,age,bal,prod,card,act,sal]]
    person = scaler.transform(person)
    
    classes = np.array(['Not Churn:- 0', 'Churn:- 1'])
    
    class_ind = model.predict(person)
    
    return classes[class_ind]

# REMEMBER TO LOAD THE MODEL AND THE SCALER!

# LOAD THE SVC MODEL
pkl_filename = "churn.pkl"
with open(pkl_filename, 'rb') as file2:
    churn_model = pickle.load(file2)

# LOAD THE SCLAER OBJECT 
scaler = joblib.load("gb_scaler.pkl")



@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = InfoForm()
    # If the form is valid on submission 
    if form.validate_on_submit():
        # Grab the data from  form.

        session['CreditScore'] = form.CreditScore.data
        session['Geography'] = form.Geography.data
        session['Gender'] = form.Gender.data
        session['Age'] = form.Age.data
        session['Balance'] = form.Balance.data
        session['NumOfProducts'] = form.NumOfProducts.data
        session['HasCrCard'] = form.HasCrCard.data
        session['IsActiveMember'] = form.IsActiveMember.data
        session['EstimatedSalary'] = form.EstimatedSalary.data
     

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    content = {}

    content['CreditScore'] = float(session['CreditScore'])
    content['Geography'] = float(session['Geography'])
    content['Gender'] = float(session['Gender'])
    content['Age'] = float(session['Age'])
    content['Balance'] = float(session['Balance'])
    content['NumOfProducts'] = float(session['NumOfProducts'])
    content['HasCrCard'] = float(session['HasCrCard'])
    content['IsActiveMember'] = float(session['IsActiveMember'])
    content['EstimatedSalary'] = float(session['EstimatedSalary'])
    results = return_prediction(model=churn_model, scaler=scaler, sample_json=content)

    return render_template('thankyou.html', results=results)


# if __name__ == '__main__':
#     app.run('0.0.0.0',port = 8080 , debug=True)


if __name__ == '__main__':
    app.run()
