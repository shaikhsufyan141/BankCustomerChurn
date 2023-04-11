
# NAME @: Sufyan Shaikh 
# TOPIC @: Churn Prediction Model Deployment
# DATE @: 26/03/2023 



# IMPORT THE DEPENDENCIES 

import pickle 
import joblib
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import numpy as np
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField,SelectField,TextField,
                     TextAreaField,SubmitField)






# REMEMBER TO LOAD THE MODEL AND THE SCALER!

# 1. LOAD THE SVC MODEL
pkl_filename = "churn.pkl"
with open(pkl_filename, 'rb') as file2:
    model = pickle.load(file2)

# 2. LOAD THE SCALER OBJECT 
gb_scaler = joblib.load("gb_scaler.pkl")


# 3. CREATE A PREICTION FUNCTION 
def return_prediction(model,scaler,sample_json):
    
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
    
    classes = np.array(['Not a Churn', 'Churn'])
    
    class_ind = model.predict(person)
    
    return classes[class_ind[0]]



# 4 . CREATE A FLASK FORM 

class InfoForm(FlaskForm):
    '''
    This general class gets a lot of form data about user.
    Mainly a way to go through many of the WTForms Fields.
    '''
    

    Name = StringField('Enter Your Full Name', validators=[DataRequired()])

    CreditScore = StringField('Enter Your CreditScore:', validators=[DataRequired()])

    Geography = RadioField('Country:', choices=[('0', 'France'), ('1', 'Germany'),("2","Spain")])

    Gender = RadioField('Please choose your Gender :', choices=[('0', 'Female'), ('1', 'Male')])

    Age = StringField('Please Enter your Age :', validators=[DataRequired()])

    Balance = StringField('Please enter  your Balance amount', validators=[DataRequired()])

    NumOfProducts = StringField('Please enter number of products ',validators=[DataRequired()])

    HasCrCard = RadioField('Do you have a Credit Card ?', choices=[ ('1', 'Yes'), ('0', 'No')])

    IsActiveMember = RadioField('Are you an active member ? :', choices=[('1', 'Yes'), ('0', 'No')])

    EstimatedSalary = StringField('Your Estimated salary :', validators=[DataRequired()])


    feedback = TextAreaField()

    submit = SubmitField('Submit')

