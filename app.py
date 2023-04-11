

# NAME @: Shaikh Sufyan
# TOPIC @: Churn Prediction Model Deployment
# DATE @: 26/03/2023 


# IMPORT THE DEPENDENCIES 

from flask import (Flask, 
                   render_template, 
                   session, redirect, 
                   url_for, request,
                   jsonify)
 
from helper_functions import (model,
                              gb_scaler,
                              return_prediction,
                              InfoForm)




app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'



# 1. View point to show the form and collect the data from user 
@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = InfoForm()
    
    # If the form is valid on submission 
    if form.validate_on_submit():
        
        # Grab the data from the  form.
        session['Name'] = form.Name.data
        session['CreditScore'] = form.CreditScore.data
        session['Geography'] = form.Geography.data
        session['Gender'] = form.Gender.data
        session['Age'] = form.Age.data
        session['Balance'] = form.Balance.data
        session['NumOfProducts'] = form.NumOfProducts.data
        session['HasCrCard'] = form.HasCrCard.data
        session['IsActiveMember'] = form.IsActiveMember.data
        session['EstimatedSalary'] = form.EstimatedSalary.data
     
        # Redirect the data to Predictioin function
        return redirect(url_for("prediction"))

        # Show the form for first visit 
    return render_template('home.html', form=form)


# 2. View point to show the result 
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
    


     # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] WEB Request  - " , content)


    # Actual prediction done by this function 
    results = return_prediction(model=model, scaler=gb_scaler,sample_json=content)


    # PRINT THE RESULT 
    print("[INFO] WEB Response - " , results)

    return render_template('thankyou.html', results=results)

# 3. View point to handle the restfull api for prediciton 
@app.route('/api/prediction', methods=['POST'])
def predict_churn():
    
    # RECIEVE THE REQUEST 
    content = request.json
    
    # PRINT THE DATA PRESENT IN THE REQUEST 
    print("[INFO] API Request - " , content)
    
    # PREDICT THE CLASS USING HELPER FUNCTION 
    results = return_prediction(model=model,scaler=gb_scaler,sample_json=content)
    
    # PRINT THE RESULT 
    print("[INFO] API Response - " , results)
          
    # SEND THE RESULT AS JSON OBJECT 
    return jsonify(results)




# 4. View Point To handle the 404 Not found Error 
@app.errorhandler(404)
def page_not_found(e):
    return render_template('notfound.html'), 404


# if __name__ == '__main__':
#     app.run('0.0.0.0',8080,debug=False)


if __name__ == '__main__':
    app.run()
