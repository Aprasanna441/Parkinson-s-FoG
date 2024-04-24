from flask import render_template,redirect,request,flash,url_for,session
# from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import  login_user, login_required, logout_user,current_user
import csv

from flask_bcrypt import Bcrypt
from models.Models import User,db  
from flask_session import Session


from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import IntegerField

from models.Models import CsvData

bcrypt=Bcrypt()

def hash_password(password):
    pass 

def check_password():
    pass








@login_required
def home():
   
 

    return render_template('home.html')
    






def login_signup():
    return render_template('authentication.html')

def signup():
      if request.method=='POST':
        email=request.form.get("email")
        password=request.form.get("password")
        password_confirm=request.form.get("password_confirm")
        user = User.query.filter_by(email=email).first()

        if  not   user:
            if password ==password_confirm:
                password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
                user=User(password=password_hash,email=email)
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return redirect("/")

            else:
                flash("Password and Confirm Password doesnt match",'error') 
                return redirect(url_for('routes.login_signup'))    
        else:
            flash("User already exist.Please Proceed to Login",'error')
            return redirect(url_for('routes.login_signup'))


def login():
    email=request.form.get("email")
    passw =request.form.get("password")
    user=User.query.filter_by(email=email).first()
    
  

    if user: 
      
        if  bcrypt.check_password_hash(user.password, passw ):
             
             login_user(user)           
             
             return redirect('/')
        else:
             print("didnt hash")
             flash("Invalid Password",'error')
             return redirect(url_for('routes.login_signup'))
    else:
        flash("User doesnt exist",'error')
        return redirect(url_for('routes.login_signup'))
    
    


def logout():
    # logout_user()
    session.pop('email',None)
    logout_user()
    return redirect(url_for('routes.login_signup'))


#nice

#to save in database
class CsvUploadForm(FlaskForm):
    csv_file = FileField('Upload CSV', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV files only!')
    ])
    user_id= IntegerField('User Id')


import pandas as pd
@login_required
def dashboard():
    form=CsvUploadForm()
    if form.validate_on_submit():
        file = request.files['csv_file']
        user_id=form.user_id.data
        df = pd.read_csv(file)

        for index, row in df.iterrows():
            new_data = CsvData(filename=file.filename,user_id=user_id,time=row['Time'],accv=row['AccV'],accml=row['AccML'],accap=row['AccAP'],visit_x=row['Visit_x'],age=row['Age'],sex=row['Sex'],years_since_dx=row['YearsSinceDx'],updrsiii_on=row['UPDRSIII_On'],updrsiii_off=row['UPDRSIII_Off'],nfogq=row['NFOGQ'],medication=row['Medication'],init=row['Init'],completion=row['Completion'],kinetic=row['Kinetic'] )
            db.session.add(new_data)
        db.session.commit()
        redirect('/home')        

        # Example: Saving to a hypothetical database
        

    return render_template('dashboard.html', form=form)










