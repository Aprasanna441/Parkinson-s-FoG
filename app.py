from flask import Flask,render_template,redirect,jsonify
from flask_migrate import Migrate
from  routes.UserRoutes import router
from models.Models import db,User
from flask_login import LoginManager,login_required,current_user

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SelectField,IntegerField,StringField
import joblib
from models.Models import CsvData
import pandas as pd
import __main__
from flask_session import Session
from CustomDecisionTree import DecisionTreeID2
from CustomNaiveBayes import CustomNaiveBayes

import os
from io import BytesIO
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
import seaborn as sns
import base64
import matplotlib.pyplot as plt
setattr(__main__, "DecisionTreeID2", DecisionTreeID2)
setattr(__main__, "CustomNaiveBayes", CustomNaiveBayes) #NB


import joblib
import pandas as pd
import numpy as np







def create_app():
    app = Flask(__name__,template_folder='templates')  # flask app object
   
    app.config.from_object('config')  # Configuring from Python Files
    db.init_app(app)
    
   
    app.register_blueprint(router, url_prefix='/account')

    login_manager = LoginManager()
    login_manager.login_view = 'routes.login_signup' 
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
     return User.query.get(user_id)
    
    migrate = Migrate(app, db)
    
    return app

app = create_app()



#to directly upload
class UploadForm(FlaskForm):
    file=SelectField('Select File')
    mlmodel = SelectField('Select Model', choices=[('NaiveBayes', 'Naive Bayes'), ('DecisionTree', 'Decision Tree')])
    total_data = IntegerField('Data Per Page')
   






@app.route('/',methods=['GET','POST'])
@login_required
def mainpage():
    form = UploadForm()
    form.file.choices = [i[0] for i in db.session.query(CsvData.filename).filter(CsvData.user_id == current_user.id).distinct().all()]

    
    
   
    

    if form.validate_on_submit():
        file=form.file.data
        total_data=form.total_data.data
        
        
        # query  = CsvData.query.filter_by(filename=file).all()
        query = db.session.query(CsvData).filter(CsvData.filename == file).all()
        df = pd.DataFrame([(item.time,item.accv,item.accml,item.accap,item.visit_x,item.age,item.sex,item.years_since_dx,item.updrsiii_on,item.updrsiii_off,item.nfogq,item.medication,item.init,item.completion,item.kinetic) for item in query],
                      columns=["Time","AccV","AccML","AccAP","Visit_x","Age","Sex","YearsSinceDx","UPDRSIII_On","UPDRSIII_Off","NFOGQ","Medication","Init","Completion","Kinetic"])
        df.to_csv('temp.csv', index=False)
        
        
       
        
        
        
        

        
        
        modelDict={
            "DecisionTree":"Custom_decision_tree_30lakh.pkl",
            "NaiveBayes":"random_forest.pkl",
            
        }

        
        selected_model=modelDict[form.mlmodel.data]

        
        
        
       


        truth=pd.read_csv("y_demo.csv")
        image,prediction,accuracy,report=upload_file(selected_model,total_data,truth["Type"][:total_data])
      
       
        
        

       
        return render_template('result.html',  image_path=image,prediction=prediction,truth=truth["Type"],total_data=total_data,selected_model=form.mlmodel.data,input_data=query[:total_data],accuracy=accuracy,report=report)
 
    return render_template('home.html', form=form)







def upload_file(mlmodel,total_data,y_true):
       
        model=joblib.load(mlmodel)
        print(y_true)

        

        data=pd.read_csv("temp.csv")
        independent_variables=np.array(data[:total_data])
        
        
        
       
        if mlmodel=="random_forest.pkl": #naive bayes forest model is from scikitlearn
            
            prediction=model.predict(independent_variables)
            classes=classes=["StartHesitation","Turn","Walking"]
            decoded_classes=[]
            for i in prediction:#changing numerical value to classes again
                class_index = np.argmax(i)
                decoded_class = classes[class_index]
                decoded_classes.append(decoded_class)
        

        else: #rest of the two models are scratch implementation
            decoded_classes = model.predict(data[:total_data])
        
            
            
        

    
       
    





   
  
    


        accuracy = accuracy_score(y_true, decoded_classes)
        
        report = classification_report(y_true, decoded_classes)
        


        cm = confusion_matrix(y_true,decoded_classes)

        # Plot confusion matrix using seaborn
        plt.figure(figsize=(8, 6))
        heatmap=sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
    

        img_buf = BytesIO()
        heatmap.figure.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_path = base64.b64encode(img_buf.read()).decode('utf-8')
        image_path = os.path.join(app.root_path, 'static', 'confusion_matrix.png')
        with open(image_path, 'wb') as f:
            f.write(img_buf.getvalue())

        return  image_path,decoded_classes,accuracy,report
        
        



    
   
    
   
if __name__ == '__main__':  # Running the app
    
    
  
    
    app.run(host='127.0.0.1', port=5000, debug=True)