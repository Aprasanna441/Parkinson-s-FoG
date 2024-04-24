from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin,db.Model):

    __tablename__ = 'User'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(80), unique=True, nullable=False)
    joined_on=db.Column(db.DateTime,default=datetime.utcnow,nullable=False)
    is_admin=db.Column(db.Boolean,default=False)
    is_active=db.Column(db.Boolean,default=True)

    password = db.Column(db.String(120), nullable=False)
   


    def __repr__(self):
        return self.email
    
# user_id,time,accv,accml,accap,visit_x,age,sex,years_since_dx,updrsiii_on,updrsiii_off,nfogq,medication,init,completion,kinetic 

class CsvData(db.Model):
    id = db.Column(db.Integer, primary_key=True ,unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey('User.id'), nullable=False)
    filename = db.Column(db.String(255))
    time = db.Column(db.String(255))
    accv = db.Column(db.Float)
    accml = db.Column(db.Float)
    accap = db.Column(db.Float)
    visit_x = db.Column(db.String(255))
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    years_since_dx = db.Column(db.Integer)
    updrsiii_on = db.Column(db.Float)
    updrsiii_off = db.Column(db.Float)
    nfogq = db.Column(db.Integer)
    medication = db.Column(db.String(255))
    init = db.Column(db.String(255))
    completion = db.Column(db.String(255))
    kinetic = db.Column(db.String(255))






        


    


    
    

