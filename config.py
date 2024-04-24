import os

from dotenv import load_dotenv
load_dotenv()
SECRET_KEY = os.getenv("MY_SECRET")

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))

# Enable debug mode.
DEBUG = True
SECRET_KEY = 'namastenapla'


SESSION_TYPE = 'filesystem'

# Connect to the database
SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

# Turn off the Flask-SQLAlchemy event system and warning
SQLALCHEMY_TRACK_MODIFICATIONS = False