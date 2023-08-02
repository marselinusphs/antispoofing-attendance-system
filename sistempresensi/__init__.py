from flask import Flask
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import joblib
from sklearn.preprocessing import StandardScaler


global_variables = {
    'semester': 2,
    'tahun_ajaran': "20222023",
    'hari': ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"],
    'logger': "",
    'bulan': ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
              "November", "Desember"]
}

try:
    df = pd.read_csv('sistempresensi/static/lbp_1_glcm24.csv')
    df = df.drop('Unnamed: 0', axis=1)
    X = df.drop('label', axis=1)
    scaler = StandardScaler().fit(X)
    model = joblib.load('sistempresensi/static/model.pkl')
except:
    df = pd.read_csv('./static/lbp_1_glcm24.csv')
    df = df.drop('Unnamed: 0', axis=1)
    X = df.drop('label', axis=1)
    scaler = StandardScaler().fit(X)
    model = joblib.load('./static/model.pkl')

app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.url_map.strict_slashes = False
db = SQLAlchemy(app)

from sistempresensi import routes
