import re
import os
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
secret_key = os.urandom(16).hex()
app.secret_key = secret_key

@app.route("/")
def home():
    return render_template("index.html")

#Recibir archivo
@app.route("/upload", methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if(uploaded_file.filename.endswith('.csv') and uploaded_file.mimetype == "text/csv"):
        session['file_uploaded'] = True
        return render_template("project.html")
    else:
        session['archivo_incorrecto'] = True
        return redirect(url_for('home'))