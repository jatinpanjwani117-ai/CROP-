<<<<<<< HEAD
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model & encoder once
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([[ 
        float(data["N"]), float(data["P"]), float(data["K"]), 
        float(data["temperature"]), float(data["humidity"]), 
        float(data["ph"]), float(data["rainfall"])
    ]])
    prediction = model.predict(features)
    crop = label_encoder.inverse_transform(prediction)[0]
    return jsonify({"crop": crop})

if __name__ == "__main__":
    app.run(debug=True)
=======
from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key in production

# Load dataset and train model once at startup
df = pd.read_csv('Crop_recommendation.csv')
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Input limits from dataset
input_limits = {
    'N': {'min': 0, 'max': 140},
    'P': {'min': 5, 'max': 145},
    'K': {'min': 5, 'max': 205},
    'temperature': {'min': 8.83, 'max': 43.68},
    'humidity': {'min': 14.26, 'max': 99.98},
    'ph': {'min': 3.5, 'max': 9.94},
    'rainfall': {'min': 20.21, 'max': 298.56}
}

# Simple in-memory user storage for prototype
users_db = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    prediction = None
    error = None
    if 'logged_in' not in session:
        return redirect(url_for('login_page'))

    if request.method == 'POST':
        try:
            user_input = []
            for f in features:
                val = float(request.form[f])
                limits = input_limits[f]
                if not (limits['min'] <= val <= limits['max']):
                    error = f"{f.title()} must be between {limits['min']} and {limits['max']}."
                    break
                user_input.append(val)
            if not error:
                prediction = model.predict([user_input])[0]
        except Exception as e:
            error = "Invalid input. Please check your entries."

    return render_template('recommendation.html', limits=input_limits, prediction=prediction, error=error)

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login_page'))
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile')
def profile():
    if 'logged_in' not in session:
        return redirect(url_for('login_page'))
    return render_template('profile.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users_db and check_password_hash(users_db[username], password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        error = "Invalid username or password."
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users_db:
            error = "Username already exists."
        else:
            users_db[username] = generate_password_hash(password)
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 0dd934d (final commit)
