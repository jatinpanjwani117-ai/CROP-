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