from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(values)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
