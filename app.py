from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

FEATURES = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

@app.route('/')
def index():
    return render_template("index.html", features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        vals = [float(request.form[feat]) for feat in FEATURES]
        arr = np.array(vals).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        pred = model.predict(arr_scaled)[0]
        prob = model.predict_proba(arr_scaled)[0]

        if pred == 0:
            label = 'Malignant (Positive)'
            confidence = round(prob[0] * 100, 2)
            emoji = "⚠️"
        else:
            label = 'Benign (Negative)'
            confidence = round(prob[1] * 100, 2)
            emoji = "✅"

        return render_template("result.html",
                               features=FEATURES,
                               label=label,
                               confidence=confidence,
                               emoji=emoji,
                               error=None)

    except Exception as e:

        return render_template("result.html",
                               features=FEATURES,
                               label=None,
                               confidence=None,
                               emoji=None,
                               error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
