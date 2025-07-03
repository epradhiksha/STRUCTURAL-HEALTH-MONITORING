from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("bridge_health_model1.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        VOLTAGE = float(request.form['VOLTAGE'])
        st355 = float(request.form['ST355'])
        st356 = float(request.form['ST356'])
        st348 = float(request.form['ST348'])

        input_data = np.array([[VOLTAGE,st355, st356, st348]])
        prediction = model.predict(input_data)[0]

        conditions = ["Healthy", "Moderate", "Critical"]
        result = conditions[prediction]

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
