from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("models/pipe.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":
        try:
            # Get form data
            input_data = [
                int(request.form['pclass']),
                request.form['sex'],
                float(request.form['age']),
                int(request.form['sibsp']),
                int(request.form['parch']),
                float(request.form['fare']),
                request.form['embarked']
            ]
            
            # Make prediction
            final_input = np.array(input_data, dtype=object).reshape(1, -1)
            result = model.predict(final_input)[0]
            
            # Format result
            prediction = "Survived" if result == 1 else "Did not survive"
            
        except Exception as e:
            prediction = "Error in prediction. Please check your inputs."
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)