from flask import Flask, request
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('model.pkl', 'rb'))
@app.route("/", methods=['GET','POST'])
def display():
    if request.method == "GET":
        return "hi"
    elif request.method == "POST":
        format = request.json
        features = [float(x) for x in format.values()]
        final = np.array(features)
        input_data_reshaped = final.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        # output = '{0:.{1}f}'.format(prediction[0][1], 2)
        output = prediction[0]
        
        if output == 1:
            return "The patient is diabetic"
        else:
            return "The patient is non-diabetic"
    else:
        return "oops"

if __name__ == "__main__":
    app.run(debug= True, port=5000)