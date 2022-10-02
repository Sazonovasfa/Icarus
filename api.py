import json
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Initializing Flask app.
app = Flask(__name__)
CORS(app)
api = Api(app)

# Loading the keras model.
model = keras.models.load_model('probModel')

# Function for transforming data into needed form
def listify(x):
    return [x]

# Api eI will be realndpoint for a model prediction.
class Prediction(Resource):

    def post(self):
        json_data = request.get_json(force=True)
        dscover = json_data['dscover']
        wind = json_data['wind']
        
	pred = model.predict(np.array(map(listify, dscover)), np.array(map(listify, wind)))

        return {'corrected_value': float(pred)}


# Configuring api
api.add_resource(Prediction, '/api')


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
