# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:54:47 2021

@author: Pc4y
"""

from flask import Flask
from flask_restful import Api, Resource, reqparse
import zClient
import pickle
import numpy as np
import json
import model_train
from kazoo.client import KazooClient,KazooState


# Creation of the Flask app
APP = Flask(__name__)
API = Api(APP)



zk = zClient.start_kazoo()




# Update the model
class Predict(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('model')
        parser.add_argument('petal_length')
        parser.add_argument('petal_width')
        parser.add_argument('sepal_length')
        parser.add_argument('sepal_width')

        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

        number = int(X_new[0])

        model, node = zClient.predict(number, zk)
        if node!=0:
        
            prediction = model.predict([X_new[1:]])[0]
            
        
            out = {'Prediction': prediction , 'Node': node}

            return out, 200
        else:
            return "Node not found!"



class Put(Resource):
    @staticmethod
    def get():

        model = pickle.dumps(pickle.load( open( "iris_naive.pkl", "rb" ) ))
        number = 2
        out = zClient.put_model(number, model, zk)
        out = pickle.loads(out)
        for k,i in out.items():
            out[k] = str(pickle.loads(i))
        #print(out)
        out = {'Models': out}
        return out, 200






        

API.add_resource(Predict, '/predict')
API.add_resource(Put, '/put')


if __name__ == '__main__':
    APP.run(debug=True, host='127.0.0.1', port='8000')





