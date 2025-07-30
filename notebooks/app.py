
import argparse
from flask import Flask, jsonify, request
from flask import render_template
import joblib
import socket
import json
import numpy as np
import pandas as pd
import os

## import model specific functions and variables
from project_setup import *
from data_modelling import *
from logger import *

app = Flask(__name__)


# Predict 
@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    basic predict function for the API
    """
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])
    
    if 'country' not in request.json:
        print("ERROR API (predict): received request, but no 'country' found within")
        return jsonify(False)
        
    if 'year' not in request.json:
        print("ERROR API (predict): received request, but no 'year' found within")
        return jsonify(False)
        
    if 'month' not in request.json:
        print("ERROR API (predict): received request, but no 'month' found within")
        return jsonify(False)
        
    if 'day' not in request.json:
        print("ERROR API (predict): received request, but no 'day' found within")
        return jsonify(False)

        
    ## predict
    _result = result = model_predict(year=request.json['year'],
                                     month=request.json['month'],
                                     day=request.json['day'],
                                     country=request.json['country'],
                                    )
    
    ## convert numpy objects so ensure they are serializable
    result = result.tolist()
    


    return(jsonify(result))

# Train 
@app.route('/train', methods=['GET','POST'])
def train():
    """
    basic train function for the API

    the 'dev' give you the ability to toggle between a DEV version and a PROD verion of training
    """

    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    if 'test' not in request.json:
        print("ERROR API (train): received request, but no 'test' found within")
        return jsonify(False)

    print("... training model")
    model = model_train()
    print("... training complete")

    return(jsonify(True))

# Log
@app.route('/logging', methods=['GET','POST'])
def load_logs():
    """
    basic logging function for the API
    """

    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    if 'env' not in request.json:
        print("ERROR API (log): received request, but no 'env' found within")
        return jsonify(False)
        
        
    if 'month' not in request.json:
        print("ERROR API (log): received request, but no 'month' found within")
        return jsonify(False)
        
    if 'year' not in request.json:
        print("ERROR API (log): received request, but no 'year' found within")
        return jsonify(False)
    
    print("... fetching logfile")
    logfile = log_load(env=request.json['env'],
                       year=request.json['year'],
                       month=request.json['month'])
    
    result = {}
    result["logfile"]=logfile
    return(jsonify(result))



if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=8080)
