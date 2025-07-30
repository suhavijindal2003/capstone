

#Standard Imports
import pandas as pd
import numpy as np
import os
import time
import joblib
import re

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Modelling
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error as mse


#Load from 
from project_setup import PROJECT_DATA_DIR, DATA_DIR, TS_DIR, LOG_DIR, MODEL_DIR

from data_ingestion import load_ts, engineer_features
from logger import _update_predict_log, _update_train_log

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = '-'


def _model_train(dataset,tag,test = False):
    """
    Train models and select the best one out of DecisionTreeRegression,  GradientBoostingRegression, AdaBoostRegression
    and XGBoostRegressor. Feed the model the timeseries_datasets.
    """
    
    ## start timer for runtime
    time_start = time.time()
    
    dataset = engineer_features(dataset, training = True)
    
    X = dataset.drop(['target','dates'], axis = 1)
    y = dataset.target
    
    #Train_Test_Split Data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 0)
    
    
    ##Train Models
    
    GridSearchParameters = {'criterion': ['mse', 'mae', 'friedman_mse'],
                            'max_depth': [None, 10,20,50],
                            'max_features': ['auto', 'sqrt', 'log2']}, \
    {'criterion': ['mse', 'mae'],
     'max_features' : ['auto', 'sqrt'] }, \
    {'loss' : ['ls', 'lad', 'huber', 'quantile'],
     'learning_rate' : [0.1,0.01,0.001]}, \
    {'loss' : ['linear', 'square',],
     'learning_rate' : [0.05, 0.1, 0.01]}, \
    {'learning_rate': [0.05, 0.1, 0.01],
     'max_depth': [1, 5, 50],
     'n_estimators': [100, 1000, 500]
    }

    params = {
        'DTR_P' : GridSearchParameters[0],
        'RFR_P' : GridSearchParameters[1],
        'GBR_P' : GridSearchParameters[2],
        'ADA_P' : GridSearchParameters[3],
        'XGB_P' : GridSearchParameters[4],
    }
    
    regressor_dict = {
        'DTR' : DecisionTreeRegressor(random_state = 42),
        'RFR' : RandomForestRegressor(random_state = 42),
        'GBR' : GradientBoostingRegressor(random_state = 42),
        'ADA' : AdaBoostRegressor(random_state = 42),
        'XGB' : xgb.XGBRegressor(seed = 42)

    }
    

    
    models = {}
    
    for model_name in regressor_dict:
        
        pipe = Pipeline(steps = [('scaler', StandardScaler()),
                                ('regressor', regressor_dict[model_name])])
        grid = GridSearchCV(regressor_dict[model_name],
                           param_grid = params[model_name + '_P'], cv = 5)
        grid.fit(X_train, y_train)
        
        models[model_name] = grid
        
     
    model_scores = []
    
    #Test which model is optimal.
    for model in models:
        y_pred = models[model].predict(X_test)
        rmse = np.sqrt(mse(y_pred, y_test))
        model_scores.append(rmse)
    
    model_index = np.argmin(model_scores)
    model_score = min(model_scores)
    model_name = list(models.keys())[model_index]
    best_model =  list(models.values())[model_index]
    
    print(f'The best model for {tag} is {model_name}.')
   
    
    #Retrain on best model.
    best_model.fit(X,y)
    
    #Save model.
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    if test:
        saved_model = os.path.join(MODEL_DIR, f'test-{tag}-{model_name}.joblib')
    else:
        saved_model = os.path.join(MODEL_DIR, f'sl-{tag}-{model_name}.joblib')
        
    
    joblib.dump(best_model,saved_model)
    
    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)
    
    
    # Update Train Log.
    _update_train_log(tag, best_model, model_index, model_score, dataset.shape, runtime, MODEL_VERSION,
                     MODEL_VERSION_NOTE,test)
    
    
    
    
def model_train(ts_dir = TS_DIR, test = False):
    """
    Train the models for each of the top ten countries (+ all).
    """
    #Check Directories
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    
    #Load ts files
    ts = load_ts(ts_dir)
    
    for country,df in ts.items():
        print(f'...training model for {country}')
        _model_train(df, country, test = test)

        
def model_load(prefix = 'sl',ts_dir = TS_DIR, model_dir = MODEL_DIR):
    """
    Function to load in Train Models
    """
    
    model_list = [file for file in os.listdir(model_dir) if file[0:len(prefix)] == prefix]
    
    if len(model_list) == 0:
        raise Exception(f'No models found with prefix: {prefix}. Did you train them?')
        
    models = {}
    print('...Loading models')
    for model in model_list:
        models[re.split('-',model)[1]] = joblib.load(os.path.join(model_dir,model))
        

        
    return models

    
def model_predict(year, month, day, country = 'all'):
    """
    Make predictions based on a country.
    """
    
    ## Timer
    time_start = time.time()
    
    ## Load all data
    ts = load_ts()
    eng_datasets = {country: engineer_features(ts[country], training = False) for country in ts.keys()}
    
    
    ## Load all models
    models = model_load()
    
    
    ## check if model for country 
    if country not in models.keys():
        raise Exception(f'ERROR: (model_predict) for country {country} is unavailable.')
        
        
    ## Check if dataset is available
    if country not in eng_datasets.keys():
        raise Exception(f'ERROR: (dataset) for country {country} is unavailable.')
    
 
    ## Load data and model for country
    model = models[country]
    eng_dataset = eng_datasets[country]
    
    
    ## Check date
    target_date = f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)}'
    print(target_date)
    
    ## Data to predict on:
    X_pred = eng_dataset[eng_dataset['dates'] == target_date].drop(['target','dates'], axis =1 )
    
    
    
    # Prediction
    y_pred = model.predict(X_pred)
    
    _update_predict_log(tag = country, y_pred = y_pred,  target_date = target_date,\
                        MODEL_VERSION = MODEL_VERSION, MODEL_VERSION_NOTE = MODEL_VERSION_NOTE)
    
    return(y_pred)       
    

if __name__ == "__main__":

    """
    basic test procedure for model.py
    """
    
    run_start = time.time()

    ## train the models if directory not exists
    if not os.path.exists(MODEL_DIR):
        print("...Training Models")
        model_train(TS_DIR)

    ## load the model
    print("...Loading Models")
    models = model_load()
    print("...models loaded: ",",".join(models.keys()))

    ## test predict
    country='all'
    year='2018'
    month='1'
    day='13'
    result = model_predict(year = year, month = month, day = day, country = country)
    
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...running time:", "%d:%02d:%02d"%(h, m, s))
    
    
    print(result)
