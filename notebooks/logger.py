
import time
import os 
import re
import csv
import sys
import uuid
import joblib
from datetime import date

#Load from 
from project_setup import PROJECT_DATA_DIR, LOG_DIR


def _update_train_log(tag, best_model, model_index, mse_score, data_shape, runtime, MODEL_VERSION,\
                     MODEL_VERSION_NOTE, test):
    """
    update train log file
    """

    ## Ensure correct directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join(LOG_DIR,"train-test.log")
    else:
        logfile = os.path.join(LOG_DIR,"train-{}-{}.log".format(today.year, today.month))
        
    ## write the data to a csv file    
    header = ['unique_id','timestamp', 'country','algorithm','mse_score','data_shape','runtime','Model_Version',
              'Model_Version_Note', 'test']
              
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(),tag ,best_model, mse_score, data_shape, runtime,
                            MODEL_VERSION, MODEL_VERSION_NOTE, test])
        writer.writerow(to_write)
        
        
        

def _update_predict_log(tag, y_pred, target_date, MODEL_VERSION, MODEL_VERSION_NOTE):
    """
    Update predict log file
    """
    
    ## Ensure correct directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    ## Name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    logfile = 'predict-{}-{}.log'.format(today.year, today.month)

    
    ## Write the log to a csv file
    logpath = os.path.join(LOG_DIR, logfile)
    
    
    header = ['unique_id', 'timestamp', 'y_pred','target_date' ,'Model_Version', 'Model_Version_Note']
    write_header = False
    if not os.path.exists(logpath):
        write_header = True
    with open(logpath,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.time(), tag,  y_pred,  target_date, 
                            MODEL_VERSION, MODEL_VERSION_NOTE])
        writer.writerow(to_write)

        
def log_load(env,year,month,verbose=True):
    """
    load requested log file
    """
    logfile = "{}-{}-{}.log".format(env,year,month)
    
    if verbose:
        print(logfile)
    return logfile
    
