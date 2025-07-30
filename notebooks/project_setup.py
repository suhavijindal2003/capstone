
import os
import time

# set the project root and its structure
PROJECT_ROOT_DIR = '..'
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'capstone')

# point to the origin of data
DATA_DIR = os.path.join(PROJECT_DATA_DIR,'cs-train')

# where should all created data be stored
PROJECT_RUNTIME_DIR = os.path.join('runtime')

# set the training data dir
TS_DIR = os.path.join(PROJECT_RUNTIME_DIR, 'ts-data')

# set imaging dir
IMAGE_DIR = os.path.join(PROJECT_RUNTIME_DIR, 'images')

# set the logging dir
LOG_DIR = os.path.join(PROJECT_RUNTIME_DIR, 'logs')

# set the model dir
MODEL_DIR = os.path.join(PROJECT_RUNTIME_DIR, 'models')

# set the unit-test dir
UNITTEST_DIR = os.path.join(PROJECT_RUNTIME_DIR, 'unittests')

# set the docker dir
DOCKER_DIR = os.path.join(PROJECT_RUNTIME_DIR, 'dockerenv')

# set test 
TEST = False

# Main Funciton 
if __name__ == "__main__":
    
    run_start = time.time()
    print ("setting up project environment")
  
    # create path if needed
    if not os.path.exists(PROJECT_RUNTIME_DIR):
        os.mkdir(os.path.join(PROJECT_RUNTIME_DIR))
            
    ## 
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("...run time:", "%d:%02d:%02d"%(h, m, s))
    
    print("done")
