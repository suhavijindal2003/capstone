#!/usr/bin/env python

"""
model tests
"""

import unittest
from project_setup import PROJECT_DATA_DIR, UNITTEST_DIR, MODEL_DIR, TEST, TS_DIR

from data_modelling import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
    
    def test_01_train(self):
        """
        test the train functionality
        """
        
        print("Test: Model-Train")
        
        ## train the model
        model_train(TS_DIR, TEST)
        
        prefix = 'test' if TEST else 'sl'
        models = [f for f in os.listdir(MODEL_DIR) if re.search(prefix,f)]
        self.assertEqual(len(models),11)
        
    def test_02_load(self):
        """
        test the train functionality
        """
        print("Test: Model-Load")
       
        ## load the model
        models = model_load()
        
        for tag, model in models.items():
            self.assertTrue("predict" in dir(model))
            self.assertTrue("fit" in dir(model))
        
        
    def test_03_predict(self):
        """
        test the predict function input
        """

        print("Test: Model-Predict-Input")
    
        ## query inputs
        query = ["2018", "1", "5", "all"]
        
        ## load model first
        y_pred = model_predict(year=query[0], month=query[1], day=query[2], country=query[3])
        self.assertTrue(y_pred.dtype==np.float64)
        
               
    def test_04_predict(self):
        """
        test the predict function accuracy
        """
        
        print("Test: Model-Predict-Accuracy")
   
         ## example predict
        example_queries = [["2018", "11", "02", "all"],
                           ["2019", "01", "01", "EIRE"],
                           ["2018", "03", "05", "all"]]

        for query in example_queries:
            y_pred = model_predict(year=query[0], month=query[1], day=query[2], country=query[3])
            self.assertTrue(y_pred.dtype==np.float64)
        
## run the tests
if __name__ == "__main__":
    unittest.main()
