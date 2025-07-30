#!/usr/bin/env python
"""
api tests

these tests use the requests package however similar requests can be made with curl

e.g.
data = '{"key":"value"}'
curl -X POST -H "Content-Type: application/json" -d "%s" http://localhost:8080/predict'%(data)
"""

import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np
import pandas as pd

port = 8080

try:
    requests.post('http://localhost:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_predict(self):
        """
        test the predict functionality
        """

        query = {"year":"2019","month":"2","day":"1","country":"all"}
        r = requests.post('http://localhost:{}/predict'.format(port),json=query)
        response = literal_eval(r.text)
        self.assertTrue(isinstance(response[0],float))

    @unittest.skipUnless(server_available,"local server is not running")
    def test_train(self):
        """
        test the train functionality
        """
      
        query = {"ts_dir":"TS_DIR", "test": "False" }
        r = requests.post('http://localhost:{}/train'.format(port),json=query)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')   
        
    def test_logging(self):
        """
        test the logging functionality
        """
        
        query = {"env":"train","year":"2021","month":"04"}
        r = requests.post('http://localhost:{}/logging'.format(port),json=query)
        response = literal_eval(r.text)
        self.assertEqual(response.get("logfile"),'train-2021-4.log')
    
### Run the tests
if __name__ == '__main__':
    unittest.main()

