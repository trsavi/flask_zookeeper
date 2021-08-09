# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:50:40 2021

@author: Pc4y
"""

import requests

url_train = 'http://127.0.0.1:8000/predict'  # localhost and the defined port + endpoint
url_put = 'http://127.0.0.1:8000/put'
body = {
    "model": 1, 
    "petal_length": 2,
    "sepal_length": 0.5,
    "petal_width": 0.5,
    "sepal_width": 4
}


response_train = requests.get(url_put)

#response_post = requests.post(url_train, data = body)

print(response_train.json())

#print(response_post.json())
