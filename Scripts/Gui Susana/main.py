#!flask/bin/python

import sys

from flask import Flask, render_template, request, redirect, Response
import random, json
import pandas as pd
import sys
sys.path.append('../Functions')
from website_functions import *

app = Flask(__name__)

@app.route('/')
def output():
    # serve index template
    return render_template('index.html')
 

@app.route('/receiver', methods = ['POST'])
def worker():
	# read json + reply
    request_type = request.form['request_type']
    
    if(request_type=="json"):
        data = get_json()
        data_string = json.dumps(data)
        
        return data_string
    
    else:
        data = process_data_request(request)
        data_string = json.dumps(data)
        #print(data_string)
        return data_string

if __name__ == '__main__':
	# run!
	app.run()
    

    
    
    
    
    
    
    