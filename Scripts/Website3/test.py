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
        
        print(request.form['source'])
          
        df = get_data()
        data_1 = df.to_json(orient='columns')
        data_1 = json.loads(data_1)

        data_2 = {"key1": "var2", "key2":"var2"}
        
        data = {"line_chart": data_1, "heat_map": data_2}
        
        data_string = json.dumps(data)
        
        return data_string

if __name__ == '__main__':
	# run!
	app.run()
    

    
    
    
    
    
    
    