#!flask/bin/python

import sys

from flask import Flask, render_template, request, redirect, Response
import random, json
import pandas as pd

app = Flask(__name__)

@app.route('/')
def output():
    # serve index template
    return render_template('index.html')

@app.route('/receiver', methods = ['POST'])
def worker():
	# read json + reply
    name = request.form['name']
    
    df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                  index=['row 1', 'row 2'],
                  columns=['col 1', 'col 2'])
    
    data = df.to_json(orient='columns')
    
    data_string = json.dumps(data)
    
    
    print(data_string)
    
    return data_string

if __name__ == '__main__':
	# run!
	app.run()