#!flask/bin/python

import sys

from flask import Flask, render_template, request, redirect, Response
import random, json

app = Flask(__name__)

@app.route('/')
def output():
    # serve index template
    return render_template('index.html')

@app.route('/receiver', methods = ['POST'])
def worker():
	# read json + reply
    data = request.form['name']
    
    print(data)
    
    return data

if __name__ == '__main__':
	# run!
	app.run()