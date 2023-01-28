from flask import Flask, render_template, request

import boto3

import datetime
import json
import os
import pickle
import numpy as np

import sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

import logging

target_name_dict = {'stat.AP' : 0,
                    'stat.CO' : 1,
                    'stat.ME' : 2,
                    'stat.ML' : 3,
                    'stat.OT' : 4
                    }

label2target = { v:k for k,v in target_name_dict.items()}

app = Flask(__name__)

#@app.route("/")
@app.route("/", methods=['GET', 'POST'])
def main():
   logging.warning('Starting the app')

   boto3.Session().resource('s3').Bucket('serverless-ml-1').download_file('finalized_vectorizer.sav', '/tmp/finalized_vectorizer.sav')
   boto3.Session().resource('s3').Bucket('serverless-ml-1').download_file('finalized_model.sav', '/tmp/finalized_model.sav')
   
   with open('/tmp/finalized_vectorizer.sav', 'rb') as handle:
       vectorizer = pickle.load(handle)

   with open('/tmp/finalized_model.sav', 'rb') as handle:
       model = pickle.load(handle)

   now = datetime.datetime.now()
   timeString = now.strftime("%Y-%m-%d %H:%M")
   cpuCount = os.cpu_count()
   templateData = {
      'title' : 'Web App for classifying abstracts on statistics',
      'time': timeString,
      'cpucount' : cpuCount,
      'skversion' : sklearn.__version__
      }
      
   logging.warning('request.method is %s', request.method)
   
   if request.method == 'GET':
      return render_template('index.html', **templateData)
   elif request.method == 'POST':
      resultText = "You wrote: " + request.form['myTextArea']
      
      logging.warning('request text is %s', request.form['myTextArea'])
      
      seq_1 = request.form['myTextArea']
      #seq_2 = pad_sequences(seq_1, padding='post', value=0, maxlen=350)
      
      logging.warning('seq_1 is %s', seq_1)
      logging.warning('text length is %s', len(request.form['myTextArea']))
           
      prob = model.predict_proba(vectorizer.transform([seq_1]))
      #prob /= prob.sum()
      logging.warning('prob0 is %s', prob)
      prob = prob.sum(axis=0)
      logging.warning('prob is %s', prob)
      ii = np.argmax(prob)
      logging.warning('ii is %s', ii)
      if max(prob) >= 0.4 and len(request.form['myTextArea']) > 40:
         final_label = label2target[ii]
      else: 
         final_label = 'not a stats abstract'
         
      logging.warning('final_label is %s', final_label)

      results = {'text' : resultText, 'label' : final_label}
      return render_template('index.html', results=results, **templateData)


if __name__ == "__main__":
   app.run()
