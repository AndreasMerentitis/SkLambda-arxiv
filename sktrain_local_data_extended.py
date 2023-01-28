import numpy as np 
import pandas as pd
import sys
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 

import pdb

def get_abstracts(files):
    abstracts = []
    labels = []
    for f in files:
       store = pd.HDFStore(f)
       df = store['/df']
       store.close()

       abstracts += list(df['abstract'])
       labels = np.hstack([labels,np.array(df['categories'])])

    labels = np.asarray([item[0] for item in labels.tolist()])
    selected_labels = ['stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT']
    labels_selected = np.asarray([item for item in labels.tolist() if item in selected_labels])

    jj = 0 
    abstracts_selected = []

    for item in labels.tolist(): 
       if item in selected_labels:
           abstracts_selected.append(abstracts[jj])
           jj = jj + 1
        
    abstracts_selected = np.asarray(abstracts_selected)
    return labels_selected, abstracts_selected


def define_model(abstracts_selected, labels_selected):

    #vectorizer = CountVectorizer(max_features=10000)
    vectorizer = TfidfVectorizer(max_features=15000)
    
    sk_sequences = vectorizer.fit_transform(abstracts_selected)

    np.random.seed(1234)
    ind = np.random.randint(0, len(labels_selected), len(labels_selected))
    print(ind.shape)
    labels_selected = labels_selected[ind]

    split_1 = int(0.8 * len(labels_selected))
    split_2 = int(0.9 * len(labels_selected))
    train_labels = labels_selected[:split_1]
    dev_labels = labels_selected[split_1:split_2]
    test_labels = labels_selected[split_2:]
    
    train_seq_sk = sk_sequences[:split_1, :]
    dev_seq_sk = sk_sequences[split_1:split_2, :]
    test_seq_sk = sk_sequences[split_2:, :]

    #%%
    #model = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=42, max_iter=20, tol=None)
    #model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=20, tol=0.001)
    #model = OneVsRestClassifier(RidgeClassifier())
    model = OneVsRestClassifier(SGDClassifier(loss='log', max_iter=200))


    return model, train_labels, dev_labels, test_labels, train_seq_sk, dev_seq_sk, test_seq_sk, vectorizer


target_name_dict = {'stat.AP' : 0,
                    'stat.CO' : 1,
                    'stat.ME' : 2,
                    'stat.ML' : 3,
                    'stat.OT' : 4
                    }

label2target = { v:k for k,v in target_name_dict.items()}

files = ["data/2015ml.h5",
         "data/2016ml.h5",
         "data/2017ml.h5",
         "data/2018ml.h5",
         "data/2019ml.h5",
        ]

labels_selected, abstracts_selected = get_abstracts(files)

print (np.unique(labels_selected))
print("---------")


for i in range(2):
    print(abstracts_selected[i])
    print(target_name_dict[labels_selected[i]])
    print("---------")

labels_selected_num = np.asarray([target_name_dict[x] for x in labels_selected.tolist()])
labels_selected_num = labels_selected_num + 1

df = pd.read_csv('texts.csv')  

# oversample the dataframe to get more negative samples that are not arxiv ML paper of any type
df_extended = df.sample(n=3000, replace='True')

texts = df_extended.texts
y_true = df_extended.labels

abstracts_selected_extended = np.concatenate((abstracts_selected, np.asarray(texts)), axis=0)
labels_selected_extended = np.concatenate((labels_selected_num, np.asarray(y_true)), axis=0)

# split the training/testing data and return the model template 
model, train_labels, dev_labels, test_labels, train_seq, dev_seq, test_seq, vectorizer = define_model(abstracts_selected_extended, labels_selected_extended)

print (pd.value_counts(labels_selected_extended))

model.fit(train_seq, train_labels)
pred = model.predict(test_seq)

print (np.unique(train_labels, return_counts=True))
print (np.unique(pred, return_counts=True))

ev = np.mean(pred == test_labels)
print(ev)

#pdb.set_trace()

## Predict for one example to show that the flow works with the model in memory 
model.predict(vectorizer.transform([np.asarray(texts)[0]]))

#%%

# save the vectorizer to disk
filename_vectorizer = 'finalized_vectorizer.sav'
pickle.dump(vectorizer, open(filename_vectorizer, 'wb'))

# save the model to disk
filename_model = 'finalized_model.sav'
pickle.dump(model, open(filename_model, 'wb'))

print("Saved model to disk")

# load the model from disk
loaded_model = pickle.load(open(filename_model, 'rb'))

category_labels = []
category_proba = []
for text in texts:
    label2target = loaded_model.predict(vectorizer.transform([text]))
    prob = loaded_model.predict_proba(vectorizer.transform([text]))
    
    prob = prob.sum(axis=0) 

    if max(prob) >= 0.4 and len(text) > 40:
         category_label = 1
         category_labels.append(category_label)
         category_proba.append(max(prob))
    else:
         category_label = 0
         category_labels.append(category_label)
         category_proba.append(max(prob))

y_predict = np.asarray(category_labels)
precision, recall, thresholds_prr = precision_recall_curve(y_true, y_predict)
fpr, tpr, thresholds_roc = roc_curve(y_true, y_predict)

#pdb.set_trace()

pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.show()

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

print (confusion_matrix(y_true.values, y_predict))

print (np.mean(y_predict == y_true.values))




