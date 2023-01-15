import numpy as np 
import tensorflow as tf
import pandas as pd
import sys
import json
import pickle

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, model_from_config, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
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
    num_words = 10000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(abstracts_selected)
    sequences = tokenizer.texts_to_sequences(abstracts_selected)
    seq = pad_sequences(sequences, padding='post', value=0, maxlen=100)


    # Tokenizers come with a convenient list of words and IDs
    dictionary = tokenizer.word_index
    # Let's save this out so we can use it later
    with open('dictionary_ML.json', 'w') as dictionary_file:
       json.dump(dictionary, dictionary_file)


    # saving
    with open('tokenizer.pickle', 'wb') as handle:
       pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # loading

    np.random.seed(1234)
    ind = np.random.randint(0, len(labels_selected), len(labels_selected))
    print(ind.shape)
    labels_selected = labels_selected[ind]
    seq = seq[ind,:]

    split_1 = int(0.8 * len(labels_selected))
    split_2 = int(0.9 * len(labels_selected))
    train_labels = labels_selected[:split_1]
    dev_labels = labels_selected[split_1:split_2]
    test_labels = labels_selected[split_2:]

    train_seq = seq[:split_1, :]
    dev_seq = seq[split_1:split_2, :]
    test_seq = seq[split_2:, :]

    #%%
    vocab_size = 10000

    #%%
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 64))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(5, activation=tf.nn.sigmoid))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model, train_labels, dev_labels, test_labels, train_seq, dev_seq, test_seq, tokenizer


def predict_text(text1):
    #%%
    seq_1 = tokenizer.texts_to_sequences(text1)

    #%%
    seq_2 = pad_sequences(seq_1, padding='post', value=0, maxlen=350)

    #%%
    prob = model.predict(seq_2)
    prob /= prob.sum()
    print(prob)
    ii = np.argmax(prob)
    print(label2target[ii])
    return prob, label2target
     

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



model, train_labels, dev_labels, test_labels, train_seq, dev_seq, test_seq, tokenizer = define_model(abstracts_selected, labels_selected)

y_train_num = np.asarray([target_name_dict[x] for x in train_labels.tolist()])
y_test_num = np.asarray([target_name_dict[x] for x in test_labels.tolist()])

train_labels_onehot = to_categorical(y_train_num)
test_labels_onehot = to_categorical(y_test_num)

history = model.fit(train_seq, train_labels_onehot, epochs=10, steps_per_epoch=32, validation_split=0.3, validation_steps=32)


#%%
ev  = model.evaluate(test_seq, test_labels_onehot, steps=10)
print(ev)

df = pd.read_csv('texts.csv')  

texts = df.texts
y_true = df.labels



## Predict for one example to show that the flow works with the model in memory 
prob, label2target = predict_text([texts[0]])

#%%

# serialize model to JSON
model_json = model.to_json()
with open("model_ML.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save('new_model_ML.h5')

print("Saved model to disk")

# load tokenizer from pickle, model from HDF5 and predict for all examples

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


load_model('new_model_ML.h5')

category_labels = []
for text in texts:
    prob, label2target = predict_text([text])
    prob = prob.sum(axis=0) 

    if max(prob) >= 0.2 and len(text[0]) > 40:
         category_label = 1
         category_labels.append(max(prob))
    else:
         category_label = 0
         category_labels.append(max(prob))


y_predict = np.asarray(category_labels)
precision, recall, thresholds_prr = precision_recall_curve(y_true, y_predict)
fpr, tpr, thresholds_roc = roc_curve(y_true, y_predict)

#pdb.set_trace()

pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.show()

roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()



