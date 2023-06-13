from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow
import tflearn
import pickle
import json
import nltk
import os


stemmer = LancasterStemmer()

with open('pwd.txt', 'r') as pwd:
    folder_location = pwd.read()

def bag_of_words(s, wrd):
    bag2 = [0 for _ in range(len(wrd))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(wrd):
            if w == se:
                bag2[i] = 1

    return np.array(bag2)

with open(f"{folder_location}intents.json") as file:
    data = json.load(file)

try:
    with open(f"{folder_location}models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    print('TENSORFLOW MODEL FOUND')
    print('INITIATING MODEL...')
except FileNotFoundError:
    print('TENSORFLOW MODEL NOT FOUND')
    print('INITIATING TRAINING PROTOCOL...')
    os.system('/usr/local/bin/python3f {folder_location}train.py')
    print('MODEL TRAINING PROTOCOL SUCCESS')
    with open(f"{folder_location}models/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

tensorflow.compat.v1.reset_default_graph()


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load(f"{folder_location}models/model.tflearn")
    print('MODEL READY TO BE USED')
except:
    model.fit(training, output, n_epoch=800, batch_size=8, show_metric=True)
    model.save(f"{folder_location}models/model.tflearn")
    print('MODEL READY TO BE USED')
