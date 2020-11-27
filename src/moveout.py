
import nltk
from nltk.stem import WordNetLemmatizer

import json

# things we need for Tensorflow
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Dense, Embedding
from keras.optimizers import SGD
from keras.layers import LSTM
import pickle
import random
import os
from keras.preprocessing.text import Tokenizer
#from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical


# import our chat-bot intents file
from tensorflow.python.keras.models import model_from_json

data_file = open('../input/intents.json').read()
intents_file = json.loads(data_file)

# build vocabulary first. Patterns(user input) are processed to build a vocabulary. Each word is stemmed to produce generic root, this would help to cover more combinations of user input:
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']

intents_labels = []
output_rows = []

#Modified in Here the creation of the training layer in order to make it similar to the one created from ATIS#################################
#Convert from output, lables to text,text to labels to sequece to data to train data to model.fit; lables to labels to unique num_classes & uniques, ids to y_train to model.fit
# loop through each sentence in our intents file

for item in intents_file['intents']:
    for pattern in item['userinputs']:
        row = {}
        xx = ""
        for x in pattern:
            xx = xx + " " + x.lower()
        row['text2'] = xx
        row['text'] = pattern.lower()
        row['intent'] = item['intent']
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add documents in the corpus
        documents.append((w, item['intent']))  # looks similar to output_rows but each pattern tokenized

        # add to our classes list
        if item['intent'] not in classes:
            classes.append(item['intent'])  # looks the same than unique_labels
    intents_labels.append(item['intent'])
    output_rows.append(row)
unique_labels = np.unique(intents_labels)


## New from here:
##


def save_model(model, filename):

    model_json = model.to_json()
    with open(filename + '.model', "w") as json_file:
        json_file.write(model_json)
        json_file.close();
    model.save_weights(filename + ".weights")

def load_model(filename):

    json_file = open(filename + '.model', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + ".weights")

    return loaded_model;


def getEmbeddingLayer(EMBEDDING_DIM):
    embeddings_index = {}
    f = open('../input/pre_train_50d.txt', encoding="utf8")
    count = 0
    words = []
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        count = count+1;
    f.close()

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS) #Everything will be defined later
    tokenizer.fit_on_texts(words)
    word_index = tokenizer.word_index

    print("total words embeddings is ", count, len(word_index))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    return tokenizer,embedding_layer


def create_embedded_model(embedding_layer, num_classes, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128, return_sequences=False
                   , input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))

    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    return model

## Validate from here also consider save and load the model

# print output[0],lables
print("number of samples", len(output_rows))
print("number of intent", len(intents_labels))

EMBEDDING_DIM = 50
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 50
text = []
labels = []
for x in output_rows:
    text.append(x['text'])
    labels.append(x['intent'])


num_classes = np.unique(labels)

print('Found %s texts.' % len(text))
print("number of classes", len(num_classes))

tokenizer, embedding_layer = getEmbeddingLayer(EMBEDDING_DIM)
sequences = tokenizer.texts_to_sequences(text)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
uniques, ids = np.unique(labels, return_inverse=True)
y_train = keras.utils.to_categorical(ids, len(uniques))


print("Training data")
print(data.shape, y_train.shape)

model = create_embedded_model(embedding_layer, y_train.shape[1], MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(data, y_train, epochs=100, batch_size=100, verbose=1)
model.save('../lib/movieassis_model.h5', hist)
save_model(model, "../lib/movie_assistant_model")
print("Model saved")

#tokenizer.texts_to_sequences('recommend me a crazy movie')
txt = "recommend me a crazy movie"
seq = tokenizer.texts_to_sequences([txt])
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict_classes(padded)
print("Prediction2")
print(pred)
print(uniques[int(pred)])
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences(["hi"]), maxlen=MAX_SEQUENCE_LENGTH))
print(pred)
print(uniques[int(pred)])
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences(["what can you do"]), maxlen=MAX_SEQUENCE_LENGTH))
print(pred)
print(uniques[int(pred)])
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences(["recommend me something"]), maxlen=MAX_SEQUENCE_LENGTH))
print(pred)
print(uniques[int(pred)])
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences(["recommend me a movie about physicology"]), maxlen=MAX_SEQUENCE_LENGTH))
print(pred)
print(uniques[int(pred)])

print("The End")
