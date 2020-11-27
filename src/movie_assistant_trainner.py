import tensorflow as tf
import json
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import pickle
import random
import os
from keras.preprocessing.text import Tokenizer
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical

class MovieAssistantTrainer:

    def __init__(self):
        pass

    def load_training_data(self, training_file):
        data_file = open(training_file).read()
        intents_file = json.loads(data_file)
        samples_text = []
        intents_labels = []
        output_rows = []
        for item in intents_file['intents']:
            for pattern in item['userinputs']:
                pattern = str(pattern).lower() # review
                row = {}
                row['text'] = pattern
                row['intent'] = item['intent']
                samples_text.append(pattern)
                intents_labels.append(item['intent'])
                output_rows.append(row)
        unique_labels = np.unique(intents_labels)
        print("Samples: ", len(output_rows))
        print("Intents: ", len(intents_labels))
        print("Unique intents: ", len(unique_labels))
        return samples_text, intents_labels, output_rows, unique_labels

    # Save model and components
    def save_model(self, model, hist, tokenizer, unique_intents, sequence_maxlen, filename):
        # Save Keras Tokenizer
        path = "../lib/"
        pickle.dump(tokenizer, open(path + filename + 'tokenizer.pkl', 'wb'))
        print("Tokenizer saved")
        # Save unique intents
        pickle.dump(unique_intents, open(path + filename + 'intents.pkl', 'wb'))
        print("Unique intents saved")
        # Save model
        model.save(path + filename + 'model.h5', hist)
        print("Model Saved")
        # Save weights
        model.save_weights(path + filename + 'weights.hdf5')
        print("Weights saved")
        pickle.dump(sequence_maxlen, open(path + filename + 'sequence_maxlen.pkl', 'wb'))
        print("vector length saved")

    def create_embedding_layer(self, output_dimension, words_number, sequence_maxlen):
        # count = 0 MOre behind
        words_list = []
        embeddings_index = {}
        with open('../input/pre_train_50d.txt', encoding="utf8") as pre_trained:
            for line in pre_trained:
                splitted_line = line.split()
                word = splitted_line[0]
                words_list.append(word)
                coefs = np.asarray(splitted_line[1:], dtype='float32')
                embeddings_index[word] = coefs
                # count = count+1
        pre_trained.close()
        print("Found %s word vectors." % len(embeddings_index))
        tokenizer = Tokenizer(num_words=words_number, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(words_list)
        word_index = tokenizer.word_index
        print("Vocabulary index: ", len(word_index))
        # Prepare embedding matrix
        hits = 0
        misses = 0
        embedding_matrix = np.zeros((len(word_index) + 1, output_dimension))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
        embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=output_dimension,
                                    weights=[embedding_matrix], input_length=sequence_maxlen, trainable=True)
        print("Emb created")
        return tokenizer, embedding_layer

    def create_model(self, embedding_layer, num_classes, sequence_maxlen, output_dimension):
        model = Sequential()
        model.add(embedding_layer)
        model.add(LSTM(128, return_sequences=False, input_shape=(sequence_maxlen, output_dimension)))
        model.add(Dense(num_classes, activation='softmax'))
        print(model.summary())
        return model

