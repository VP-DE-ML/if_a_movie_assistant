
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from src.movie_assistant_trainner import MovieAssistantTrainer


output_dimension = 50
words_number = 100000
sequence_maxlen = 50


train_main = MovieAssistantTrainer()
samples_text, intents_labels, output_rows, unique_labels = train_main.load_training_data('../input/rmovie.json')
tokenizer, embedding_layer = train_main.create_embedding_layer(output_dimension, words_number, sequence_maxlen)
sequences = tokenizer.texts_to_sequences(samples_text)
sequences_padded = pad_sequences(sequences, maxlen=sequence_maxlen)
unique_intents, intent_ids = np.unique(intents_labels, return_inverse=True)
y_train = keras.utils.to_categorical(intent_ids, len(unique_intents))
print(sequences_padded.shape, y_train.shape)
model = train_main.create_model(embedding_layer, y_train.shape[1], sequence_maxlen, output_dimension)
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
hist = model.fit(sequences_padded, y_train, epochs=100, batch_size=100, verbose=1)
train_main.save_model(model, hist, tokenizer, unique_intents, sequence_maxlen, "rmovie_")



#test
text = "recommend me a crazy adventure movie"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])
text = "I want an horror movie"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])
text = "how about an horror movie"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])
text = "recommend me something like an action movie"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])
text = "recommend me a movie about physicology"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])
text = "recommend me a movie with a lot of drama"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])
text = "i would like to watch an action movie"
print(text)
pred = model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=sequence_maxlen))
print(pred)
print(unique_intents[int(pred)])

print("Training sequence completed")
