import pickle
from keras.models import load_model
import json

from keras_preprocessing.sequence import pad_sequences

sequence_maxlen = 50
model = load_model('../lib/movie_assistant_model.h5')
data_file = json.loads(open('../input/intents.json').read())
unique_intents = pickle.load(open('../lib/movie_assistant_intents.pkl', 'rb'))
mv_tokenizer = pickle.load(open('../lib/movie_assistant_tokenizer.pkl', 'rb'))
#with open('../lib/movie_assistant_tokenizer.pkl', 'rb') as handle:
#    mv_tokenizer = pickle.load(handle)


#tokenizer.texts_to_sequences('recommend me a crazy movie')
input_test = "recommend me a crazy movie"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "hi"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "what are you able to do?"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "please recommend me something"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "recommend me a crazy movie"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "recommend me a movie about physicology"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "great recommendation"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "great recommendation, thank you"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "see ya"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "kns ish fhasdoufh oasadf lj;sad"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "i want to know the year of the movie matrix"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "who are the actors with more movies"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "who is a good actor"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)
input_test = "what is a good movie"
seq = mv_tokenizer.texts_to_sequences([input_test])
padded = pad_sequences(seq, maxlen=sequence_maxlen)
pred = model.predict_classes(padded)
print(pred)
print(" " + unique_intents[int(pred)] + ": " + input_test)

print("The End")





