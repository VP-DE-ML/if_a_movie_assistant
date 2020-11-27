
import pickle
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import json
import random
from src.call_action import IdentifyAction
from src.spark_db import SparkDB
from src.save_prediction import SavePrediction


spark_db = SparkDB()
mSpark = spark_db.create_session()
identify_action = IdentifyAction()

class MovieAssistantCore:

    def __init__(self):
        self.mv_tokenizer = pickle.load(open('../lib/movie_assistant_core_tokenizer.pkl', 'rb'))
        self.recommendation_model = load_model('../lib/movie_assistant_core_model.h5')
        self.data_file = json.loads(open('../input/intents.json').read())
        self.unique_intents = pickle.load(open('../lib/movie_assistant_core_intents.pkl', 'rb'))
        self.intents_response_file = json.loads(open('../input/intents.json').read())
        self.output_file = "../output/core_prediction.csv"
        self.prediction_module = "core_module"
        self.save_prediction = SavePrediction(self.output_file, self.prediction_module)
        self.sequence_maxlen = 50
        self.concatenate_input = ""

    def predict_intent(self, user_input):
        seq = self.mv_tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=self.sequence_maxlen)
        intent_number = int(self.recommendation_model.predict_classes(padded))
        intent_description = self.unique_intents[intent_number]
        print(intent_number)
        print(" " + self.unique_intents[intent_number] + ": " + user_input)
        print(seq)
        return intent_number, intent_description

    def get_response(self, intent_description):
        intents_list = self.intents_response_file['intents']
        for each_intent in intents_list:
            if (each_intent['intent'] == intent_description):
                result = random.choice(each_intent['responses'])
                break
        print(result)
        return result

    def chatbot_response(self, user_input):
        if self.concatenate_input != "":
            user_input = self.concatenate_input + " " + user_input
        user_input = user_input.replace("|", "")
        intent_number, intent_description = self.predict_intent(user_input)
        self.save_prediction.write_prediction([user_input, intent_description])
        response_bot = self.get_response(intent_description)
        self.concatenate_input, output_dataset = identify_action.call_action(spark_db, mSpark, intent_description, user_input)
        print(self.concatenate_input)
        if output_dataset != "":
            response_bot = output_dataset
        return response_bot

    def parse_output(self, output_dataset):
        return output_dataset
