
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import pickle

class IdentifyAction:

    def __init__(self):
        self.sequence_maxlen = 50
        self.recommend_movie_model = load_model('../lib/rmovie_model.h5')
        self.recommend_movie_tokenizer = pickle.load(open('../lib/rmovie_tokenizer.pkl', 'rb'))
        self.recommend_movie_unique_intents = pickle.load(open('../lib/rmovie_intents.pkl', 'rb'))
        self.new_filter = ""

    def call_spark_assistant(self, spark_db, spark_session, genre):
        self.new_filter = "genre = " + "\"" + genre + "\""
        output_message = spark_db.recommend_movie_by(spark_session, self.new_filter)
        print(output_message)
        return output_message

    def recommend_movie(self, spark_db, spark_session, user_input):
        input_text = str(user_input).lower()
        input_tokenized = word_tokenize(user_input)
        # input_tokenized = user_input.split()
        valid_input = False
        for genre in self.recommend_movie_unique_intents:  # valid until all the cases has been implemented
            if str(genre).lower() in input_tokenized:
                valid_input = True
                break
        if valid_input:
            seq = self.recommend_movie_tokenizer.texts_to_sequences([input_text])
            padded = pad_sequences(seq, maxlen=self.sequence_maxlen)
            intent_number = int(self.recommend_movie_model.predict_classes(padded))
            intent_description = self.recommend_movie_unique_intents[intent_number]
            output_message = self.call_spark_assistant(spark_db, spark_session, intent_description)
        else:
            output_message = "Since I am still learning... " \
                              "At this moment I only recommend movies if you specify one of this genres: " \
                              "Adventure, Comedy, Action, Drama, Crime, " \
                              "Thriller, Horror, Animation, Biography, " \
                              "Sci-Fi, Musical, Family, Fantasy, Mystery, " \
                              "War, Romance, Western, science fiction. \n\nSo please ask again.."
        return output_message

    def call_action(self, spark_db, spark_session, action_name, user_input):
        concatenate_input = ""
        output_message = ""
        if action_name == "more_details_for_recommendation":
            concatenate_input = "recommend me"
        elif action_name == "movie_recommendation":
            output_message = self.recommend_movie(spark_db, spark_session, user_input)
        else:
            output_message = ""
        return concatenate_input, output_message
