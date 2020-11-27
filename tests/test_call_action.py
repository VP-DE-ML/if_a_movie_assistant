import unittest
from src.spark_db import SparkDB
from src.call_action import IdentifyAction

identify_action = IdentifyAction()
spark_db = SparkDB()
#new_filter = "genre = " + "'" + "Action" + "'"
spark_session = spark_db.create_session()
#action_name = "more_details_for_recommendation"
#user_input = ""

class TestCallAction(unittest.TestCase):

    def test_call_action_more_details_for_recommendation(self):
        concatenate_input, output_message = identify_action.call_action("spark_db", "spark_session", "more_details_for_recommendation", "please recommend me ")
        print("concatenate_input: " + concatenate_input)
        print("output_message: " + output_message)
        self.assertEqual(concatenate_input, "recommend me", "Should be equal")
        self.assertEqual(output_message, "", "Should be equal")

    def test_call_action_movie_recommendation_pending(self):
        concatenate_input, output_message = identify_action.call_action("spark_db", "spark_session", "movie_recommendation", "please recommend me something")
        print("concatenate_input: " + concatenate_input)
        print("output_message: " + output_message)
        self.assertEqual(concatenate_input, "", "Should be equal")
        self.assertEqual(output_message, "Since I am still learning... At this moment I only recommend movies if you specify one of this genres: Adventure, Comedy, Action, Drama, Crime, Thriller, Horror, Animation, Biography, Sci-Fi, Musical, Family, Fantasy, Mystery, War, Romance, Western, science fiction", "Should be equal")

    def test_call_action_movie_recommendation_available(self):
        concatenate_input, output_message = identify_action.call_action(spark_db, spark_session, "movie_recommendation", "please recommend me an action movie")
        print("concatenate_input: " + concatenate_input)
        print("output_message: " + output_message)
        spark_db.stop_spark_session(spark_session)
        self.assertEqual(concatenate_input, "", "Should be equal")
        self.assertEqual(output_message, "Recommending first 5 movies: \n\n 1 Dangal with Aamir Khan\n 2 Elite Squad: The Enemy Within with Wagner Moura\n 3 Turbo Kid with Munro Chambers\n 4 The Matrix with Keanu Reeves\n 5 Terminator 2: Judgment Day with Arnold Schwarzenegger", "Should be equal")


if __name__ == '__main__':
    unittest.main()
