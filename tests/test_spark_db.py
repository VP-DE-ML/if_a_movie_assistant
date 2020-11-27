import unittest
from src.spark_db import SparkDB


class TestSparkDB(unittest.TestCase):

    def test_recommend_movie_by(self):
        spark_db = SparkDB()
        new_filter = "genre = " + "'" + "Action" + "'"
        spark_session = spark_db.create_session()
        db_output = spark_db.recommend_movie_by(spark_session, new_filter)
        expected_output = "Recommending first 5 movies: \n\n 1 Dangal with Aamir Khan\n 2 Elite Squad: The Enemy Within with Wagner Moura\n 3 Turbo Kid with Munro Chambers\n 4 The Matrix with Keanu Reeves\n 5 Terminator 2: Judgment Day with Arnold Schwarzenegger"
        print(db_output)
        spark_db.stop_spark_session(spark_session)
        self.assertEqual(db_output, expected_output, "Should be equals:")

    def test_retrieve_movie_information(self):
        pass

if __name__ == '__main__':
    unittest.main()
