import unittest
from src.movie_assistant_trainner import MovieAssistantTrainer


class TestTrainMovieAssistant(unittest.TestCase):

    def test_sum(self):
        test_db = MovieAssistantTrainer()

        print("The end")
        self.assertEqual(sum([1, 2]), 3, "Should be 6")

if __name__ == '__main__':
    unittest.main()
