import csv
import unittest
from src.save_prediction import SavePrediction

output_file = "../output/test_prediction.csv"
prediction_module = "test_module"
user_input = "user_input_for_test"
intent_prediction = "test_input_prediction"
test_row = [intent_prediction, user_input]


class TestSavePrediction(unittest.TestCase):

    def test_save_prediction(self):
        save_prediction = SavePrediction(output_file, prediction_module)
        saved_row = save_prediction.write_prediction(test_row)
        with open(output_file, ) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')#, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for csv_row in csv_reader:
                print("reading")
                print(csv_row)
                pass
        print("saved row: " + str(saved_row))
        print("last row: " + str(csv_row))
        self.assertListEqual(saved_row, csv_row)

if __name__ == '__main__':
    unittest.main()
