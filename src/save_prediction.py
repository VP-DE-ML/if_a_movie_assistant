import csv
from datetime import datetime


class SavePrediction:

    def __init__(self, output_file, prediction_module):
        self.output_file = output_file
        self.prediction_module = prediction_module
        self.session_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.validated = 'False'

    def write_prediction(self, row):
        row.append(self.prediction_module)
        row.append(self.session_time)
        row.append(self.validated)
        with open(self.output_file, 'a', newline='') as csv_file:
            scv_writer = csv.writer(csv_file, delimiter='|') #, quotechar='|', quoting=csv.QUOTE_MINIMAL)
            scv_writer.writerow(row)
            print("row saved: ")
            print(row)
        return row
