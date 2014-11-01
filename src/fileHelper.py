import numpy as np
import csv


class Images:
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = np.array([])

    def loadData(self, filename):
        with open(filename) as csvfile:
            file_reader = csv.reader(csvfile)

            #Deal with headers
            features = file_reader.next()
            i = 0
            for fields in features:
              if i != 0:
                self.feature_names = np.append(self.feature_names, fields)
              i = i + 1

            #Skip header line and seed data/targets
            next(file_reader, None)
            for row in file_reader:
                self.data.append(row[1:])
                self.target.append(row[0])

class Wines:
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = np.array([])

    def loadData(self, filename):
        with open(filename) as csvfile:
            file_reader = csv.reader(csvfile)

            #Deal with headers
            features = file_reader.next()
            i = 0
            for fields in features:
              if i != 0:
                self.feature_names = np.append(self.feature_names, fields)
              i = i + 1

            #Skip header line and seed data/targets
            next(file_reader, None)
            for row in file_reader:
                self.data.append(row[:10])
                self.target.append(row[11])