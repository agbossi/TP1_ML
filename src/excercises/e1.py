import os
import pandas as pd
from src import NaiveBayes
from src.Data_resamplers import train_test_split


def get_classification(classified_element):
    max_v = 0
    classification = None
    for k, v in classified_element.items():
        if max_v < v:
            classification = k
            max_v = v
    return classification


path = os.path.abspath('../../Data/PreferenciasBritanicos(1).xlsx')
df = pd.read_excel(path)

training_percent = 0.4
sets = train_test_split(df, training_percent)[0]
classifier = NaiveBayes.DiscreteNaiveBayes()
classifier.train(training_set=sets[0])
probabilities = classifier.test(test_set=sets[1])
print(probabilities)

