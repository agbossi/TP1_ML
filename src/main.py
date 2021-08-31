import os
import NaiveBayes as model
import pandas as pd


path = os.path.abspath('../Data/PreferenciasBritanicos(1).xlsx')
df = pd.read_excel(path)
classifier = model.DiscreteNaiveBayes()
classifier.train(df)
