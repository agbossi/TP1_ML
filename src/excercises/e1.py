import os
import pandas as pd
from src import NaiveBayes
from src.Data_resamplers import train_test_split

path = os.path.abspath('../../Data/PreferenciasBritanicos(1).xlsx')
df = pd.read_excel(path)

training_percent = 0.7
sets = train_test_split(df, training_percent)[0]
classifier = NaiveBayes.DiscreteNaiveBayes()
classifier.train(training_set=sets[0])
example_df = pd.DataFrame(data=[[0, 1, 1, 0, 1], [1, 0, 1, 1, 0]], columns=['scones', 'cerveza', 'wiskey', 'avena',	'futbol'])
probabilities_e1 = classifier.test(example_df)
print(probabilities_e1)
print('\n')

