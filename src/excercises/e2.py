import os
import pandas as pd
from src.Data_resamplers import train_test_split
from src import NaiveBayes

# path = os.path.abspath('../../Data/Noticias_argentinas.xlsx')
# df = pd.read_excel(path)
# useful_df = df[['titular', 'categoria']]
# final_df = useful_df[(useful_df['categoria'] == 'Deportes')
#               | (useful_df['categoria'] == 'Economia')
#               | (useful_df['categoria'] == 'Entretenimiento')
#               | (useful_df['categoria'] == 'Ciencia y Tecnologia')]

# final_df.to_excel('Noticias_subset.xlsx', index=False)


def create_word_bag(df, classifier):
    word_bag = {}
    for i in range(len(df.index)):
        filtered_words = classifier.filter_title(title=df.iloc[i, 0])
        for word in filtered_words:
            if word not in word_bag:
                word_bag[word] = 1
            else:
                word_bag[word] += 1
    return word_bag


def filter_bag(word_bag, occur):
    blacklist = set()
    for k, v in word_bag.items():
        if v <= occur:
            blacklist.add(k)
    return blacklist


path = os.path.abspath('../../Data/Noticias_subset.xlsx')
df = pd.read_excel(path)

classifier = NaiveBayes.TextNaiveBayes()

word_bag = create_word_bag(df, classifier)
blacklist = filter_bag(word_bag=word_bag, occur=1)

training_percent = 0.4
sets = train_test_split(df, training_percent)[0]
classifier.train(training_set=sets[0], blacklist=blacklist)
probabilities = classifier.test(test_set=sets[1])
