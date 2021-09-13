import enum
import os
import pandas as pd
import numpy as np
from src.Data_resamplers import train_test_split
from src import Metrics
from src.Stat_analysis import calculate_stats
from src import NaiveBayes
import matplotlib.pyplot as plt


# path = os.path.abspath('../../Data/Noticias_argentinas.xlsx')
# df = pd.read_excel(path)
# useful_df = df[['titular', 'categoria']]
# final_df = useful_df[(useful_df['categoria'] == 'Deportes')
#               | (useful_df['categoria'] == 'Economia')
#               | (useful_df['categoria'] == 'Entretenimiento')
#               | (useful_df['categoria'] == 'Ciencia y Tecnologia')]

# final_df.to_excel('Noticias_subset.xlsx', index=False)


def create_word_bag(df):
    word_bag = {}
    for i in range(len(df.index)):
        words = df.iloc[i, 0].split()
        filtered_words = []
        for word in words:
            stripped_word = word.lower().strip(",\a.?¡\\¦.¿!â\'\"°()\r[]\'{}\t-+%\a#\"$/:;_")
            filtered_words.append(stripped_word)
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


class Classifications(enum.Enum):
    ECONOMIA = 0
    DEPORTES = 1
    CIENCIA_Y_TECNOLOGIA = 2
    ENTRETENIMIENTO = 3


def get_classification_index(classification):
    if classification == 'Economia':
        index = Classifications.ECONOMIA.value
    elif classification == 'Deportes':
        index = Classifications.DEPORTES.value
    elif classification == 'Ciencia y Tecnologia':
        index = Classifications.CIENCIA_Y_TECNOLOGIA.value
    else:
        index = Classifications.ENTRETENIMIENTO.value
    return index


def benchmark(iterations, df):
    classifier = NaiveBayes.TextNaiveBayes()

    training_percent = 0.4
    summaries = []
    precisions = []
    f1_scores = []
    accuracies = []
    recalls = []
    for i in range(iterations):
        sets = train_test_split(df, training_percent)[0]
        word_bag = create_word_bag(df=sets[0])
        blacklist = filter_bag(word_bag=word_bag, occur=1)
        classifier.train(training_set=sets[0], blacklist=blacklist)
        probabilities = classifier.test(test_set=sets[1])
        confusion_matrix = Metrics.ConfusionMatrix(['Economia', 'Deportes', 'Ciencia y Tecnologia', 'Entretenimiento'])

        for j in range(len(probabilities)):
            classification = get_classification_index(NaiveBayes.get_classification(classified_element=probabilities[j]))
            real_classification = get_classification_index(sets[1].iloc[j, -1])
            confusion_matrix.add_entry(real_classification, classification)
        confusion_matrix.summarize()
        summaries.append(np.array(confusion_matrix.get_summary()))
        precisions.append(confusion_matrix.get_precisions())
        accuracies.append(confusion_matrix.get_accuracies())
        f1_scores.append(confusion_matrix.get_f1_scores())
        recalls.append(confusion_matrix.get_recalls())
    return


def build_roc_curve(class_value, probabilities):
    roc_points = []
    for i in range(500):
        roc_matrix = Metrics.RocConfusionMatrix(class_value)
        step = i / 500
        for j in range(len(probabilities)):
            real_classification = sets[1].iloc[j, -1]
            roc_matrix.add_entry(real_classification, probabilities[j][class_value], step)
        roc_points.append(roc_matrix.get_roc_point())

    x = [a for a, b in roc_points]
    y = [b for a, b in roc_points]

    xl = np.linspace(0.01, 0, 1)
    yl = xl

    plt.plot(x, y, 'o', color='black')
    plt.plot(xl, xl)
    plt.show()
    return


path = os.path.abspath('../../Data/Noticias_subset.xlsx')
df = pd.read_excel(path)
# iterations = 2
# benchmark(iterations, df)

classifier = NaiveBayes.TextNaiveBayes()
training_percent = 0.4
sets = train_test_split(df, training_percent)[0]
word_bag = create_word_bag(df=sets[0])
blacklist = filter_bag(word_bag=word_bag, occur=1)
classifier.train(training_set=sets[0], blacklist=blacklist)
probabilities = classifier.test(test_set=sets[1])
confusion_matrix = Metrics.ConfusionMatrix(['Economia', 'Deportes', 'Ciencia y Tecnologia', 'Entretenimiento'])

for j in range(len(probabilities)):
    classification = get_classification_index(NaiveBayes.get_classification(classified_element=probabilities[j]))
    real_classification = get_classification_index(sets[1].iloc[j, -1])
    confusion_matrix.add_entry(real_classification, classification)
confusion_matrix.summarize()


# clases = ['Economia', 'Deportes', 'Ciencia y Tecnologia', 'Entretenimiento']
# build_roc_curve('Economia', probabilities)







