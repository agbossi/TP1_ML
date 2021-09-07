# missing data :
# https://datascience.stackexchange.com/questions/3711/how-does-the-naive-bayes-classifier-handle-missing-data-in-training
# potencially usefull tips:
# https://machinelearningmastery.com/better-naive-bayes/

import nltk
# Nos estamos bajando diccionarios que mapean palabras a sus versiones normalizadas
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def laplace_correction(occurrences, total, possible_values):
    return (occurrences + 1) / (total + possible_values)


class DiscreteNaiveBayes:

    def __init__(self):
        self.classes_probabilities = None
        self.cond_frequencies = None
        self.data_count = 0
        self.classes = 0
        self.training_set = None
        self.attribute_names = None
        self.attribute_values = None
        # calcular una probabilidad a partir de valor que no esta en estructura frequencias tengo el total
        # para correccion de laplace
        self.class_attr_set_freq = None

    def cond_P(self, attribute, value, given):
        total_cond_attr_frequency = self.class_attr_set_freq[given][attribute]
        if value not in self.cond_frequencies[given][attribute]:
            cond_attr_frequency = 0
        else:
            cond_attr_frequency = self.cond_frequencies[given][attribute][value]
        return laplace_correction(cond_attr_frequency, total_cond_attr_frequency, len(self.attribute_values[attribute]))

    def P(self, class_value):
        return self.classes_probabilities[class_value]

    def calculate_total_probability(self, test_elem):
        accum = 0
        for class_value in self.classes_probabilities:
            prod = 1
            for i in range(len(self.attribute_names) - 1):
                curr_attr = self.attribute_names[i]
                prod *= self.cond_P(attribute=curr_attr, value=test_elem[i], given=class_value)
            prod *= self.P(class_value)
            accum += prod
        return accum

    def train(self, training_set):
        self.training_set = training_set
        # to divide later to get probabilities
        self.data_count = len(training_set.index)
        # store columns for easier counting and access
        self.attribute_names = self.training_set.columns
        self.attribute_values = self.get_attribute_values()
        # classes frequencies to dict obtained from classes column
        classes_frequencies = training_set[self.attribute_names[-1]].value_counts().to_dict()
        self.classes_probabilities = {k: laplace_correction(v, self.data_count, len(classes_frequencies)) for k, v in classes_frequencies.items()}
        self.classes = len(self.classes_probabilities)
        ret = self.calculate_cond_table()
        p = self.cond_P(self.attribute_names[0], 0, 'E')
        return ret

    def test(self, test_set):
        results = {}
        for i in range(len(test_set)):
            test_elem = test_set.iloc[i, :]
            probabilities = {}
            # denominador comun a todas las clases
            total_probability_values = self.calculate_total_probability(test_elem)
            for class_value in self.classes_probabilities:
                prod = 1
                for j in range(len(self.attribute_names) - 1):
                    prod *= self.cond_P(attribute=self.attribute_names[j],
                                        value=test_elem[self.attribute_names[j]],
                                        given=class_value)
                prod *= self.P(class_value)
                arg = prod / total_probability_values
                probabilities[class_value] = arg
            results[i] = probabilities
        return results

    def calculate_cond_table(self):
        cond_probabilities = {}
        cond_frequencies = {}
        total_class_attr_set_freq = {}
        # already have all classes without repetition here
        for class_value in self.classes_probabilities:
            attr_prob = {}
            # to store key: possible values for given attribute,
            # value: count of that value for given class
            attr_dict = {}
            # list of total values for conditional subsets for each attr
            class_attr_set_freq = {}
            # loop through attributes, excluding class column
            for i in range(len(self.attribute_names) - 1):
                curr_attr = self.attribute_names[i]
                # get attribute column and class column
                attribute_set = self.training_set[[curr_attr, self.attribute_names[-1]]]
                # keep rows with class_value as class
                class_attribute_set = attribute_set[(attribute_set[self.attribute_names[-1]] == class_value)]
                # count excluding na
                frequencies = class_attribute_set[curr_attr].value_counts(dropna=True).to_dict() # esto devuelve valor - cantidad (ej: 1: 7)
                # divide for ammount to get probability
                probabilities = {k: laplace_correction(v, sum(frequencies.values()), len(self.attribute_values[curr_attr])) for k, v in frequencies.items()}
                # add how many elements are for given conditional sub dataset
                class_attr_set_freq[curr_attr] = sum(frequencies.values())
                # store values for that attribute and class
                attr_dict[curr_attr] = frequencies
                attr_prob[curr_attr] = probabilities
            # store values of all attributes for class
            cond_frequencies[class_value] = attr_dict
            cond_probabilities[class_value] = attr_prob
            # total elements for conditional sub data sets
            total_class_attr_set_freq[class_value] = class_attr_set_freq
        self.class_attr_set_freq = total_class_attr_set_freq
        # conditional frequency map
        self.cond_frequencies = cond_frequencies
        return cond_probabilities

    def get_attribute_values(self):
        attribute_values = {}
        for i in range(len(self.attribute_names) - 1):
            curr_attribute = self.attribute_names[i]
            attribute_values[curr_attribute] = self.training_set[curr_attribute].unique()
        return attribute_values


class TextNaiveBayes:

    def __init__(self):
        self.classes_probabilities = None
        self.cond_frequencies = None
        self.data_count = 0
        self.classes = 0
        self.training_set = None
        self.text_column = None
        self.class_column = None
        self.words_amount_per_class = None
        self.lemmatizer = WordNetLemmatizer()

    def cond_P(self, value, given):
        total_words_in_class = self.words_amount_per_class[given]
        if value not in self.cond_frequencies[given]:
            cond_attr_frequency = 0
        else:
            cond_attr_frequency = self.cond_frequencies[given][value]
            # cond_freq tiene mapa con palabras y ocurrencias para categoria -> len es cant de palabras distintas
        return laplace_correction(cond_attr_frequency, total_words_in_class, len(self.cond_frequencies[given]))

    def P(self, class_value):
        return self.classes_probabilities[class_value]
        # return laplace_correction(self.classes_frequencies[class_value], self.data_count, self.classes)

    def calculate_total_probability(self, test_words):
        accum = 0
        for class_value in self.classes_probabilities:
            prod = 1
            for word in test_words:
                prod *= self.cond_P(value=word, given=class_value)
            prod *= self.P(class_value)
            accum += prod
        return accum

    # expects dataframe with 2 columns. text and class
    def train(self, training_set):
        self.class_column = training_set.columns[-1]
        self.text_column = training_set.columns[0]
        # to divide later to get probabilities
        self.data_count = len(training_set.index)
        # classes frequencies to dict obtained from classes column
        classes_frequencies = training_set[self.class_column].value_counts().to_dict()
        self.classes_probabilities = {k: laplace_correction(v, self.data_count, len(classes_frequencies)) for k, v in classes_frequencies.items()}
        self.classes = len(self.classes_probabilities)
        # useful for probabilty calculation since its more than rows per class
        # just initialized the structure
        self.words_amount_per_class = {k: 0 for k in self.classes_probabilities}
        self.cond_frequencies = self.calculate_cond_table()

    def test(self, test_set):
        results = {}
        for i in range(len(test_set)):
            test_elem = test_set.iloc[i, :]
            probabilities = {}
            test_words = self.filter_title(title=test_elem[self.text_column])
            # denominador comun a clases
            total_probability_values = self.calculate_total_probability(test_words)
            for class_value in self.classes_probabilities:
                prod = 1
                for word in test_words:
                    prod *= self.cond_P(value=word, given=class_value)
                prod *= self.P(class_value)
                arg = prod / total_probability_values
                probabilities[class_value] = arg
            results[i] = probabilities
        return results

    def calculate_cond_table(self):
        cond_table = {}
        # already have all classes without repetition here
        for class_value in self.classes_probabilities:
            # keep rows with class_value as class
            class_attribute_set = self.training_set[(self.training_set[self.class_column] == class_value)]
            cond_table[class_value] = self.filter_words(class_attribute_set)
        return cond_table

    def filter_words(self, df):
        class_dictionary = {}
        for i in range(df.index):
            filtered_words = self.filter_title(title=df.iloc[i, 0])
            self.aggregate(words=filtered_words, class_value=df.iloc[i, 1], class_dictionary=class_dictionary,)
        return class_dictionary

    def aggregate(self, words, class_value, class_dictionary):
        for word in words:
            self.words_amount_per_class[class_value] += 1
            if word not in class_dictionary:
                class_dictionary[word] = 1
            else:
                class_dictionary[word] += 1

    def filter_title(self, title):
        words = title.split()
        ret = []
        for word in words:
            stripped_word = word.strip(",.?¡¿!\'\"()[]{}-+%#$/:;_")
            if not stripped_word.isnumeric():
                stripped_word = stripped_word.lower()
                lem = self.lemmatizer.lemmatize(word=stripped_word, pos='v')
                if lem not in stopwords.words('spanish'):
                    ret.append(lem)
        return ret

