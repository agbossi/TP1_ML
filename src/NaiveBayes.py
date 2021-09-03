# missing data :
# https://datascience.stackexchange.com/questions/3711/how-does-the-naive-bayes-classifier-handle-missing-data-in-training
# potencially usefull tips:
# https://machinelearningmastery.com/better-naive-bayes/

def get_max_class(arg, class_value, arg_max, classification):
    if arg < arg_max:
        return classification
    else:
        return class_value


# TODO: ajustar con correccion de laplace
class DiscreteNaiveBayes:

    LIKES = 1
    DISLIKES = 0

    def __init__(self):
        self.classes_frequencies = None
        self.cond_probabilities = None
        self.data_count = 0
        self.classes = 0
        self.training_set = None
        self.attribute_names = None
        # TODO: poner otro mapa de mapa de mapas para el total de elems para una pcond, asi cuando quiera
        # calcular una probabilidad a partir de valor que no esta en estructura frequencias tengo el total
        # para correccion de laplace
        self.class_attr_set_freq = None

    def cond_P(self, attribute, value, given):
        return self.cond_probabilities[given][attribute][value]

    def P(self, class_value):
        return self.laplace_correction(self.classes_frequencies[class_value], self.data_count)

    def laplace_correction(self, occurrences, total):
        return (occurrences + 1) / (total + self.classes)

    def calculate_total_probability(self, test_elem):
        accum = 0
        for class_value in self.classes_frequencies:
            prod = 1
            for i in range(len(self.attribute_names) - 1):
                curr_attr = self.attribute_names[i]
                prod *= self.cond_P(attribute=curr_attr, value=test_elem[i], given=class_value)
            prod *= self.classes_frequencies[class_value]
            accum += prod
        return accum

    def train(self, training_set):
        self.training_set = training_set
        # store columns for easier counting and access
        self.attribute_names = self.training_set.columns
        # classes frequencies to dict obtained from classes column
        self.classes_frequencies = training_set[self.attribute_names[-1]].value_counts().to_dict()
        self.classes = len(self.classes_frequencies)
        # to divide later to get probabilities
        self.data_count = len(training_set.index)
        self.cond_probabilities = self.calculate_cond_table()

    def test(self, test_set):
        results = {}
        for i in range(len(test_set)):
            arg_max = 0
            classification = None
            probabilities = {}
            total_probability_values = self.calculate_total_probability(test_set[i])
            for class_value in self.classes_frequencies:
                prod = 1
                for j in range(len(self.attribute_names) - 1):
                    prod *= self.cond_P(attribute=self.attribute_names[j],
                                        value=test_set.get_value(i, self.attribute_names[j]),
                                        given=class_value)
                prod *= self.P(class_value)
                arg = prod / total_probability_values
                probabilities[class_value] = arg
                classification = get_max_class(arg, class_value, arg_max, classification)
            results[i] = probabilities

    def save_prod(self, attr_dict, cond_class_frequencies):
        for i in range(len(self.attribute_names) - 1):
            curr_attr = self.attribute_names[i]
            # if the frequency of this conditional probability is 0
            if attr_dict[curr_attr] is None:
                attr_dict[curr_attr] = self.laplace_correction(0, cond_class_frequencies[i])
        return

    def calculate_cond_table(self):
        cond_probabilities = {}

        # already have all classes without repetition here
        for class_value in self.classes_frequencies:
            # to store key: possible values for given attribute,
            # value: count of that value for given class
            attr_dict = {}
            # list of total values for conditional subsets for each attr
            class_attr_set_freq = []
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
                probabilities = {k: self.laplace_correction(v, len(class_attribute_set)) for k, v in frequencies.items()}
                # add how many elements are for given conditional sub dataset
                class_attr_set_freq.append(len(class_attribute_set))
                # store values for that attribute and class
                attr_dict[curr_attr] = probabilities
            self.save_prod(attr_dict, class_attr_set_freq)
            # store values of all attributes for class
            cond_probabilities[class_value] = attr_dict
        return cond_probabilities


class TextNaiveBayes:

    def __init__(self):
        self.classes_frequencies = None
        self.cond_probabilities = None
        self.data_count = 0
        self.classes = 0
        self.training_set = None
        self.text_column = None
        self.class_column = None
        # set of words to ignore in processing
        self.words_blacklist = None
        self.words_amount_per_class = None

    # expects dataframe with 2 columns. text and class
    def train(self, training_set, words_blacklist):
        self.class_column = training_set.columns[-1]
        self.text_column = training_set.columns[0]
        self.words_blacklist = words_blacklist
        # classes frequencies to dict obtained from classes column
        self.classes_frequencies = training_set[self.class_column].value_counts().to_dict()
        self.classes = len(self.classes_frequencies)
        # to divide later to get probabilities
        self.data_count = len(training_set.index)
        # useful for probabilty calculation since its more than rows per class
        self.words_amount_per_class = {k: 0 for k in self.classes_frequencies}
        self.cond_probabilities = self.calculate_cond_table()

    def P(self, class_value):
        return self.laplace_correction(self.classes_frequencies[class_value], self.data_count)

    def laplace_correction(self, occurrences, total):
        return (occurrences + 1) / (total + self.classes)

    def filter_title(self, title, class_dictionary, class_value):
        words = title.split()
        for word in words:
            if word.strip(",.?¡¿!\'\"()[]{}-+%#$/:;_").lower() not in self.words_blacklist and not word.isnumeric():
                self.words_amount_per_class[class_value] += 1
                if class_dictionary[word] is None:
                    class_dictionary[word] = 1
                else:
                    class_dictionary[word] += 1

    def filter_words(self, df):
        class_dictionary = {}
        for i in df.index:
            self.filter_title(df.iloc[i, 0], class_dictionary, df.iloc[i, 1])
        return class_dictionary

#    def save_prod(self, attr_dict):
#        for i in range(len(self.attribute_names) - 1):
#            curr_attr = self.attribute_names[i]
#            # if the frequency of this conditional probability is 0
#            if attr_dict[curr_attr] is None:
#                attr_dict[curr_attr] = self.laplace_correction(0, cond_class_frequencies[i])
#        return

    def calculate_cond_table(self):
        cond_table = {}
        # already have all classes without repetition here
        for class_value in self.classes_frequencies:
            # keep rows with class_value as class
            class_attribute_set = self.training_set[(self.training_set[self.class_column] == class_value)]
            cond_table[class_value] = self.filter_words(class_attribute_set)

        return cond_table

    # def test(self, test_set):
        