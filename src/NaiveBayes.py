# missing data :
# https://datascience.stackexchange.com/questions/3711/how-does-the-naive-bayes-classifier-handle-missing-data-in-training
# potencially usefull tips:
# https://machinelearningmastery.com/better-naive-bayes/

def get_max_class(arg, class_value, arg_max, classification):
    if arg < arg_max:
        return classification
    else:
        return class_value


class CategoricalNaiveBayes:

    LIKES = 1
    DISLIKES = 0

    def __init__(self):
        self.classes_frequencies = None
        self.cond_probabilities = None
        self.data_count = 0
        self.classes = 0
        self.training_set = None
        self.attribute_names = None

    def cond_P(self, attribute, value, given):
        return self.cond_probabilities[given][attribute][value]

    def P(self, class_value):
        return self.classes_frequencies[class_value] / self.data_count

    def calculate_total_probability(self):
        for class_value in self.classes_frequencies:
            for i in range(len(self.attribute_names) - 1):
                prod = 1
                curr_attr = self.attribute_names[i]
                for attribute_value in self.cond_probabilities[class_value][curr_attr]:
                    prod *= self.cond_P(curr_attr, attribute_value, given=class_value)
        return 0

    def train(self, training_set):
        self.training_set = training_set
        # classes frequencies to dict obtained from classes column
        self.classes_frequencies = training_set[:, -1:].value_counts().to_dict()
        self.classes = len(self.classes_frequencies)
        # to divide later to get probabilities
        self.data_count = len(training_set.index)
        # store columns for easier counting
        self.attribute_names = self.training_set.columns()
        self.cond_probabilities = self.calculate_cond_table()

    def test(self, test_set):
        # TODO valores de probabilidad total estan como el orto
        total_probability_values = self.calculate_total_probability()
        results = {}
        for i in range(len(test_set)):
            arg_max = 0
            classification = None
            probabilities = {}
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

    def calculate_cond_table(self):
        cond_probabilities = {}

        # already have all classes without repetition here
        for class_value in self.classes_frequencies:
            # to store key: possible values for given attribute,
            # value: count of that value for given class
            attr_dict = {}
            # loop through attributes, excluding class column
            for i in range(len(self.attribute_names) - 1):
                curr_attr = self.attribute_names[i]
                # get attribute column and class column
                attribute_set = self.training_set.loc[:, [curr_attr, self.attribute_names[-1]]]
                # keep rows with class_value as class
                class_attribute_set = attribute_set[(attribute_set[self.attribute_names[-1]] == class_value)]
                # count excluding na
                frequencies = class_attribute_set.value_counts(dropna=True).to_dict()
                # divide for ammount to get probability
                probabilities = {k: v / len(class_attribute_set) for k, v in frequencies.items()}
                # store values for that attribute and class
                attr_dict[curr_attr] = probabilities
            # store values of all attributes for class
            cond_probabilities[class_value] = attr_dict
        return cond_probabilities
