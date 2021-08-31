
import enum

SKIP_LABEL_INDEX = 1


class MatrixComponents(enum.Enum):
    true_positive = 0
    false_positive = 1
    true_negative = 2
    false_negative = 3


# TODO plot heatmap
class ConfusionMatrix:
    def __init__(self, possible_classifications):
        self.matrix = [[int(0) for i in range(len(possible_classifications))] for j in
                       range(len(possible_classifications))]
        self.stats_matrix = None
        self.classifications = possible_classifications
        self.entries = 0

    def add_entry(self, real_classification, classification):
        self.matrix[real_classification][classification] += 1
        self.entries += 1

    def summarize(self):
        stats_matrix = [[int(0) for j in range(4)] for i in range(len(self.classifications))]
        for k in range(len(self.classifications)):
            curr_classification_amount = 0
            stats_matrix[k][MatrixComponents.true_positive.value] += self.matrix[k][k]
            for l in range(len(self.classifications)):
                if k != l:
                    stats_matrix[k][MatrixComponents.false_positive.value] += self.matrix[l][k]
                    stats_matrix[k][MatrixComponents.false_negative.value] += self.matrix[k][l]
                    curr_classification_amount += (self.matrix[k][l] + self.matrix[l][k] + self.matrix[k][k])
            stats_matrix[k][MatrixComponents.false_negative.value] += curr_classification_amount \
                                                                      - stats_matrix[k][
                                                                          MatrixComponents.true_positive.value] \
                                                                      - stats_matrix[k][
                                                                          MatrixComponents.false_positive.value] \
                                                                      - stats_matrix[k][
                                                                          MatrixComponents.false_negative.value]
            stats_matrix[k].insert(0, self.classifications[k].name)
        stats_matrix.insert(0, [" ", "TP", "FP", "TN", "FN"])
        self.stats_matrix = stats_matrix
        return stats_matrix

    def print_confusion_matrix(self):
        header = []
        for k in range(len(self.classifications)):
            self.matrix[k].insert(0, self.classifications[k].name)
            header.append(self.classifications[k].name)
        header.insert(0, " ")
        self.matrix.insert(0, header)
        print_m(self.matrix)

    def print_summary(self):
        print_m(self.stats_matrix)

    def get_precisions(self):
        precisions = []
        if self.stats_matrix is None:
            self.summarize()
            for i in range(len(self.classifications)):
                precision = [self.classifications[i].name, self.get_precision(i + SKIP_LABEL_INDEX)]  # paso la linea con texto
                precisions.append(precision)
        return precisions

    def get_accuracy(self, index):
        if self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value] + self.stats_matrix[index][
            SKIP_LABEL_INDEX + MatrixComponents.true_negative.value]\
                + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_negative.value]\
                + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_positive.value] == 0:
            accuracy = 0
        else:
            accuracy = ((self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value]
                          + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_negative.value])
                         / (self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value] +
                            self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_negative.value] +
                            self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_negative.value] +
                            self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_positive.value]))
        return accuracy

    def get_accuracies(self):
        if self.stats_matrix is None:
            self.summarize()
        accuracies = []
        for i in range(len(self.classifications)):
            accuracy = [self.classifications[i].name, self.get_accuracy(i + SKIP_LABEL_INDEX)]  # paso la linea con texto
            accuracies.append(accuracy)
        return accuracies

    def get_precision(self, index):
        if self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value] \
                + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_positive.value] == 0:
            precision = 0
        else:
            precision = self.stats_matrix[index][1 + MatrixComponents.true_positive.value] \
                / (self.stats_matrix[index][1 + MatrixComponents.true_positive.value]
                    + self.stats_matrix[index][1 + MatrixComponents.false_positive.value])
        return precision

    def get_recalls(self):
        if self.stats_matrix is None:
            self.summarize()
        recalls = []
        for i in range(len(self.classifications)):
            recall = [self.classifications[i].name, self.get_recall(i + SKIP_LABEL_INDEX)]  # paso la linea con texto
            recalls.append(recall)
        return recalls

    def get_recall(self, index):
        if self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value] \
                + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_negative.value] == 0:
            recall = 0
        else:
            recall = self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value] \
                     / (self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_positive.value]
                        + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_negative.value])
        return recall

    def get_f1_scores(self):
        if self.stats_matrix is None:
            self.summarize()
        f1_scores = []
        for i in range(len(self.classifications)):
            f1_score = [self.classifications[i].name, self.get_f1_score(i + SKIP_LABEL_INDEX)]  # paso la linea con texto
            f1_scores.append(f1_score)
        return f1_scores

    def get_f1_score(self, index):
        if self.get_recall(index) + self.get_precision(index) == 0:
            f1_score = 0
        else:
            f1_score = 2 * self.get_precision(index) * self.get_recall(index) / (
                    self.get_recall(index) + self.get_precision(index))
        return f1_score

    def get_fp_rates(self):
        if self.stats_matrix is None:
            self.summarize()
        fp_rates = []
        for i in range(len(self.classifications)):
            fp_rate = [self.classifications[i].name, self.get_fp_rate(i + SKIP_LABEL_INDEX)]
            fp_rates.append(fp_rate)
        return fp_rates

    def get_fp_rate(self, index):
        if self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_positive.value] \
            + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_negative.value] == 0:
            tp_rate = 0
        else:
            tp_rate = self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_positive.value] \
            / (self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.false_positive.value]
            + self.stats_matrix[index][SKIP_LABEL_INDEX + MatrixComponents.true_negative.value])
        return tp_rate

    def get_tp_rates(self):
        if self.stats_matrix is None:
            self.summarize()
        tp_rates = []
        for i in range(len(self.classifications)):
            tp_rate = [self.classifications[i].name, self.get_tp_rate(i + SKIP_LABEL_INDEX)]
            tp_rates.append(tp_rate)
        return tp_rates

    def get_tp_rate(self, index):
        return self.get_recall(index)


POSITIVE = 0
NEGATIVE = 1


class RocConfusionMatrix:

    def __init__(self, positive_class):
        self.matrix = [[0, 0],
                       [0, 0]]
        self.entries = 0
        self.stats_matrix = None
        # map 2 classes as binary classificator
        self.positive_class = positive_class

    def print_summary(self):
        print_m(self.stats_matrix)

    def add_entry(self, real_classification, classification, classification_probability, threshold):
        if real_classification == classification and classification_probability >= threshold:
            # either tp or tn
            if real_classification == self.positive_class:
                # tp
                self.matrix[POSITIVE][POSITIVE] += 1
            else:
                # tn
                self.matrix[NEGATIVE][NEGATIVE] += 1
        # if misclassified
        elif real_classification != classification:
            # either fp or fn
            if real_classification == self.positive_class:
                # fn
                self.matrix[POSITIVE][NEGATIVE] += 1
            else:
                # fp
                self.matrix[NEGATIVE][POSITIVE] += 1
        self.entries += 1

    def summarize(self):
        stats_matrix = [[" ", "TP", "FP", "TN", "FN"], [0, 0, 0, 0]]
        stats_matrix[SKIP_LABEL_INDEX][MatrixComponents.true_positive.value] = self.matrix[POSITIVE][POSITIVE]
        stats_matrix[SKIP_LABEL_INDEX][MatrixComponents.false_positive.value] = self.matrix[NEGATIVE][POSITIVE]
        stats_matrix[SKIP_LABEL_INDEX][MatrixComponents.false_negative.value] = self.matrix[POSITIVE][NEGATIVE]
        stats_matrix[SKIP_LABEL_INDEX][MatrixComponents.true_negative.value] = self.matrix[NEGATIVE][NEGATIVE]
        stats_matrix[SKIP_LABEL_INDEX].insert(0, self.positive_class)
        self.stats_matrix = stats_matrix
        return stats_matrix

    def get_precision(self):
        if self.matrix[POSITIVE][POSITIVE] + self.matrix[NEGATIVE][POSITIVE] == 0:
            precision = 0
        else:
            precision = self.matrix[POSITIVE][POSITIVE] \
                        / (self.matrix[POSITIVE][POSITIVE] + self.matrix[NEGATIVE][POSITIVE])
        return precision

    def get_accuracy(self):
        if self.matrix[POSITIVE][POSITIVE] + self.matrix[POSITIVE][NEGATIVE] \
            + self.matrix[NEGATIVE][POSITIVE] + self.matrix[NEGATIVE][NEGATIVE] == 0:
            accuracy = 0
        else:
            accuracy = self.matrix[POSITIVE][POSITIVE] + self.matrix[NEGATIVE][NEGATIVE] \
            / (self.matrix[POSITIVE][POSITIVE] + self.matrix[POSITIVE][NEGATIVE]
            + self.matrix[NEGATIVE][POSITIVE] + self.matrix[NEGATIVE][NEGATIVE])
        return accuracy

    def get_recall(self):
        if self.matrix[POSITIVE][POSITIVE] + self.matrix[POSITIVE][NEGATIVE] == 0:
            recall = 0
        else:
            recall = self.matrix[POSITIVE][POSITIVE] \
                     / (self.matrix[POSITIVE][POSITIVE] + self.matrix[POSITIVE][NEGATIVE])
        return recall

    def get_f1_score(self):
        if self.get_recall() + self.get_precision() == 0:
            f1_score = 0
        else:
            f1_score = 2 * self.get_precision() * self.get_recall() / (
                    self.get_recall() + self.get_precision())
        return f1_score

    def get_tp_rate(self):
        return self.get_recall()

    def get_fp_rate(self):
        if self.matrix[NEGATIVE][POSITIVE] + self.matrix[NEGATIVE][NEGATIVE] == 0:
            fp = 0
        else:
            fp = self.matrix[NEGATIVE][POSITIVE]\
                 / (self.matrix[NEGATIVE][POSITIVE] + self.matrix[NEGATIVE][NEGATIVE])
        return fp

    def get_roc_point(self):
        x = self.get_tp_rate()
        y = self.get_fp_rate()
        return x, y


def print_m(matrix):
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            print(matrix[i][j], end=" ")
        print('\n')
