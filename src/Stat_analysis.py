#  To work with pandas data frames
import math
import os

import pandas as pd
import matplotlib.pyplot as plt

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cov.html
# https://www.geeksforgeeks.org/how-to-create-a-correlation-matrix-using-pandas/

# ======================================================================================================================
# ======================================================================================================================
#                                                  Plots
# ======================================================================================================================
# ======================================================================================================================


def boxplot(data_frame, column, sample):
    plot = data_frame.boxplot(column)
    plt.title(sample)
    plt.show()


def scatter_plot(data_frame, x_variable, y_variable, sample):
    data_frame.plot.scatter(x=x_variable, y=y_variable, title=(x_variable + " y " + y_variable + " " + sample))
    plt.show(block=True)


# plot boxplot, histogram for each variable
def plot_variables(data_frame, column, sample):
    boxplot(data_frame, column, sample)
    histogram(data_frame, column, sample)


# ======================================================================================================================
# ======================================================================================================================
#                                                  Histogram
# ======================================================================================================================
# ======================================================================================================================

# bins calculated from quantiles using Freedman-Diaconis rule
def calculate_optimal_bins(data_frame, column):
    # Computing IQR
    Q1 = data_frame[column].quantile(0.25)
    Q3 = data_frame[column].quantile(0.75)
    IQR = Q3 - Q1

    # Freedman-Diaconis rule
    bin_width = 2 * IQR / len(data_frame) ** (1. / 3)
    bins = (data_frame[column].max() - data_frame[column].min()) / bin_width
    return bins


def histogram(data_frame, column, sample):
    # Get optimal beans for variable
    bins = math.ceil(calculate_optimal_bins(data_frame, column))
    # Plot pandas histogram from dataframe with df.plot.hist (not df.hist)
    ax = data_frame[column].plot.hist(bins=bins, density=True, edgecolor='w', linewidth=0.5, label=column)

    # Save default x-axis limits for final formatting because the pandas kde
    # plot uses much wider limits which usually decreases readability
    xlim = ax.get_xlim()

    # Plot pandas KDE
    data_frame[column].plot.density(color='k', alpha=0.5, ax=ax)  # same as df['var'].plot.kde()

    # Reset x-axis limits and edit legend and add title
    ax.set_xlim(xlim)
    # ax.legend(labels=['KDE'], frameon=False)
    # ax.set_title('Pandas histogram overlaid with KDE', fontsize=14, pad=15)
    plt.title(column + " " + sample)
    plt.show()

# ======================================================================================================================
# ======================================================================================================================
#                                                   na replacers
# ======================================================================================================================
# ======================================================================================================================
# https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e


# replace na with columns mean
def replace_by_mean(data_frame, fields):
    for field in fields:
        data_frame[field] = data_frame[field].fillna(data_frame[field].mean())
    return data_frame


# replace na with columns median.
def replace_by_median(data_frame, fields):
    for field in fields:
        data_frame[field] = data_frame[field].fillna(data_frame[field].median())
    return data_frame


# replace na with mode
def replace_by_mode(data_frame, fields):
    for field in fields:
        data_frame[field] = data_frame[field].fillna(data_frame[field].mode())
    return data_frame


# replace na with most frequent value. Used for categorical variables
# value_counts sorts descending by default
def replace_by_most_frequent(data_frame, fields):
    for field in fields:
        frequencies = data_frame[field].value_counts()
        data_frame[field] = data_frame[field].fillna(frequencies.keys()[0])
    return data_frame

