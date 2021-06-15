import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import string

"""Load Amazon review data, remove stopwords and punctuation, tokenize sentences 
and return text, title and stars of each review

Arguments:
    file_path(string): path of the csv file to load
    title_index(int): index of column with titles
    review_index(int): index of columns with reviews
    star_index(int): index of column with number of stars
    limit_rows: maximum number of rows to load

Return:
    titles: list of tokenize titles of Amazon reviews
    reviews: list of tokenize full text of Amazon reviews
    stars: list of number of stars of Amazon reviews"""


def load_amazon_data(file_path, title_index, review_index, star_index, limit_rows=None):
    reviews = []
    titles = []
    stars = []
    stopwords_list = stopwords.words('english')
    counter = 1
    with open(file_path, 'r', encoding="utf8") as csvfile:
        datastore = csv.reader(csvfile, delimiter=',')
        next(datastore)  # skip header
        for row in datastore:
            review_tokens = word_tokenize(row[review_index])  # tokenize sentence
            review_filtered = [w for w in review_tokens if w not in stopwords_list and w not in string.punctuation]
            reviews.append(review_filtered)
            title_tokens = word_tokenize(row[title_index])  # tokenize title
            title_filtered = [w for w in title_tokens if w not in stopwords_list and w not in string.punctuation]
            titles.append(title_filtered)
            stars.append(row[star_index])
            if limit_rows is not None and counter >= limit_rows:  # lazy evaluation
                break
            counter += 1
    return titles, reviews, stars


'''
@author DTrimarchi10 https://github.com/DTrimarchi10/confusion_matrix
This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
Arguments
---------
cf:            confusion matrix to be passed in
group_names:   List of strings that represent the labels row by row to be shown in each square.
categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
count:         If True, show the raw number in the confusion matrix. Default is True.
percent:     If True, show the proportions for each category. Default is True.
cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
               Default is True.
xyticks:       If True, show x and y ticks. Default is True.
xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
other_labels:  String with other labels to add below the chart. Default is Empty string.
sum_stats:     If True, display summary statistics below the figure. Default is True.
figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
               See http://matplotlib.org/examples/color/colormaps_reference.html

title:         Title for the heatmap. Default is None.
'''


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          other_labels="",
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for _ in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\nXAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score) + "\n" + other_labels
        else:
            stats_text = "\nXAccuracy={:0.3f}".format(accuracy) + "\n" + other_labels
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
