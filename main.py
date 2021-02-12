import csv
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# SET PARAMETERS
data_split = 0.2  # percentage of validation
vocab_size = 10000
embedding_dim = 16
max_length = 20
trunc_type = 'post'
oov_tok = 'OOV'

CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'data')

# LOAD DATA FROM CSV
reviews = []
titles = []
stars = []
stopwords = stopwords.words('english')

with open(os.path.join(DATA_PATH,'train.csv','r')) as csvfile:
    datastore = csv.reader(csvfile, delimiter=',')
    next(datastore)  # skip header
    for row in datastore:
        review_tokens = word_tokenize(row[2])
        title_tokens = word_tokenize(row[1])
        review_filtered = [w for w in review_tokens if w not in stopwords]
        title_filtered = [w for w in title_tokens if w not in stopwords]
        reviews.append(review_filtered)
        titles.append(title_filtered)
        stars.append(row[0])

