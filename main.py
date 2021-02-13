import csv
import os
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# SET PARAMETERS
data_split = 0.1  # percentage of validation
vocab_size = 100000
embedding_dim = 16
max_length = 10  # long titles are not common
pad_type = 'post'
trunc_type = 'post'
oov_tok = 'OOV'

CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'data')

# LOAD DATA FROM CSV
# reviews = []
titles = []
stars = []
test_titles = []
test_stars = []
stopwords = stopwords.words('english')

with open(os.path.join(DATA_PATH, 'train.csv'), 'r', encoding="utf8") as csvfile:
    datastore = csv.reader(csvfile, delimiter=',')
    next(datastore)  # skip header
    for row in datastore:
        # review_tokens = word_tokenize(row[2])
        # review_filtered = [w for w in review_tokens if w not in stopwords]
        # reviews.append(review_filtered)
        title_tokens = word_tokenize(row[1])
        title_filtered = [w for w in title_tokens if w not in stopwords]
        titles.append(title_filtered)
        stars.append(row[0])

with open(os.path.join(DATA_PATH, 'test.csv'), 'r', encoding="utf8") as csvfile:
    datastore = csv.reader(csvfile, delimiter=',')
    next(datastore)  # skip header
    for row in datastore:
        # review_tokens = word_tokenize(row[2])
        # review_filtered = [w for w in review_tokens if w not in stopwords]
        # reviews.append(review_filtered)
        title_tokens = word_tokenize(row[1])
        title_filtered = [w for w in title_tokens if w not in stopwords]
        test_titles.append(title_filtered)
        test_stars.append(row[0])

# TRAINING AND VALIDATION SPLIT
random.seed(33)
data_dim = len(titles)
validation_dim = int(data_dim * data_split)
training_dim = data_dim - validation_dim

training_titles = titles[:training_dim]  # data are already shuffled
training_stars = stars[:training_dim]
validation_titles = titles[training_dim:]
validation_stars = stars[training_dim:]

# TOKENIZE
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_titles)
training_sequences = tokenizer.texts_to_sequences(training_titles)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_titles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_titles)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=pad_type, truncating=trunc_type)
