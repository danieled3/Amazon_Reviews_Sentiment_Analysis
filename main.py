import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import os
import random
import string

# SET PARAMETERS
data_split = 0.1  # percentage of validation
vocab_size = 100000  # max vocab size to use
embedding_dim = 16  # embedding dimension for each word
max_length = 10  # number of words to consider for each title
pad_type = 'post'
trunc_type = 'post'
oov_tok = 'OOV'  # token to use for rare words out of dictionary
rows_for_training = 100000 # load less data to speed up loading and model training for hyperparameters tuning
rows_for_test = 20000 # load less data to speed up loading

# FUNCTION TO LOAD DATA
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
            if (limit_rows!=None and counter>=limit_rows) : #lazy evaluation
                break
            counter += 1
    return (titles, reviews, stars)


# SET DATA PATH
CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'data')

# LOAD TRAIN DATA FROM CSV
titles, reviews, stars = load_amazon_data(os.path.join(DATA_PATH, 'train.csv'),
                                 title_index=1,
                                 review_index=2,
                                 star_index=0,
                                 limit_rows=rows_for_training)
# LOAD TEST DATA FROM CSV
test_titles, test_reviews, test_stars = load_amazon_data(os.path.join(DATA_PATH, 'test.csv'),
                                           title_index=1,
                                           review_index=2,
                                           star_index=0,
                                           limit_rows=rows_for_test)

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

# TRANSFORM INTO NUMPY ARRAY TO WORK WITH TENSORFLOW 2.x
training_padded = np.array(training_padded)
validation_padded = np.array(validation_padded)
test_padded = np.array(test_padded)

# TRANSFORM LABELS INTO VECTOR (i.e. 1->[1,0,0,0,0] etc.)
training_stars_diff = [int(x) - 1 for x in training_stars]
validation_stars_diff = [int(x) - 1 for x in validation_stars]
test_stars_diff = [int(x) - 1 for x in test_stars]
training_stars_diff = np.array(training_stars_diff)
validation_stars_diff = np.array(validation_stars_diff)
test_stars_diff = np.array(test_stars_diff)

# EARLY STOPPING CALLBACK
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# CHECKPOINT CALLBACK (CLASSIFICATION LSTM)
checkpoint_class_lstm_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'model_class_lstm.h5'),
                                                   save_best_only=True)

# CREATE LSTM MODEL (CLASSIFICATION LSTM)
model_class_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # add return_sequences=True in the previous layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model_class_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_class_lstm.summary()

# TRAIN MODEL (CLASSIFICATION LSTM)
history = model_class_lstm.fit(training_padded,
                    tf.keras.utils.to_categorical(training_stars_diff),
                    epochs=50,
                    validation_data=(validation_padded, tf.keras.utils.to_categorical(validation_stars_diff)),
                    callbacks=[checkpoint_class_lstm_cb, early_stopping_cb]
                    )


# CREATE CUSTOM METRICS
# Round the output of model and show the percentage of correct predictions
def ordinal_accuracy(y_true, y_pred):
    return K.cast(K.equal(y_true,
                          K.round(y_pred)),
                  K.floatx())

# CHECKPOINT CALLBACK (REGRESSION LSTM)
checkpoint_reg_lstm_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'model_reg_lstm.h5'),
                                                   save_best_only=True)
# CREATE LSTM MODEL (REGRESSION LSTM)
model_reg_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # add return_sequences=True in the previous layer
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Lambda(lambda x: (x * 5.0) - 0.5)  # in the accuracy built-in metrics round(y_pred)=y_real
])

model_reg_lstm.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),  # lr<1e-1 to converge
              metrics=[ordinal_accuracy, 'mae']  # use custom metric
              )
model_reg_lstm.summary()

# TRAIN MODEL (REGRESSION LSTM)
history = model_reg_lstm.fit(training_padded,
                    training_stars_diff,
                    epochs=50,
                    validation_data=(validation_padded, validation_stars_diff),
                    callbacks=[checkpoint_reg_lstm_cb, early_stopping_cb]
                    )

# CHECKPOINT CALLBACK (CLASSIFICATION LSTM)
checkpoint_reg_conv_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'model_reg_conv.h5'),
                                                   save_best_only=True)

# CREATE LSTM MODEL (REGRESSION CONV)
model_reg_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Lambda(lambda x: (x * 5.0) - 0.5)  # in the accuracy built-in metrics round(y_pred)=y_real
])

model_reg_conv.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),  # lr<1e-1 to converge
              metrics=[ordinal_accuracy, 'mae']  # use custom metric
              )
model_reg_conv.summary()

# TRAIN MODEL (REGRESSION CONV)
history = model_reg_conv.fit(training_padded,
                    training_stars_diff,
                    epochs=50,
                    validation_data=(validation_padded, validation_stars_diff),
                    callbacks=[checkpoint_reg_conv_cb, early_stopping_cb]
                    )




# EVALUATE PERFORMANCES ON TEST SET
model.evaluate(test_padded, test_stars_diff)

# PRINT TRAINING-VALIDATION LOSS AND ACCURACY
acc = history.history['mae']
val_acc = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training MAE')
plt.plot(epochs, val_acc, 'b', label='Validation MAE')
plt.title('Training and validation MAE')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
