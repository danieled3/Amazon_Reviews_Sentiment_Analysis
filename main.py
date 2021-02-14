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
        title_tokens = word_tokenize(row[1])  # stemming and remove capital words
        title_filtered = [w for w in title_tokens
                          if w not in stopwords and w not in string.punctuation]  # remove stopwords and punctuation
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

# TRANSFORM INTO NUMPY ARRAY TO WORK WITH TENSORFLOW 2.x
training_padded = np.array(training_padded)
validation_padded = np.array(validation_padded)
test_padded = np.array(test_padded)

training_stars_diff = [int(x) - 1 for x in training_stars]  # transform in ordinal categories 1->[1,0,0,0,0] etc.
validation_stars_diff = [int(x) - 1 for x in validation_stars]
test_stars_diff = [int(x) - 1 for x in test_stars]
training_stars_diff = np.array(training_stars_diff)
validation_stars_diff = np.array(validation_stars_diff)
test_stars_diff = np.array(test_stars_diff)

# CALLBACKS BUILDING
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'amazon_sentiment_analysis_reg_model.h5'),
                                                   save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# CREATE LSTM MODEL (CLASSIFICATION LSTM)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # add return_sequences=True in the previous layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# TRAIN MODEL (CLASSIFICATION LSTM)
history = model.fit(training_padded,
          tf.keras.utils.to_categorical(training_stars_diff),
          epochs=50,
          validation_data=(validation_padded, tf.keras.utils.to_categorical(validation_stars_diff))
          )


# CREATE CUSTOM METRICS
# Round the output of model and show the percentage of correct predictions
def ordinal_accuracy(y_true, y_pred):
    return K.cast(K.equal(y_true,
                          K.round(y_pred)),
                  K.floatx())


# CREATE LSTM MODEL (REGRESSION LSTM)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # add return_sequences=True in the previous layer
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Lambda(lambda x: (x * 5.0) - 0.5)  # in the accuracy built-in metrics round(y_pred)=y_real
])

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),  # lr<1e-1 to converge
              metrics=[ordinal_accuracy, 'mae']  # use custom metric
              )
model.summary()

# TRAIN MODEL (REGRESSION LSTM)
history = model.fit(training_padded,
          training_stars_diff,
          epochs=50,
          validation_data=(validation_padded, validation_stars_diff),
          callbacks=[checkpoint_cb, early_stopping_cb]
          )

# CREATE LSTM MODEL (REGRESSION CONV)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Lambda(lambda x: (x * 5.0) - 0.5)  # in the accuracy built-in metrics round(y_pred)=y_real
])

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),  # lr<1e-1 to converge
              metrics=[ordinal_accuracy, 'mae']  # use custom metric
              )
model.summary()

# TRAIN MODEL (REGRESSION CONV)
history = model.fit(training_padded,
          training_stars_diff,
          epochs=50,
          validation_data=(validation_padded, validation_stars_diff),
          callbacks=[checkpoint_cb, early_stopping_cb]
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