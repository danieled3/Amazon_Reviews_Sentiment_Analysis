''' Load shuffled and balanced dataset with title, full reviews and stars evaluation of Amazon products.
Then train and save
- a classification model with a LSTM layer
- a regression model with a LSTM layer
- a regression model with 1D Convolutional layer
to predict stars evaluation from review title.

In the end it evaluates performances of models by considering confusion matrixes.
Then compare them with the performance of human classifiers.
The classifications made by humans were produced by AWS Ground Truth'''

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, mean_absolute_error
import os
import random
import my_utils

# SET PARAMETERS
data_split = 0.1  # percentage of validation
vocab_size = 100000  # max vocab size to use
embedding_dim = 64  # embedding dimension for each word
max_length = 20  # number of words to consider for each title
pad_type = 'post'
trunc_type = 'post'
oov_tok = 'OOV'  # token to use for rare words out of dictionary
rows_for_training = 50000  # load less data to speed up loading and model training for hyperparameters tuning
rows_for_test = 5000  # load less data to speed up loading

# SET DATA PATH
CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'data')

# LOAD TRAIN DATA FROM CSV
titles, reviews, stars = my_utils.load_amazon_data(os.path.join(DATA_PATH, 'train.csv'),
                                                   title_index=1,
                                                   review_index=2,
                                                   star_index=0,
                                                   limit_rows=rows_for_training)
# LOAD TEST DATA FROM CSV
test_titles, test_reviews, test_stars = my_utils.load_amazon_data(os.path.join(DATA_PATH, 'test.csv'),
                                                                  title_index=1,
                                                                  review_index=2,
                                                                  star_index=0,
                                                                  limit_rows=rows_for_test)

# LOAD HUMAN CLASSIFIED DATA (first 100 rows of test set)
human_stars = []

with open('data/test_100_human_classification.txt', 'r') as f:
    for line in f.readlines():
        human_stars = human_stars + [int(line[0])]
human_stars = np.array(human_stars)

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

# TRANSFORM LABELS IN POSITION INDEX (i.e. subtract 1)
training_stars_diff = [int(x) - 1 for x in training_stars]
validation_stars_diff = [int(x) - 1 for x in validation_stars]
test_stars_diff = [int(x) - 1 for x in test_stars]
training_stars_diff = np.array(training_stars_diff)
validation_stars_diff = np.array(validation_stars_diff)
test_stars_diff = np.array(test_stars_diff)

# EARLY STOPPING CALLBACK
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# CHECKPOINT CALLBACK (CLASSIFICATION LSTM)
checkpoint_class_lstm_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'models/model_class_lstm.h5'),
                                                              save_best_only=True)

# CREATE LSTM MODEL (CLASSIFICATION LSTM)
model_class_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # to add return_sequences=True in the previous layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax')
])

model_class_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_class_lstm.summary()

# TRAIN MODEL (CLASSIFICATION LSTM)
model_class_lstm.fit(training_padded,
                     tf.keras.utils.to_categorical(training_stars_diff),
                     epochs=50,
                     validation_data=(
                         validation_padded, tf.keras.utils.to_categorical(validation_stars_diff)),
                     callbacks=[checkpoint_class_lstm_cb, early_stopping_cb]
                     )


# CREATE CUSTOM METRICS
# Round the output of model and show the percentage of correct predictions
def ordinal_accuracy(y_true, y_pred):
    return K.cast(K.equal(y_true,
                          K.round(y_pred)
                          ),
                  K.floatx()
                  )


# CHECKPOINT CALLBACK (REGRESSION LSTM)
checkpoint_reg_lstm_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'models/model_reg_lstm.h5'),
                                                            save_best_only=True)
# CREATE LSTM MODEL (REGRESSION LSTM)
model_reg_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),  # add return_sequences=True in the previous layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Lambda(lambda x: (x * 5.0) - 0.5)  # in the accuracy built-in metrics round(y_pred)=y_real
])

model_reg_lstm.compile(loss=tf.keras.losses.Huber(),
                       optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),  # lr<1e-1 to converge
                       metrics=[ordinal_accuracy, 'mae']  # use custom metric
                       )
model_reg_lstm.summary()

# TRAIN MODEL (REGRESSION LSTM)
model_reg_lstm.fit(training_padded,
                   training_stars_diff,
                   epochs=50,
                   validation_data=(validation_padded, validation_stars_diff),
                   callbacks=[checkpoint_reg_lstm_cb, early_stopping_cb]
                   )

# CHECKPOINT CALLBACK (CLASSIFICATION LSTM)
checkpoint_reg_conv_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENT_PATH, 'models/model_reg_conv.h5'),
                                                            save_best_only=True)

# CREATE LSTM MODEL (REGRESSION CONV)
model_reg_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Lambda(lambda x: (x * 5.0) - 0.5)  # in the accuracy built-in metrics round(y_pred)=y_real
])

model_reg_conv.compile(loss=tf.keras.losses.Huber(),
                       optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),  # lr<1e-1 to converge
                       metrics=[ordinal_accuracy, 'mae']  # use custom metric
                       )
model_reg_conv.summary()

# TRAIN MODEL (REGRESSION CONV)
model_reg_conv.fit(training_padded,
                   training_stars_diff,
                   epochs=50,
                   validation_data=(validation_padded, validation_stars_diff),
                   callbacks=[checkpoint_reg_conv_cb, early_stopping_cb]
                   )

# COMPUTE CONFUSION MATRIXES
cm_class_lstm = confusion_matrix(test_stars_diff,
                                 np.argmax(model_class_lstm.predict(test_padded), axis=1)
                                 )
cm_reg_lstm = confusion_matrix(test_stars_diff,
                               np.round(model_reg_lstm.predict(test_padded)).reshape(rows_for_test).astype('int32')
                               )
cm_reg_conv = confusion_matrix(test_stars_diff,
                               np.round(model_reg_conv.predict(test_padded)).reshape(rows_for_test).astype('int32')
                               )
cm_hum = confusion_matrix(test_stars_diff[:100],
                          human_stars,
                          )

# COMPUTE MAE
mae_class_lstm = mean_absolute_error(test_stars_diff,
                                     np.argmax(model_class_lstm.predict(test_padded), axis=1)
                                     )
mae_reg_lstm = mean_absolute_error(test_stars_diff,
                                   np.round(model_reg_lstm.predict(test_padded)).reshape(rows_for_test).astype('int32')
                                   )
mae_reg_conv = mean_absolute_error(test_stars_diff,
                                   np.round(model_reg_conv.predict(test_padded)).reshape(rows_for_test).astype('int32')
                                   )
mae_hum = mean_absolute_error(test_stars_diff[:100],
                              human_stars,
                              )
# PRINT CONFUSION MATRIXES
my_utils.make_confusion_matrix(cm_class_lstm,
                               figsize=(5, 5),
                               cbar=False,
                               title='Classification Model LSTM',
                               percent=False,
                               cmap='Blues',
                               other_labels="Mean Absolute Error={:0.3f}".format(mae_class_lstm))
my_utils.make_confusion_matrix(cm_reg_lstm,
                               figsize=(5, 5),
                               cbar=False,
                               title='Regression Model LSTM',
                               percent=False,
                               cmap='Reds',
                               other_labels="Mean Absolute Error={:0.3f}".format(mae_reg_lstm))
my_utils.make_confusion_matrix(cm_reg_conv,
                               figsize=(5, 5),
                               cbar=False,
                               title='Regression Model with 1D Convolution',
                               percent=False,
                               cmap='Oranges',
                               other_labels="Mean Absolute Error={:0.3f}".format(mae_reg_conv))
my_utils.make_confusion_matrix(cm_hum,
                               figsize=(5, 5),
                               cbar=False,
                               title='Human Classification',
                               percent=False,
                               cmap='Greens',
                               other_labels="Mean Absolute Error={:0.3f}".format(mae_reg_conv))
