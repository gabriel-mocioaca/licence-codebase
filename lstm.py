import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.engine.input_layer import Input
from pre_process import preprocess
import matplotlib.pyplot as plt
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

vocab_size = 5000
embedding_dim = 32
max_length = 60
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

train_dataset = pd.read_csv('corona_nlp_train.csv', header=0, delimiter=',', usecols=['OriginalTweet', 'Sentiment'])
test_dataset = pd.read_csv('corona_nlp_test.csv', header=0, delimiter=',', usecols=['OriginalTweet', 'Sentiment'])

train_features = preprocess(train_dataset)
test_features = preprocess(test_dataset)

train_labels = train_dataset.pop('Sentiment')
test_labels = test_dataset.pop('Sentiment')

train_labels=train_labels.replace({'Extremely Positive':'Positive','Extremely Negative':'Negative'})
test_labels = test_labels.replace({'Extremely Positive':'Positive','Extremely Negative':'Negative'})

le = preprocessing.LabelEncoder()
le.fit_transform(train_labels)
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_features)
vocab_size = len(tokenizer.word_index)

train_sequences = tokenizer.texts_to_sequences(train_features)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(test_features)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


training_label_seq = np.array(le.transform(train_labels), dtype=float)
validation_label_seq = np.array(le.transform(test_labels), dtype=float)

print(len(train_features))
print(len(train_labels))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, embedding_dim),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()


model.compile(loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

num_epochs = 50
early_stop=EarlyStopping(monitor='val_accuracy',patience=3)
reduceLR=ReduceLROnPlateau(monitor='val_accuracy',patience=2)

history = model.fit(train_padded, 
    training_label_seq, 
    epochs=num_epochs, 
    validation_data=(validation_padded, validation_label_seq),
    batch_size=64, 
    callbacks=[reduceLR,early_stop])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()