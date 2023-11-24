import random
import ssl
from encodings.utf_8 import decode

import numpy as np
import tensorflow as tf
import keras
from keras import models
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file("shakespeare.txt", 'http://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read(). decode('utf-8').lower()

#choosing text from the script
text = text[300000:800000]

#make set of characters that are present in the text
characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3
'''
#feature
sentences = []

#target variable
next_character = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i:i+SEQ_LENGTH])
        next_character.append(text[i+SEQ_LENGTH])

#ATTEMPT TO CONVERT THE TRAINING STRING DATA INTO NUMERICAL DATA USING NUMPY
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

#REMAINING
#we index all the sentences
for i, sentences in enumerate(sentences):

    #we index all the characters in the sentences
  for t, character in enumerate(sentences):
      #sentence number i at position number t occurs at character number (a,b,c,d anything})
          x[i, t, char_to_index[character]] = 1
          y[i, char_to_index[next_character[i]]] = 1

#DATA PREPARATION ENDING HERE
#NOW WE BUILD THE NEURALNINE
# MODEL TRAINING
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=RMSprop(learning_rate=0.01))
# model.fit(x, y, batch_size=256, epochs=4)
# model.save('textgenerator.model')
'''
model = tf.keras.models.load_model('textgenerator.model')


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text (length, temperature):
    start_index = random.randint(0,len(text) - SEQ_LENGTH-1)
    #the text that we are gonna end up with
    generated = ''
    #starting sentence
    sentence = text[start_index:start_index+SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1,SEQ_LENGTH,len(characters)))
        for t,character in enumerate(sentence):
            x[0,t, char_to_index[character]]=1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print('----------0.2---------')
print(generate_text(165, 0.2))

print('----------0.4---------')
print(generate_text(300,0.4))

print('----------0.6---------')
print(generate_text(300,0.6))

print('----------0.8---------')
print(generate_text(300,0.8))

print('----------1---------')
print(generate_text(300,1.0))





