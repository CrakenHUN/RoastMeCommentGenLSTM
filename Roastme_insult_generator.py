from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import model_from_json
import numpy as np
import sys
import random
import io
import os


def saveModel(modelfileName, weightsfileName):
    # Saving to file
    print("Saving model...")
    model_json = model.to_json()
    with open(modelfileName + ".json", "w") as json_file:
        json_file.write(model_json + ".h5")

    # serialize weights to HDF5
    model.save_weights(weightsfileName)
    print("Saved model to disk")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('***** Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(filtered_text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('***** diversity:', diversity)

        generated = ''
        sentence = filtered_text[start_index: start_index + maxlen]
        generated += sentence
        print('***** Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indicies[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indicies_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        saveModel("model_" + str(epoch), "weights_"+str(epoch))

if sys.argv[1] == "-help":
    print()
    print("-help: this menu")
    print("-train: train the model")
    print("[model.json file path] [weights.h5 file path] [output length]: run prediction (seed from training data)")
    print()
    quit()

max_text_length = 500000
maxlen = 40 #40 for SimpleBot 80 for SmarterBot

text = open('RoastMe.txt', 'r', encoding='utf-8').read().lower()

print(f'len(text): {len(text)}')

# Filter text
chars_to_keep = "abcdefghijklmnopqrstuvwxyz \n’.?!-,()[]:0123456789'"
filtered_text = ''.join(c for c in text if c in chars_to_keep)

# Cut the text short to our max data length
filtered_text = filtered_text[:max_text_length]

print(f'len(filtered_text): {len(filtered_text)}')

chars = sorted(list(set(filtered_text)))
print(f'len(chars): {len(chars)}')

char_indicies = dict((c, i) for i, c in enumerate(chars))
indicies_char = dict((i, c) for i, c in enumerate(chars))

# print(char_indicies)
# print(indicies_char) #~6520 different characters without filtering in all the txt-s combined

# we split it into sentences, with the next incoming characters
step = 3
sentences = []
next_chars = []

for i in range(0, len(filtered_text) - maxlen, step):
    sentences.append(filtered_text[i: i + maxlen])
    next_chars.append(filtered_text[i+maxlen])

print(f'len(sentences): {len(sentences)}')

print('\nSample:')
i = random.randint(0, len(sentences))
print(f'{i}th sentence: {sentences[i]}')
print(f'Next char: {next_chars[i]}')

if sys.argv[1] == "-train":
    # numpying the heck out of these
    print("Vectorizing...")
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indicies[char]] = 1
        y[i, char_indicies[next_chars[i]]] = 1

    # print("\nnumpy-ed:")
    # print(f'x[i]:\n{x[i]}\n')
    # print(f'y[i]:\n{y[i]}')

    # Let's do the network
    model = Sequential()
    model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars), activation='softmax'))

    epochs = 40

    optimizer = RMSprop(lr=0.01, decay=1e-2/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # Helper functions
    # Stolen from keras LSTM notebook
    # https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
    #(linked in description)
    # Call the on_epoch_end callback at the end of every epoch
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    # Actually train it
    print("Training...")
    model.fit(x, y,
              batch_size=128,
              epochs=epochs,
              callbacks=[print_callback])


    saveModel("finalModel", "finalWeights")
else:
    # Loading model
    # load json and create model
    json_file = open(sys.argv[1], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(sys.argv[2])
    print("Loaded model from disk")


    print()
    print('***** Generating text...')

    start_index = random.randint(0, len(filtered_text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('***** diversity:', diversity)

        generated = ''
        sentence = filtered_text[start_index: start_index + maxlen]
        generated += sentence
        print('\n\n***** Generating with seed: "' + sentence + '"\n\n')
        sys.stdout.write(generated)

        for i in range(int(sys.argv[3])):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indicies[char]] = 1.

            preds = loaded_model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indicies_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
