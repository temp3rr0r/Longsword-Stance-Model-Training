'''Train a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
import h5py
from numpy import array
import csv

# import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

max_features = 65536
maxlen = 17 * 2 # 6 * 2
batch_size = 4
epoch_size = 400 # 40
class_count = 7
useTest = False

print('Loading data...')
r = np.genfromtxt("longsword.csv", delimiter=',')

print ('Start Array r:')
print (r)

# TODO: add columns. If positive: 0, 32500. If negative 32500, 0. Essentially slit it into to classes for possives and negatives
r2 = np.copy(r)
r[r < 0] = 0
r2[r2 > 0] = 0
r2 *= -1
r = np.insert(r, 1, values=r2[:,1], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 2, values=r2[:,2], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 3, values=r2[:,3], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 4, values=r2[:,4], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 5, values=r2[:,5], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 6, values=r2[:,6], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 7, values=r2[:,7], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 8, values=r2[:,8], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 9, values=r2[:,9], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 10, values=r2[:,10], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 11, values=r2[:,11], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 12, values=r2[:,12], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 13, values=r2[:,13], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 14, values=r2[:,14], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 15, values=r2[:,15], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 16, values=r2[:,16], axis=1) # inser2t values befor2e column 3
r = np.insert(r, 17, values=r2[:,17], axis=1) # inser2t values befor2e column 3


r2 = np.copy(r)
#r = np.insert(r, 1, values=r2[:,2:r.shape[0]], axis=1) # inser2t values befor2e column 3

#r = np.delete(r, [0], axis=1) # Remove timestamp

#np.random.shuffle(r)
proportion = 0.15
if useTest == True:
    proportion15Percent = int(proportion * r.shape[0])
    x_validate = r[0:proportion15Percent, 1:maxlen + 1]
    y_validate = r[0:proportion15Percent, 0]
    x_test= r[proportion15Percent + 1:2 * proportion15Percent, 1:maxlen + 1]
    y_test = r[proportion15Percent + 1:2 * proportion15Percent, 0]
    x_train = r[2 * proportion15Percent + 1:len(r), 1:maxlen + 1]
    y_train = r[2 * proportion15Percent + 1:len(r), 0]

else:
    proportion30Percent = int(proportion * r.shape[0])
    x_validate = r[0:proportion30Percent, 1:maxlen + 1]
    y_validate = r[0:proportion30Percent, 0]
    # x_test = r[proportion15Percent + 1:2 * proportion15Percent, 1:maxlen + 1]
    # y_test = r[proportion15Percent + 1:2 * proportion15Percent, 0]
    x_train = r[proportion30Percent + 1:len(r), 1:maxlen + 1]
    y_train = r[proportion30Percent + 1:len(r), 0]

print ('Y validate:')
print (y_validate)

if useTest == True:
    print(len(x_train), 'train sequences (' + str((1 - 2 * proportion) * 100) + '%)')
else:
    print(len(x_train), 'train sequences (' + str((1 - proportion) * 100) + '%)')
print(len(x_validate), 'validate sequences (' + str(proportion * 100) + '%)')
if useTest == True:
    print(len(x_test), 'test sequences (' + str(proportion * 100) + '%)')

print('Pad sequences (samples x time)')
print('x_train shape:', x_train.shape)
if useTest == True:
    print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
if useTest == True:
    y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(class_count, activation='softmax')) # sigmoid
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch_size,
          validation_data=[x_validate, y_validate])

# Prediction
if useTest == True:
    print('Prediction')
    prediction = model.predict(x_test)
    print(prediction)
    print('Prediction Arg max')
    predictionArgMax = np.argmax(prediction, axis=1)
    print(predictionArgMax)
    print('Expected')
    print(y_test)

    print ('Classification Accuracy %: ', (predictionArgMax == y_test).sum() / len(y_test))

# Store model
# serialize model to JSON
model_json = model.to_json()
with open("bidirectionalRetrainingLstmLongswordModel.json", "w") as json_file:
    json_file.write(model_json)
    print("Saved weights to disk")
# serialize weights to HDF5

model.save_weights("bidirectionalRetrainingLstmLongswordModelWeights.h5")
print("Saved model to disk")


## after
# from keras.models import model_from_json
# json_file = open('bidirectionalClassLstmLongswordModel.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("bidirectionalClassLstmLongswordModelWeights.h5")
#
# x_test = np.array([[ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]])
# prediction = loaded_model.predict(x_test)
# predictionArgMax = np.argmax(prediction, axis=1)
# print('Class: ', predictionArgMax[0])
# print('Accuracy: ', prediction[0, predictionArgMax[0]])