from __future__ import print_function
import numpy as np
from keras.models import model_from_json

#json_file = open('bidirectionalClassLstmLongswordModel.json', 'r')
json_file = open('bidirectionalRetrainingLstmLongswordModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("bidirectionalClassLstmLongswordModelWeights.h5")
loaded_model.load_weights("bidirectionalRetrainingLstmLongswordModelWeights.h5")

#x_test = np.array([[ 0, 8370, 0,10747, 0, 9541, 0, 240, 0, 456, 0, 27]])
x_test = np.array([[ 0,0,16493,852,0,193,1793,843,0,0,107,0]]) # Expected Class: 4
prediction = loaded_model.predict(x_test)
predictionArgMax = np.argmax(prediction, axis=1)
print('Predicted Class: ', predictionArgMax[0], ', Expected Class: 4')
print('Confidence: ', prediction[0, predictionArgMax[0]])