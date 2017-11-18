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
x_test = np.array([[ 0,0,0,0,0,558,0,0,1244,0,91,0,0,0,0,0,0,0,5038,1860,159,248,0,16212,3172,0,653,0,1524,65389,177,65461,0,127]]) # Expected Class: 5
prediction = loaded_model.predict(x_test)
predictionArgMax = np.argmax(prediction, axis=1)
print('Predicted Class: ', predictionArgMax[0], ', Expected Class: 5')
print('Confidence: ', prediction[0, predictionArgMax[0]])