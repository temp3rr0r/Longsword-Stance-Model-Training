# Longsword Stance Model Training #

Longsword Stance Model Training: Deep Learning model & training python scripts. The model is genenerated with Keras, as a multivariate Bidirectional Long-Short Term Memory (LSTM) network, for classification of longsword movement gestures. Softmax is being used as the activation function and sparse-categorical cross-entropy on the final dense layer.

The trained model weights & model structruture are being stored to physical drive as json and hdf5 files. They can be later be restored for predictions with minimal execution time: ~2.5 millisecond for 1-4 rows x 12 features. 

## Technologies
- Deep Learning
- Bidirection Long-Short Term Memory (LSTM)

## Hardware
- Nvidia Jetson TX2

## SDKs & Libraries
- keras
- numpy
- h5py
- csv, json