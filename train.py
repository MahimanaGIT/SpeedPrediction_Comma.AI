from __future__ import print_function
import cv2.cv2 as cv2
import tensorflow as tf
import tensorflow.keras as keras
import glob
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Flatten
import sklearn
from tensorflow.keras.layers import Embedding
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


PATH_TO_MODEL = "speed_recognition.PB"
PATH_TO_TRAIN_DIR = "data/images/train/"
PATH_TO_TRAIN_OUTPUT_DATASET = "data/train.txt"

def dataGenerator(filenames, groundTruth, shuffle=True, batchSize=10, targetSize=(250,250), dataset="train"):
	while(True):	
		noOfSample = len(filenames)
		noOfSteps = noOfSample // batchSize
		for i in range (0,noOfSteps):
			images=[]
			outputs=[]
			# print(i*batchSize)
			for j in range (0,batchSize):
				# print(i*batchSize + j)
				image = cv2.imread(filenames[i*batchSize + j]) 
				image = cv2.resize(image, targetSize)
				output = groundTruth[i*batchSize + j]

				images.append(image)
				outputs.append(output)

				lr_image = np.fliplr(image)
				images.append(lr_image)
				outputs.append(output)
			images = np.array(images)
			outputs = np.array(outputs)
			if(shuffle):
				images, outputs = sklearn.utils.shuffle(images, outputs)
			yield images, outputs

class CustomCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(epoch==0):
			self.val_loss=logs.get('val_loss')
		elif(logs.get('val_loss')<self.val_loss):
			print("\nSaving Model for minimum validation loss\n")
			self.val_loss=logs.get('val_loss')
			model.save('data/trained_model')

def make_model(inputShape=(250,250)):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size=(5,5), strides=(2,2), input_shape = (inputShape[1], inputShape[0], 3), activation='elu', kernel_initializer = 'he_normal'))
	model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=(5,5), strides=(2,2), kernel_initializer = 'he_normal', activation='elu'))
	model.add(tf.keras.layers.Conv2D(filters = 48, kernel_size=(5,5), strides=(2,2), kernel_initializer = 'he_normal', activation='elu'))
	model.add(tf.keras.layers.Dropout(0.3))
	model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), strides=(1,1), kernel_initializer = 'he_normal', activation='elu'))
	model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size=(3,3), strides=(1,1), kernel_initializer = 'he_normal', padding = 'valid', activation='elu'))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, kernel_initializer = 'he_normal', activation='elu'))
	model.add(tf.keras.layers.Dense(64, kernel_initializer = 'he_normal', activation='elu'))
	model.add(tf.keras.layers.Dense(16, kernel_initializer = 'he_normal', activation='elu'))
	model.add(tf.keras.layers.Dense(1, kernel_initializer = 'he_normal'))

	adam=tf.keras.optimizers.Adam(learning_rate=1e-4)
	model.compile(loss='mse',optimizer='adam')
	return model


if __name__ == '__main__':
	split=0.9
	targetSize=(240, 320)
	EPOCH = 20
	batch_size = 128

	filenames_original = [img for img in glob.glob(PATH_TO_TRAIN_DIR + "*.jpg")]
	filenames_original.sort() # ADD THIS LINE

	groundTruth_original = np.loadtxt(PATH_TO_TRAIN_OUTPUT_DATASET)
	groundTruth_original = groundTruth_original[1:]

	filenames_original, groundTruth_original = sklearn.utils.shuffle(filenames_original, groundTruth_original)

	length = len(filenames_original)
	train_samples=int(length*split)

	filenames_train=filenames_original[:train_samples]
	groundTruth_train=groundTruth_original[:train_samples]
	filenames_valid=filenames_original[train_samples:]
	groundTruth_valid=groundTruth_original[train_samples:]
	
	trainDataGen = dataGenerator(filenames_train, groundTruth_train, shuffle=True, batchSize=batch_size, targetSize=targetSize, dataset="train")
	validDataGen = dataGenerator(filenames_valid, groundTruth_valid, shuffle=True, batchSize=32, targetSize=targetSize, dataset="valid")

	log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

	custom_callback = CustomCallback()
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	early_stopping_callback =tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	callbacks=[tensorboard_callback, early_stopping_callback, custom_callback]

	# model = tf.keras.models.load_model(PATH_TO_MODEL)
	model = make_model(inputShape=targetSize)
	history = model.fit(x=trainDataGen, 
		steps_per_epoch=len(filenames_train)//batch_size, 
		validation_data=validDataGen,
		validation_steps=len(filenames_valid)//32,
		shuffle=True ,
		epochs=EPOCH,
		callbacks=callbacks)

	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(loss))

	import matplotlib.pyplot as plt

	plt.plot(epochs, loss, 'r', label='Training Loss')
	plt.plot(epochs, val_loss, 'b', label='Validation Loss')
	plt.title('Training and Validation Loss')
	plt.legend(loc=0)
	plt.figure()

	plt.show()
	model.save('data/trained_model')

	# TODO: 
	# 1. Save model 
	# 2. Save model after 10 epochs or 10 minutes

	print(model.summary())