 
from keras.preprocessing.image import img_to_array 
from keras.applications import imagenet_utils 
import tensorflow as tf 
from tensorflow.keras import backend
from PIL import Image 
import numpy as np 
import flask 
import io 
import cv2

from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model,Sequential
from keras.layers import *
from keras.models import load_model
import os

# Create Flask application and initialize Keras model 
app = flask.Flask(__name__) 


global graph 
graph = tf.compat.v1.get_default_graph()
	#print("Loaded model from disk")
	 
# Every ML/DL model has a specific format 
# of taking input. Before we can predict on 
# the input image, we first need to preprocess it. 
def prepare_image(image, std,mean): 

	# Resize the image to the target dimensions 
	image = cv2.resize(image,(100,100)) 
	
	# PIL Image to Numpy array 
	
	
	# Expand the shape of an array, 
	# as required by the Model 
	image = image.reshape(1,100,100,3)
	
	# preprocess_input function is meant to 
	# adequate your image to the format the model requires 
	image = (image-mean)/std

	# return the processed image 
	return image 

# Now, we can predict the results. 

@app.route("/predict", methods =["POST"]) 
def predict(): 
	with graph.as_default():
		data = {} # dictionary to store result 
		data["success"] = False
		input_shape = (100, 100, 3)
		num_classes = 7
		weight_decay = 0.01

		model = Sequential()
		model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape))
		model.add(Conv2D(32, 3, padding='same', activation='relu'))
		model.add(MaxPool2D())
		model.add(Dropout(0.2))
		model.add(Conv2D(64, 3, padding='same', activation='relu'))
		model.add(Conv2D(64, 3, padding='same', activation='relu'))
		model.add(MaxPool2D())
		model.add(Dropout(0.2))
		model.add(Conv2D(128, 3, padding='same', activation='relu'))
		model.add(Conv2D(128, 3, padding='same', activation='relu'))
		model.add(MaxPool2D())
		model.add(Dropout(0.2))
		model.add(GlobalAveragePooling2D())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(7, activation='softmax'))

		model.compile(optimizer='adam',
				  loss='binary_crossentropy',
			  	  metrics=['accuracy'])

		model1 = load_model('model_small_new.h5')

	# Check if image was properly sent to our endpoint 
		if flask.request.method == "POST": 
			if flask.request.files.get("image"): 
				image = flask.request.files["image"].read() 
				image = Image.open(io.BytesIO(image)) 
				image = np.array(image)
				std=46.6305619048
				mean=159.81635866
			# Resize it to required pixels 
			# (required input dimensions for ResNet) 
				image = prepare_image(image,std,mean) 
			
		# Predict ! global preds, results 
			
				preds = model1.predict(image) 
				cate = int(np.argmax(preds[0]))
				prob = float(preds[0][cate])
				if prob<0.5:
					cate=7
				

				data["success"] = True
			
				dat={
				"category": cate,
				"probability": prob
				}
			



	# return JSON response 
		return flask.jsonify(dat) 



if __name__ == "__main__": 
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))  
	app.run(host='0.0.0.0', threaded=True,debug=True) 
