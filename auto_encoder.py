import os
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Add, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.models import model_from_json


def normalize(img):
	return (img/255)

def denormalize(img):
	return (img) * 255
	
def load_images(filename):
	return np.load(filename)


rain_data_path = "/dataset/rain_images.npy"
derain_data_path = "/dataset/derain_images.npy"

rain_data = normalize(load_images(rain_data_path))
print("Rain Data Loaded.")
derain_data = normalize(load_images(derain_data_path))
print("DeRain Data Loaded.")


def create_model():
	inp=Input(shape=(256,256,3))
	layer1=Conv2D(filters=16,kernel_size=4,activation="relu",padding="same")(inp)
	layer2=MaxPooling2D((2,2),padding="same")(layer1)

	layer3=Conv2D(filters=32,kernel_size=5,activation="relu",padding="same")(layer2)
	layer4=MaxPooling2D((2,2),padding="same")(layer3)

	layer5=Conv2D(filters=64,kernel_size=3,activation="relu",padding="same")(layer4)

	layer6=Conv2DTranspose(filters=32,kernel_size=3,activation="relu",padding="same")(layer5)
	layer7= UpSampling2D((2, 2))(layer6)

	layer8=Conv2DTranspose(filters=16,kernel_size=5,activation="relu",padding="same")(layer7)
	layer9= UpSampling2D((2, 2))(layer8)

	output = Conv2DTranspose(3, kernel_size=3, activation='sigmoid',padding = 'same')(layer9)


	model=Model(inputs=inp,outputs=output)

	return model

def calcPSNR(y_true, y_pred):

	derain_images = denormalize(y_true)
	generated_images = denormalize(y_pred)

	mse = ((derain_images - generated_images) ** 2).mean(axis=None)
	psnr = 20*np.log10(255/(mse**(1/2.0)))

	return psnr

model = create_model()

model.load_model("auto-encoder")

#Compile and train
model.compile(optimizer='adam', loss='mse',metrics = [psnr])
model.fit(rain_images,derain_images,epochs=20,batch_size=35)

#save model
model.save("/output/model")

#write to json
generator_json = model.to_json()
with open("/output/modelJson.json", "w") as json_file:
    json_file.write(generator_json)

model.save_weights("/output/auto-encoder.h5")











    
    
