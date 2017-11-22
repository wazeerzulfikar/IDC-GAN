from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf

from PIL import Image

import numpy as np
import os

from model import *
from losses import *

# load json and create model
json_file = open('/saved/generator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

generator = model_from_json(loaded_model_json)
# load weights into new model
generator.load_weights("/saved/generator_90.h5")
print("Loaded model from disk")

rain_data_path = "/dataset/rain_modified"
derain_data_path = "/dataset/derain_modified"

def load_images(folder):
	check = 0

	images = []
	for filename in os.listdir(folder):
		if filename.endswith(".jpg"):
			img = Image.open(os.path.join(folder, filename))
			img = img.resize((256,256))
			img = normalize(np.asarray(img))
			images.append(img)
	return np.array(images)


def normalize(img):
	return (img/127.5) - 1


rain_data = load_images(rain_data_path)
print("Rain Data Loaded.")
derain_data = load_images(derain_data_path)
print("DeRain Data Loaded.")

input_images = rain_data
derain_images = derain_data

generated_images = generator.predict(input_images)

generated_images += 1
generated_images *=127.5

mse = ((rain_data - generated_images) ** 2).mean(axis=None)
psnr = 20*np.log10(255/(mse**(1/2.0)))


print("Peak Signal to Noise Ratio for 0:", psnr)






