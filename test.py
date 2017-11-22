from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from PIL import Image

import numpy as np
import os

from model import *
from losses import *



# load json and create model
json_file = open('models/generator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

generator = model_from_json(loaded_model_json)
# load weights into new model
generator.load_weights("models/generator_90.h5")
print("Loaded model from disk")

rain_data_path = "dataset/rain_modified"
derain_data_path = "dataset/derain_modified"


def log10(x):
  numerator = K.log(x)
  denominator = K.log(K.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def load_images(folder):
	check = 1

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

print(generator.summary())


input_images = rain_data[0:10]
derain_images = derain_data[0:10]

generated_images = generator.predict(input_images)


mse = generator_l2_loss(derain_images, generated_images)
psnr = 20*log10(255/(mse**(1/2.0)))

print("Peak Signal to Noise Ratio:", psnr)




