import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from keras.optimizers import Adam
from keras.models import model_from_json
import keras.backend as K

import scipy.misc as misc

from model import *
from losses import *

rain_data_path = "/dataset/rain_images.npy"
derain_data_path = "/dataset/derain_images.npy"

n_epoch = 100
save_step = 10
disp_step = 1
ngf = 64
ndf = 48
batch_size = 7
img_shape = (256,256,3)

def load_images(filename):
	return np.load(filename)

def normalize(img):
	return (img/127.5) - 1

def denormalize(img):
	return (img+1)*127.5

def calcPSNR(derain_images, generated_images):

	derain_images = denormalize(derain_images)
	generated_images = denormalize(generated_images)

	mse = ((derain_images - generated_images) ** 2).mean(axis=None)
	psnr = 20*np.log10(255/(mse**(1/2.0)))

	print("Peak Signal to Noise Ratio:", psnr)

rain_data = normalize(load_images(rain_data_path))
print("Rain Data Loaded.")
derain_data = normalize(load_images(derain_data_path))
print("DeRain Data Loaded.")

generator = create_generator(3, 3, ngf)
print(generator.summary())
discriminator = create_discriminator(3, 3, ndf, 3)
print(discriminator.summary())

#generator.load_weights(os.path.join("/saved","generator_60.h5"))
#discriminator.load_weights(os.path.join("/saved","discriminator_60.h5"))

discriminator_on_generator, x_generator, x_discriminator, vgg_model_out = generator_containing_discriminator(generator, discriminator)

g_optim = Adam(lr=0.002,beta_1=0.5)
d_optim = Adam(lr=0.002,beta_1=0.5)

discriminator.compile(d_optim, loss=discriminator_loss)
generator.compile(g_optim, loss='mse')
discriminator_on_generator.compile(g_optim, loss = [refined_loss(d_out=x_discriminator,vgg_out=vgg_model_out),constant_loss,constant_loss])

generator_json = generator.to_json()
with open(os.path.join("/output","generator.json"), "w") as json_file:
    json_file.write(generator_json)

discriminator_json = discriminator.to_json()
with open(os.path.join("/output","discriminator.json"), "w") as json_file:
    json_file.write(discriminator_json)

discriminator_on_generator_json = discriminator_on_generator.to_json()
with open(os.path.join("/output","discriminator_on_generator.json"), "w") as json_file:
    json_file.write(discriminator_on_generator_json)

for i in range(n_epoch):
	for batch_idx in range(0, len(rain_data), batch_size):
		batch_x = rain_data[batch_idx:batch_idx+batch_size]
		batch_y = derain_data[batch_idx:batch_idx+batch_size]

		generated_images = generator.predict(batch_x)

		real_pairs = np.concatenate((batch_x, batch_y), axis=3)
		fake_pairs = np.concatenate((batch_x, generated_images), axis=3)

		discriminator.trainable = True

		x = np.concatenate((real_pairs, fake_pairs))
		y = np.concatenate((np.ones((batch_size, 32, 32, 1)),np.zeros((batch_size, 32, 32, 1))))
		d_loss = discriminator.train_on_batch(x, y)

		discriminator.trainable = False
		rand = np.ones((batch_size, 32, 32, 1))
		g_loss = discriminator_on_generator.train_on_batch(batch_x, [batch_y,rand,rand])
		
		discriminator.trainable = True

	print("Epoch : %d"%i)
	print("Discriminator Loss : ",d_loss)
	print("Generator Loss : ",g_loss)

	if i%save_step == 0:
		discriminator.save_weights(os.path.join("/output","discriminator_%d.h5"%i))
		generator.save_weights(os.path.join("/output","generator_%d.h5"%i))
		discriminator_on_generator.save_weights(os.path.join("/output","discriminator_on_generator_%d.h5"%i))

	if i%disp_step == 0:
		predict_batch =  rain_data[:100]
		true_batch = derain_data[:100]
		generated_images = generator.predict(predict_batch)
		calcPSNR(true_batch, generated_images)
		
		if i %save_step == 0:
			for k,img in enumerate(generated_images[:10]):
				img +=1
				img*=127.5
				img = Image.fromarray(img.astype(np.uint8))
				img.save(os.path.join('/output',"epoch_%d_%d.jpg"%(i,k)))
