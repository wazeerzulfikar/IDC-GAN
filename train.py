import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from keras.optimizers import Adam
from keras.models import model_from_json
import keras.backend as K

from model import *
from losses import *

rain_data_path = "/dataset/rain_modified"
derain_data_path = "/dataset/derain_modified"

n_epoch = 100
save_step = 20
disp_step = 10
ngf = 64
ndf = 48
batch_size = 7
img_shape = (256,256,3)

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
			img = normalize(np.asarray(img))
			images.append(img)
	return np.array(images)


def normalize(img):
	return (img/127.5) - 1


rain_data = load_images(rain_data_path)
print("Rain Data Loaded.")
derain_data = load_images(derain_data_path)
print("DeRain Data Loaded.")


generator = create_generator(3, 3, ngf)
print(generator.summary())
discriminator = create_discriminator(3, 3, ndf, 3)
print(discriminator.summary())
discriminator_on_generator, x_generator, x_discriminator, vgg_model_out = generator_containing_discriminator(generator, discriminator)

g_optim = Adam(lr=0.002,beta_1=0.5)
d_optim = Adam(lr=0.002,beta_1=0.5)

discriminator.compile(d_optim, loss=discriminator_loss)
generator.compile(g_optim, loss='mse')
discriminator_on_generator.compile(g_optim, loss = [refined_loss(d_out=x_discriminator,vgg_out=vgg_model_out),constant_loss,constant_loss])

generator_json = generator.to_json()
with open("generator.json", "w") as json_file:
    json_file.write(generator_json)

discriminator_json = discriminator.to_json()
with open("discriminator.json", "w") as json_file:
    json_file.write(discriminator_json)

for i in range(n_epoch):
	for batch_idx in range(0, len(rain_data), batch_size):
		batch_x = rain_data[batch_idx:batch_idx+batch_size]
		batch_y = derain_data[batch_idx:batch_idx+batch_size]

		print("Loaded images")

		generated_images = generator.predict(batch_x)

		real_pairs = np.concatenate((batch_x, batch_y), axis=3)
		fake_pairs = np.concatenate((batch_x, generated_images), axis=3)

		print("Pairs made")

		x = np.concatenate((real_pairs, fake_pairs))
		y = np.concatenate((np.ones((batch_size, 32, 32, 1)),np.zeros((batch_size, 32, 32, 1))))
		d_loss = discriminator.train_on_batch(x, y)

		# perc_loss = perceptual_loss(batch_y, generated_images)

		# print(perc_loss)

		print("Concatenations done")
		discriminator.trainable = False
		rand = np.ones((batch_size, 32, 32, 1))
		g_loss = discriminator_on_generator.train_on_batch(batch_x, [batch_y,rand,rand])
		
		discriminator.trainable = True

	if i%save_step == 0:
		discriminator.save_weights(os.path.join("/output","discriminator_%d.h5"%i))
		generator.save_weights(os.path.join("/output","generator_%d.h5"%i))

	if i%disp_step == 0:
		print("Epoch : %d"%i)
		print("Discriminator Loss : ",d_loss)
		print("Generator Loss : ",g_loss)
		generated_images = generator.predict(batch_x)

		#Calculating PSNR
		mse = generator_l2_loss(batch_y, generated_images)
		psnr = 20*log10(255/(mse**(1/2.0)))

		print("Peak Signal to Noise Ratio:", psnr)


		for k,img in enumerate(generated_images):
			img +=1
			img*=127.5
			img = Image.fromarray(img.astype(np.uint8))
			img.save(os.path.join('/output',"epoch_%d_%d.jpg"%(i,k)))
