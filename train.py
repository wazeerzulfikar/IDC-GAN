import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from keras.optimizers import Adam
import keras.backend as K

from model import *
from losses import *

rain_data_path = "./dataset/rain_modified"
derain_data_path = "./dataset/derain_modified"

n_epoch = 25
ngf = 64
ndf = 48
batch_size = 7
lambda1 = 150
lambda2 = 150
img_shape = (256,256,3)


def load_images(folder):
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


generator = create_generator(3, 3, ngf, img_shape)
print(generator.summary())
discriminator = create_discriminator(3, 3, ndf, 3, img_shape)
print(discriminator.summary())
discriminator_on_generator, x_generator, x_discriminator = generator_containing_discriminator(generator, discriminator, img_shape)

g_optim = Adam(lr=0.0001,beta_1=0.5)
d_optim = Adam(lr=0.0001,beta_1=0.5)

discriminator.compile(d_optim, loss=discriminator_loss)
generator.compile(g_optim, loss='mse')
discriminator_on_generator.compile(g_optim, loss = [refined_loss(d_out=x_discriminator),lambda x,y: 1])

for i in range(n_epoch):
	print("Epoch : %d"%i)
	for batch_idx in range(0, len(rain_data), batch_size):
		batch_x = rain_data[batch_idx:batch_idx+batch_size]
		batch_y = derain_data[batch_idx:batch_idx+batch_size]

		generated_images = generator.predict(batch_x)
		real_pairs = np.concatenate((batch_x, batch_y), axis=3)
		fake_pairs = np.concatenate((batch_x, generated_images), axis=3)

		x = np.concatenate((real_pairs, fake_pairs))
		y = np.concatenate((np.ones((batch_size, 32, 32, 1)),np.zeros((batch_size, 32, 32, 1))))
		d_loss = discriminator.train_on_batch(x, y)

		discriminator.trainable = False
		d_out = np.ones((batch_size, 32, 32, 1))
		g_loss = discriminator_on_generator.train_on_batch(batch_x, [batch_y,d_out])
		
		discriminator.trainable = True


	print("Discriminator Loss : ",d_loss)
	print("Generator Loss : ",g_loss)
	generated_images = generator.predict(batch_x)
	for k,img in enumerate(generated_images):
		img +=1
		img*=127.5
		img = Image.fromarray(img.astype(np.uint8))
		img.save(os.path.join('/output',"epoch_%d_%d.jpg"%(i,k)))




