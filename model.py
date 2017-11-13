from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

def create_discriminator(input_nc, output_nc, ndf, n_layers):
	discriminator = Sequential()

	discriminator.add(Conv2D(ndf,(4,4),2,input_shape=(128,128,input_nc+output_nc)))
	discriminator.add(LeakyReLU(0.2))

	for i in range(1,n_layers-1):
		nf_mult = min(2**n,8)
		model.add(Conv2D(ndf*nf_mult,(4,4),2))
		model.add(BatchNormalization())
		model.add(LeakyReLU(0.2))

	nf_mult = min(2**n_layers,8)
	discriminator.add(Conv2D(ndf*nf_mult, (4,4),1))
	discriminator.add(BatchNormalization())
	discriminator.add(Conv2D(1,(4,4),1))
	discriminator.add(Activation('sigmoid'))

	return discriminator




