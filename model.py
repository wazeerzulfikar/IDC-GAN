import functools
from functools import partial
from functools import reduce

from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Conv2DTranspose
from keras.layers import Add, Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

from keras.applications.vgg16 import VGG16


_Conv2D = partial(Conv2D, padding="same")
_Conv2DTranspose = partial(Conv2DTranspose, padding="same")

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)


def LR_Conv_BN(*args, **kwargs):
	return compose(
		LeakyReLU(0.02),
		_Conv2D(*args,**kwargs),
		BatchNormalization()
		)


def R_DeConv_BN(*args, **kwargs):
	return compose(
		Activation('relu'),
		_Conv2DTranspose(*args, **kwargs),
		BatchNormalization()
		)


def create_generator(input_nc, output_nc, ngf):

	inputs = Input(shape=(None,None,3))

	#Encoder

	e1 = _Conv2D(ngf, (3,3), strides=1)(inputs)

	e2 = LR_Conv_BN(ngf, (3,3), strides=1)(e1)

	e3 = LR_Conv_BN(ngf, (3,3), strides=1)(e2)

	e4 = LR_Conv_BN(ngf, (3,3), strides=1)(e3)

	e5 = LR_Conv_BN(int(ngf/2), (3,3), strides=1)(e4)

	e6 = LR_Conv_BN(1, (3,3), strides=1)(e5)

	#Decoder

	d1 = LeakyReLU(0.02)(e6)
	d1 = _Conv2DTranspose(int(ngf/2), (3,3), strides=1)(d1)
	d1 = BatchNormalization()(d1)

	d2 = R_DeConv_BN(ngf, (3,3), strides=1)(d1)
	d2 = Add()([d2,e4])

	d3 = R_DeConv_BN(ngf, (3,3), strides=1)(d2)

	d4 = R_DeConv_BN(ngf, (3,3), strides=1)(d3)
	d4 = Add()([d4,e2])

	d5 = R_DeConv_BN(ngf, (3,3), strides=1)(d4)

	d6 = Activation('relu')(d5)
	d6 = _Conv2DTranspose(output_nc, (3,3), strides=1)(d6)

	o1 = Activation('tanh')(d6)

	model = Model(inputs,o1)

	return model


def create_discriminator(input_nc, output_nc, ndf, n_layers):
	model = Sequential()

	model.add(_Conv2D(ndf, (4,4), strides=2, input_shape=(None,None,input_nc+output_nc)))
	model.add(LeakyReLU(0.2))

	for i in range(1,n_layers-1):
		nf_mult = min(2**i,8)
		model.add(_Conv2D(ndf*nf_mult,(4,4),strides=2))
		model.add(BatchNormalization())
		model.add(LeakyReLU(0.2))

	nf_mult = min(2**n_layers, 8)
	model.add(_Conv2D(ndf*nf_mult, (4,4), strides=2))
	model.add(BatchNormalization())
	model.add(_Conv2D(1, (4,4), strides=1))
	model.add(Activation('sigmoid'))

	return model


def generator_containing_discriminator(generator, discriminator):
    inputs = Input(None,None,3)
    x_generator = generator(inputs)
    
    merged = Concatenate(axis=3)([inputs, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    concatenated = Concatenate(axis=0)([inputs, x_generator])
    base_model = VGG16(weights='imagenet', include_top=False)

    vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)
    vgg_model.trainable = False

    vgg_model_out = vgg_model(concatenated)
    
    model = Model(inputs=inputs, outputs=[x_generator,x_discriminator,vgg_model_out])
    
    return model, x_generator, x_discriminator, vgg_model_out

if __name__ == '__main__':
	print("HEY")