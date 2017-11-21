import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model


from model import *

lambda1 = 6.6e-3
lambda2 = 1

def constant_loss(y_true,y_pred):
	return K.constant(1)

def perc_loss(generated_image, actual_image):
	base_model = VGG16(weights='imagenet', include_top=False)

	model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)
	# generated_features = layer_output([generated_image])[0]
	# actual_features = layer_output([actual_image])[0]
	generated_features = model.predict(generated_image)
	actual_features = model.predict(actual_image)

	# loss = K.mean(K.square(K.flatten(generated_features) - K.flatten(actual_features)), axis=-1)
	return loss

def entropy_loss(d_out):
	return -1*K.mean(K.log(K.flatten(d_out)))

def discriminator_loss(y_true,y_pred):
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.flatten(y_true)), axis=-1)

def generator_l2_loss(y_true,y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


def refined_perceptual_loss(y_true, y_pred):
	return discriminator_loss(y_true, y_pred)+lambda1*perceptual_loss(y_true,y_pred)+lambda2*generator_l2_loss(y_true, y_pred)


def refined_loss(d_out):
	def loss(y_true, y_pred):
		return lambda1*entropy_loss(d_out)+ generator_l2_loss(y_true, y_pred)
	return loss
