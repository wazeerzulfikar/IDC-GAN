import tensorflow as tf
from keras import backend as K

from model import *

lambda1 = 6.6e-3
lambda2 = 1

def perc_loss(generated_image, actual_image, modelD):
	layer = 8 # Depends from which layer we want features. This might be wrong
	layer_output = K.function([modelD.layers[0].input],[modelD.layers[layer].output])

	generated_features = layer_output([generated_image])[0]
	actual_features = layer_output([actual_image])[0]

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
