from keras import backend as K
import tensorflow as tf
import numpy as np

from model import *

lambda1 = 6.6e-3
lambda2 = 1

def constant_loss(y_true,y_pred):
	return K.constant(1)

def perceptual_loss(vgg_out):
	actual_features, generated_features = tf.split(vgg_out,2)
	loss = K.mean(K.square(K.flatten(generated_features) - K.flatten(actual_features)), axis=-1)
	return loss

def entropy_loss(d_out):
	return -1*K.mean(K.log(K.flatten(d_out)))

def discriminator_loss(y_true,y_pred):
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.flatten(y_true)), axis=-1)

def generator_l2_loss(y_true,y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


def refined_perceptual_loss(y_true, y_pred):
	return discriminator_loss(y_true, y_pred)+lambda1*perceptual_loss(y_true,y_pred)+lambda2*generator_l2_loss(y_true, y_pred)


def refined_loss(d_out,vgg_out):
	def loss(y_true, y_pred):
		return lambda1*entropy_loss(d_out)+ generator_l2_loss(y_true, y_pred) + lambda2*perceptual_loss(vgg_out)
	return loss
