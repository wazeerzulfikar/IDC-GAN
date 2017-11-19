import tensorflow as tf
from keras import backend as K

from model import *

def perc_loss(generated_image, actual_image, modelD):
	layer = 8 # Depends from which layer we want features. This might be wrong
	layer_output = K.function([modelD.layers[0].input],[modelD.layers[layer].output])

	generated_features = layer_output([generated_image])[0]
	actual_features = layer_output([actual_image])[0]

	loss = ((generated_features - actualfeatures)**2)/2
	return loss
