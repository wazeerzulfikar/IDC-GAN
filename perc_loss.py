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






# def content_loss(endpoints_dict, content_layers):
#     content_loss = 0
#     for layer in content_layers:
#         generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
#         size = tf.size(generated_images)
#         content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
#     return content_loss




