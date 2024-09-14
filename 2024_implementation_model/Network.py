#!/usr/bin/env python3
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
sys.path.append('/root/code')

import tensorflow as tf
from models.SCNN18_Flatten import SCNN18_Flatten
from models.SCNN18_Flatten_random_zoom import SCNN18_Flatten_random_zoom

def build_model(input_shape, online=True):

    # The Encoder
    if online:
        model = SCNN18_Flatten(input_shape=input_shape).model()
    else:
        model = SCNN18_Flatten_random_zoom(input_shape=input_shape).model()

    # The Projector
    x = tf.keras.layers.Dense(512)(model.output)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation('relu')(x)
    if online:
        x = tf.keras.layers.Dense(256)(x)
    else:
        output = tf.keras.layers.Dense(256)(x)

    if online:
        # The Predictor
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        output = tf.keras.layers.Dense(256)(x)

    return tf.keras.Model(inputs=model.inputs, outputs=output)
