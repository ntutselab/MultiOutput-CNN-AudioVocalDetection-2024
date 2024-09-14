#!/usr/bin/env python3
import tensorflow as tf
from models.SCNN18_Flatten import SCNN18_Flatten
import models.Network as Network
from tensorflow.keras import Model

class BYOL(tf.keras.Model):
    def __init__(self, input_shape):
        super(BYOL, self).__init__()
        self.online_model = Network.build_model(input_shape=input_shape, online=True)
        self.target_model = Network.build_model(input_shape=input_shape, online=False)
        
    def call(self, x1, x2):
        z1_online = self.online_model(x1)
        z2_online = self.online_model(x2)
        z1_target = self.target_model(x1)
        z2_target = self.target_model(x2)

        p1 = tf.math.l2_normalize(z1_online, axis=1)
        p2 = tf.math.l2_normalize(z2_online, axis=1)
        z1_target = tf.math.l2_normalize(z1_target, axis=1)
        z2_target = tf.math.l2_normalize(z2_target, axis=1)

        loss1 = self.compute_loss(p1, z2_target)
        loss2 = self.compute_loss(p2, z1_target)

        return tf.reduce_mean(loss1 + loss2)

    def compute_loss(self, p, z):
        z = tf.stop_gradient(z)
        return 2 - 2 * tf.reduce_sum(p * z, axis=1)

    def update_target_network(self, momentum=0.99):
        for target_weights, online_weights in zip(self.target_model.weights, self.online_model.weights):
            target_weights.assign(momentum * target_weights + (1 - momentum) * online_weights)
