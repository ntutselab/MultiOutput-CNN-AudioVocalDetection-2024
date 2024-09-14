#!/usr/bin/env python3
import signal
import scipy.signal
from scipy.signal import *

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

class STFT_network (tf.keras.layers.Layer):
    def __init__ (self, window_length = 2048, over_lapping = 512, padding = "same", **kwargs):
        super(STFT_network, self).__init__(**kwargs)
        self.window_length = window_length
        self.frequency_bin = int(self.window_length/2)
        self.over_lapping  = over_lapping
        self.padding = padding
        
        self.fourier_basis = np.fft.fft(np.eye(self.window_length))
        self.discrete_fourier_transform_window = scipy.signal.hann(self.window_length, sym = False)
        self.discrete_fourier_transform_window = self.discrete_fourier_transform_window.reshape((1, -1))
        
        self.kernel = np.multiply(self.fourier_basis, self.discrete_fourier_transform_window)
        del (self.fourier_basis)
        
        self.kernel = self.kernel[:self.frequency_bin, :]
        self.real_kernel_init = np.real(self.kernel)
        self.imag_kernel_init = np.imag(self.kernel)
        self.real_kernel_init = self.real_kernel_init.T
        self.imag_kernel_init = self.imag_kernel_init.T
        self.real_kernel_init = self.real_kernel_init[:, None, :]
        self.imag_kernel_init = self.imag_kernel_init[:, None, :]
        

    def build (self, inputs_shape):
        """
        self.frequency_bin : integer
        self.window_length : integer
        self.strides : same as over_lapping number
        kernel_initializer : 3D Tensor [width = window_length, height = 1, output size : self.frequency_bin]
        """  
        self.real_fourier_convolution = Conv1D(
            filters = self.frequency_bin, 
            kernel_size = self.window_length, 
            strides = self.over_lapping, 
            padding = self.padding,
            kernel_initializer = tf.keras.initializers.Constant(value = self.real_kernel_init),
            trainable = False)
        self.imag_foruier_convolution = Conv1D(
            filters = self.frequency_bin, 
            kernel_size = self.window_length, 
            strides = self.over_lapping, 
            padding = self.padding,
            kernel_initializer = tf.keras.initializers.Constant(value = self.imag_kernel_init),
            trainable = False)
        super(STFT_network, self).build(inputs_shape)
        

    def call (self, input_signal, **kwargs):
        'inputs_signal : 3D Tensor [batch_size, signal_length, channel_number]'
        real = self.real_fourier_convolution(input_signal)
        imag = self.imag_foruier_convolution(input_signal)
        real, imag = reshaped_STFT(real, imag)
        return real, imag

def reshaped_STFT (real, imag):
    real = tf.reshape(real, (-1, 63, 1024, 1))
    imag = tf.reshape(imag, (-1, 63, 1024, 1))
    
    return real, imag


class complex_USCLLayer(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, stride=1, pooling=False, padding='same', index=0, activation='relu', **kwargs):
        super(complex_USCLLayer, self).__init__(**kwargs)
        self.pooling = pooling

        if self.pooling:
            self.pool = complex_MaxPooling(kernel_size)

        self.conv_1 = complex_Conv2D(filters, kernel_size, strides=stride, name=f'cuscl_{filters}_{index}_1', padding=padding)
        self.bn_1 = complex_NaiveBatchNormalization(name=f'cuscl_{filters}_{index}_1_bn')
        self.relu_1 = complex_Activation(activation, name=f'cuscl_{filters}_{index}_1_{activation}')

    def call(self, real_inputs, imag_inputs):
        real, imag = self.conv_1(real_inputs, imag_inputs)
        real, imag = self.relu_1(real, imag)
        real, imag = self.bn_1(real, imag)

        if self.pooling:
            real, imag = self.pool(real, imag)

        return real, imag
    
'COMPLEX POOLING'
class complex_MaxPooling (tf.keras.layers.Layer):
    def __init__(self, 
                pool_size=(2, 2),
                strides=None,
                padding='valid',
                **kwargs):
        super(complex_MaxPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides   = strides
        self.padding   = padding


    def build (self, inputs_shape):
        self.real_maxpooling = tf.keras.layers.MaxPool2D(
            pool_size = self.pool_size, 
            strides = self.strides, 
            padding = self.padding)
        self.imag_maxpooling = tf.keras.layers.MaxPool2D(
            pool_size = self.pool_size, 
            strides = self.strides, 
            padding = self.padding)
        super(complex_MaxPooling, self).build(inputs_shape)
        

    def call (self, real_inputs, imag_inputs):
        real_outputs = self.real_maxpooling(real_inputs)
        imag_outputs = self.imag_maxpooling(imag_inputs)
        return real_outputs, imag_outputs

'COMPLEX CONVOLUTION 2D'
class complex_Conv2D (tf.keras.layers.Layer):
    def __init__(self, 
                filters,
                kernel_size,
                strides=(1, 1),
                padding='valid',
                activation = None,
                use_bias   = True,
                kernel_initializer = 'glorot_uniform',
                bias_initializer   = 'zeros',
                **kwargs):
        super(complex_Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        self.use_bias    = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        
        
    def build (self, inputs_shape):
        self.real_Conv2D = tf.keras.layers.Conv2D(
            filters = self.filters,
            kernel_size = self.kernel_size, 
            strides = self.strides,
            padding = self.padding,
            activation = self.activation,
            use_bias = self.use_bias,
            kernel_initializer = self.kernel_initializer,
            bias_initializer = self.bias_initializer) 
        self.imag_Conv2D = tf.keras.layers.Conv2D(
            filters = self.filters,
            kernel_size = self.kernel_size, 
            strides = self.strides,
            padding = self.padding,
            activation = self.activation,
            use_bias = self.use_bias,
            kernel_initializer = self.kernel_initializer,
            bias_initializer = self.bias_initializer) 
        super(complex_Conv2D, self).build(inputs_shape)

        
    def call(self, real_inputs, imag_inputs):
        real_outputs = self.real_Conv2D(real_inputs) - self.imag_Conv2D(imag_inputs)
        imag_outputs = self.imag_Conv2D(real_inputs) + self.real_Conv2D(imag_inputs)
        return real_outputs, imag_outputs

class complex_NaiveBatchNormalization (tf.keras.layers.Layer):
    '''
    tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer='zeros', gamma_initializer='ones',
                                        moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                        gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
                                        fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
                                        **kwargs)
    '''
    def __init__ (self, axis = -1, 
                        momentum = 0.99, 
                        epsilon = 0.001, 
                        center = True, 
                        scale = True,
                        beta_initializer = 'zeros', 
                        gamma_initializer = 'ones',
                        moving_mean_initializer = 'zeros',
                        moving_variance_initializer = 'ones',
                        beta_regularizer = None, 
                        gamma_regularizer = None, 
                        beta_constraint = None,
                        gamma_constraint = None,
                        renorm = False,
                        renorm_clipping = None, 
                        renorm_momentum = 0.99,
                        fused = None, 
                        trainable = True, 
                        virtual_batch_size = None,
                        adjustment = None,
                        **kwargs):
        super(complex_NaiveBatchNormalization, self).__init__(**kwargs)

        self.axis = axis
        self.momentum = momentum
        self.epsilon  = epsilon
        self.center   = center
        self.scale    = scale 
        self.beta_initializer            = beta_initializer
        self.gamma_initializer           = gamma_initializer
        self.moving_mean_initializer     = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer            = beta_regularizer
        self.gamma_regularizer           = gamma_regularizer
        self.beta_constraint             = beta_constraint
        self.gamma_constraint            = gamma_constraint
        self.renorm                      = renorm
        self.renorm_clipping             = renorm_clipping
        self.renorm_momentum             = renorm_momentum
        self.fused                       = fused
        self.trainable                   = trainable
        self.virtual_batch_size          = virtual_batch_size
        self.adjustment                  = adjustment

        self.real_batchnormalization = tf.keras.layers.BatchNormalization(axis = self.axis,
                                                                        momentum = self.momentum,
                                                                        epsilon = self.epsilon,
                                                                        center = self.center,
                                                                        scale = self.scale,
                                                                        beta_initializer = self.beta_initializer,
                                                                        gamma_initializer = self.gamma_initializer,
                                                                        moving_mean_initializer = self.moving_mean_initializer,
                                                                        moving_variance_initializer = self.moving_variance_initializer,
                                                                        beta_regularizer = self.beta_regularizer,
                                                                        gamma_regularizer = self.gamma_regularizer,
                                                                        beta_constraint = self.beta_constraint,
                                                                        gamma_constraint = self.gamma_constraint,
                                                                        renorm = self.renorm,
                                                                        renorm_clipping = self.renorm_clipping,
                                                                        renorm_momentum = self.renorm_momentum,
                                                                        fused = self.fused,
                                                                        trainable = self.trainable,
                                                                        virtual_batch_size = self.virtual_batch_size,
                                                                        adjustment = self.adjustment)

        self.imag_batchnormalization = tf.keras.layers.BatchNormalization(axis = self.axis,
                                                                        momentum = self.momentum,
                                                                        epsilon = self.epsilon,
                                                                        center = self.center,
                                                                        scale = self.scale,
                                                                        beta_initializer = self.beta_initializer,
                                                                        gamma_initializer = self.gamma_initializer,
                                                                        moving_mean_initializer = self.moving_mean_initializer,
                                                                        moving_variance_initializer = self.moving_variance_initializer,
                                                                        beta_regularizer = self.beta_regularizer,
                                                                        gamma_regularizer = self.gamma_regularizer,
                                                                        beta_constraint = self.beta_constraint,
                                                                        gamma_constraint = self.gamma_constraint,
                                                                        renorm = self.renorm,
                                                                        renorm_clipping = self.renorm_clipping,
                                                                        renorm_momentum = self.renorm_momentum,
                                                                        fused = self.fused,
                                                                        trainable = self.trainable,
                                                                        virtual_batch_size = self.virtual_batch_size,
                                                                        adjustment = self.adjustment)
        

    def call (self, real_inputs, imag_inputs, training = True):

        real_outputs = self.real_batchnormalization (real_inputs, training = training)
        imag_outputs = self.imag_batchnormalization (imag_inputs, training = training)

        return real_outputs, imag_outputs


class complex_Activation (tf.keras.layers.Layer):

    def __init__(self, activation, **kwargs):
        super(complex_Activation, self).__init__(**kwargs)

        self.activation = activation

        self.real_Activation = tf.keras.layers.Activation (activation = self.activation)
        self.imag_Activation = tf.keras.layers.Activation (activation = self.activation)
    
    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Activation (real_inputs)
        imag_outputs = self.imag_Activation (imag_inputs)

        return real_outputs, imag_outputs


class complex_Dropout (tf.keras.layers.Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(complex_Dropout, self).__init__(**kwargs)

        self.rate = rate

        self.real_Dropout = tf.keras.layers.Dropout (rate = self.rate)
        self.imag_Dropout = tf.keras.layers.Dropout (rate = self.rate)
    
    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Dropout (real_inputs)
        imag_outputs = self.imag_Dropout (imag_inputs)

        return real_outputs, imag_outputs
    

class complex_Flatten (tf.keras.layers.Layer):

    def __init__(self, data_format=None, **kwargs):
        super(complex_Flatten, self).__init__(**kwargs)

        self.real_Flatten = tf.keras.layers.Flatten ()
        self.imag_Flatten = tf.keras.layers.Flatten ()
    
    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Flatten (real_inputs)
        imag_outputs = self.imag_Flatten (imag_inputs)

        return real_outputs, imag_outputs
