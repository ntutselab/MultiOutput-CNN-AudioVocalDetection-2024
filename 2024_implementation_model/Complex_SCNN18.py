#!/usr/bin/env python3
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers as klayers
from models.Complex_Layers import *

class Complex_SCNN18(Model):

    def __init__(self, input_shape, nb_classes, **kwargs):
        super(Complex_SCNN18, self).__init__(**kwargs)

        self.input_layer = klayers.Input(input_shape, name='input')

        self.STFT = STFT_network()
        self.cuscl_conv1 = complex_USCLLayer(64, (3, 2), (3, 2), False, 'valid', 1, name='cuscl_conv1')
        self.cuscl_conv2 = complex_USCLLayer(64, (1, 2), (1, 1), False, 'same', 2, name='cuscl_conv2')
        self.cuscl_conv3 = complex_USCLLayer(64, (1, 2), (1, 1), False, 'same', 3, name='cuscl_conv3')
        self.cuscl_conv4 = complex_USCLLayer(64, (1, 2), (1, 1), True, 'same', 4, name='cuscl_conv4')
        self.cuscl_conv5 = complex_USCLLayer(64, (1, 2), (1, 1), True, 'same', 5, name='cuscl_conv5')
        self.cuscl_conv6 = complex_USCLLayer(64, (1, 2), (1, 1), False, 'same', 6, name='cuscl_conv6')
        self.cuscl_conv7 = complex_USCLLayer(64, (1, 2), (1, 1), True, 'same', 7, name='cuscl_conv7')
        self.cuscl_conv8 = complex_USCLLayer(128, (1, 2), (1, 1), False, 'same', 8, name='cuscl_conv8')
        self.cuscl_conv9 = complex_USCLLayer(128, (1, 2), (1, 1), False, 'same', 9, name='cuscl_conv9')
        self.cuscl_conv10 = complex_USCLLayer(128, (1, 2), (1, 1), True, 'same', 10, name='cuscl_conv10')
        self.cuscl_conv11 = complex_USCLLayer(128, (1, 2), (1, 1), True, 'same', 11, name='cuscl_conv11')
        self.cuscl_conv12 = complex_USCLLayer(128, (1, 2), (1, 1), False, 'same', 12, name='cuscl_conv12')
        self.cuscl_conv13 = complex_USCLLayer(128, (1, 2), (1, 1), True, 'same', 13, name='cuscl_conv13')
        self.cuscl_conv14 = complex_USCLLayer(256, (1, 2), (1, 1), True, 'same', 14, name='cuscl_conv14')
        self.cuscl_conv15 = complex_USCLLayer(256, (1, 2), (1, 1), True, 'same', 15, name='cuscl_conv15')
        self.cuscl_conv16 = complex_USCLLayer(256, (1, 2), (1, 1), False, 'same', 16, name='cuscl_conv16')

        self.final_cuscl = complex_Conv2D(256, (3, 2), (1, 1), 'same', name='final_cuscl')
        self.final_cuscl_relu = complex_Activation('relu')
        self.final_cuscl_bn = complex_NaiveBatchNormalization(name='final_cuscl_bn')
        self.final_cpool = complex_MaxPooling((3, 2))

        self.final_cconv = complex_Conv2D(256, (1, 1), padding='same', name='final_cconv')
        self.final_crelu = complex_Activation('relu')
        self.final_cbn = complex_NaiveBatchNormalization(name='final_cbn')

        self.cdropout = complex_Dropout(0.5, name='cuscl_dropout')
        self.cflatten = complex_Flatten(name='cflatten')
        self.out_dense1 = klayers.Dense(nb_classes, name="Dense_2nb_softmax_1", activation='softmax')
        self.out_dense2 = klayers.Dense(nb_classes, name="Dense_2nb_softmax_2", activation='softmax')

    def model(self):
        return Model(inputs=[self.input_layer], outputs=self.call(self.input_layer))
    
    def call(self, x, **kwargs):
        real, imag = self.STFT(x)
        real, imag = self.cuscl_conv1(real, imag)
        real, imag = self.cuscl_conv2(real, imag)
        real, imag = self.cuscl_conv3(real, imag)
        real, imag = self.cuscl_conv4(real, imag)
        real, imag = self.cuscl_conv5(real, imag)
        real, imag = self.cuscl_conv6(real, imag)
        real, imag = self.cuscl_conv7(real, imag)
        real, imag = self.cuscl_conv8(real, imag)
        real, imag = self.cuscl_conv9(real, imag)
        real, imag = self.cuscl_conv10(real, imag)
        real, imag = self.cuscl_conv11(real, imag)
        real, imag = self.cuscl_conv12(real, imag)
        real, imag = self.cuscl_conv13(real, imag)
        real, imag = self.cuscl_conv14(real, imag)
        real, imag = self.cuscl_conv15(real, imag)
        real, imag = self.cuscl_conv16(real, imag)
        real, imag = self.final_cuscl(real, imag)
        real, imag = self.final_cuscl_relu(real, imag)
        real, imag = self.final_cuscl_bn(real, imag)
        real, imag = self.final_cpool(real, imag)
        real, imag = self.final_cconv(real, imag)
        real, imag = self.final_crelu(real, imag)
        real, imag = self.final_cbn(real, imag)
        real, imag = self.cdropout(real, imag)
        real, imag = self.cflatten(real, imag)

        return self.out_dense1(real), self.out_dense2(imag)
