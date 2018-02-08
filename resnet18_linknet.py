# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     resnet18_linknet
   version:        v1.0 
   Description :
   Author :       liuhengli
   date：          18-1-18
   license:        Apache Licence
-------------------------------------------------
   Change Activity:
                   18-1-18:
-------------------------------------------------
"""
__author__ = 'liuhengli'

from resnet18 import *
from tensorflow.contrib import slim
import tensorflow as tf


class LinkNet_resnt18():

    def __init__(self, inputs, num_classes=20, weight_decay=2e-4, is_training=False):
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.is_training = is_training
        self.block_fn = building_block
        self.inputs = inputs
        self.layers = [2, 2, 2, 2]
        self.model_name = 'resnet18-linknet'

    def convBnRelu(self, input, num_channel, kernel_size, stride, is_training, scope, padding = 'SAME'):
        x = slim.conv2d(input, num_channel, [kernel_size, kernel_size], stride=stride, activation_fn=None, scope=scope+'_conv1', padding = padding)
        x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm1')
        x = tf.nn.relu(x, name=scope+'_relu1')
        return x

    def deconvBnRelu(self, input, num_channel, kernel_size, stride, is_training, scope, padding = 'VALID'):
        x = slim.conv2d_transpose(input, num_channel, [kernel_size, kernel_size], stride=stride, activation_fn=None, scope=scope+'_fullconv1', padding = padding)
        x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm1')
        x = tf.nn.relu(x, name=scope+'_relu1')
        return x  

    def decoder_block(self, inputs, in_channels, out_channels, stride=2, scope='decoder_block'):
        net = conv2d_same(inputs, in_channels, 3, 1)
        net = conv2d_same(net, in_channels // 4, 1, 1)
        net = slim.conv2d_transpose(net, in_channels // 4, 3, stride, padding='SAME', scope=scope)
        net = conv2d_same(net, out_channels, 1, 1)
        return net

    def build_model(self):
        end_points = {}
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training):
            with tf.variable_scope(self.model_name):
                with slim.arg_scope(resnet_arg_scope(weight_decay=self.weight_decay)):
                    with tf.variable_scope('init_block'):
                        net = conv2d_same(inputs=self.inputs, filters=64, kernel_size=7, stride=2)
                        net = tf.identity(net, 'initial_conv')
                        net = slim.max_pool2d(inputs=net, kernel_size=3, stride=2, padding='SAME')
                        self.init_block = net
                        end_points['init_block'] = self.init_block

                    with tf.variable_scope('encode'):
                        with tf.variable_scope('encode_block1'):
                            net = block_layer(
                                inputs=net, filters=64, block_fn=self.block_fn, blocks=self.layers[0],
                                strides=1, name='encode_block1')
                        encode1 = net
                        end_points['encode1'] = encode1
                        with tf.variable_scope('encode_block2'):
                            net = block_layer(
                                inputs=net, filters=128, block_fn=self.block_fn, blocks=self.layers[1],
                                strides=2, name='encode_block2')
                        encode2 = net
                        end_points['encode2'] = encode2
                        with tf.variable_scope('encode_block3'):
                            net = block_layer(
                                inputs=net, filters=256, block_fn=self.block_fn, blocks=self.layers[2],
                                strides=2, name='encode_block3')
                        encode3 = net
                        end_points['encode3'] = encode3
                        with tf.variable_scope('encode_block4'):
                            net = block_layer(
                                inputs=net, filters=512, block_fn=self.block_fn, blocks=self.layers[3],
                                strides=2, name='encode_block4')
                        encode4 = net
                        end_points['encode4'] = encode4
                    with tf.variable_scope('decode'):
                        net = self.decoder_block(encode4, 512, 256, stride=2, scope='decoder_block1')
                        decode4 = net + encode3
                        end_points['decode4'] = decode4
                        net = self.decoder_block(decode4, 256, 128, stride=2, scope='decoder_block2')
                        decode3 = net + encode2
                        end_points['decode3'] = decode3
                        net = self.decoder_block(decode3, 128, 64, stride=2, scope='decoder_block3')
                        decode2 = net + encode1
                        end_points['decode2'] = decode2
                        net = self.decoder_block(decode3, 64, 64, stride=2, scope='decoder_block4')
                        decode1 = net + self.init_block
                        end_points['decode1'] = decode1
                    f1 = self.deconvBnRelu(decode1, 32, 3, stride=2, is_training=self.is_training, scope='f1', padding='SAME')
                    f2 = self.convBnRelu(f1, 32, 3, stride=1, is_training=self.is_training, padding='SAME', scope='f2')           
                # with tf.variable_scope('classifier'):
                logits = slim.conv2d_transpose(f2, self.num_classes, [2, 2], stride=2, activation_fn=None, normalizer_fn=None, scope='logits', padding='VALID')
                end_points['classifier'] = logits
                return logits, end_points


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    model = LinkNet_resnt18(inputs, is_training=True)
    result, end_points = model.build_model()
    for i in end_points.keys():
        print(i, end_points[i].shape)













