# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import slim
"""
-------------------------------------------------
   File Name：     resnet18
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

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def conv2d_same(inputs, filters, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.
  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, filters, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, filters, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def building_block(inputs, filters, projection_shortcut, strides):
    """Standard building block for residual networks with BN before convolutions.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block.
    """
    shortcut = inputs
    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        # print("shortcut: ", shortcut.shape)

    # print(inputs.shape)
    net = slim.conv2d(inputs=inputs, num_outputs=filters // 2, kernel_size=1, stride=1)
    net = conv2d_same(inputs=net, filters=filters // 2, kernel_size=3, stride=strides)
    # net = conv2d_same(inputs=net, filters=filters // 2, kernel_size=3, stride=1)
    net = slim.conv2d(inputs=net, num_outputs=filters, kernel_size=1, stride=1)
    # print(inputs.shape)
    return net + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, name):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters

  def projection_shortcut(inputs):
    return conv2d_same(inputs=inputs, filters=filters_out, kernel_size=1, stride=strides)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, projection_shortcut, strides)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, None, 1)

  return tf.identity(inputs, name)


def resnet_18_generator(inputs, block_fn, layers, num_classes):
    """Generator for ImageNet ResNet v2 models.
    Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
    """
    end_points = {}
    inputs = conv2d_same(inputs=inputs, filters=64, kernel_size=7, stride=2)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = slim.max_pool2d(inputs=inputs, kernel_size=3, stride=2, padding='SAME')
    inputs = tf.identity(inputs, 'initial_max_pool')
    end_point = 'resnet18_init_block'
    end_points[end_point] = inputs
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, name='block_layer1')
    end_point = 'resnet18_block_layer1'
    end_points[end_point] = inputs
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, name='block_layer2')
    end_point = 'resnet18_block_layer2'
    end_points[end_point] = inputs
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, name='block_layer3')
    end_point = 'resnet18_block_layer3'
    end_points[end_point] = inputs
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, name='block_layer4')
    end_point = 'resnet18_block_layer4'
    end_points[end_point] = inputs
    inputs = slim.avg_pool2d(inputs=inputs, kernel_size=7, stride=1, padding='VALID')
    inputs = tf.identity(inputs, 'final_avg_pool')
    end_point = 'resnet18_final_avg_pool'
    end_points[end_point] = inputs
    inputs = tf.reshape(inputs, [-1, 512 if block_fn is building_block else 2048])
    inputs = slim.fully_connected(inputs=inputs, num_outputs=num_classes, activation_fn=None)
    inputs = tf.identity(inputs, 'final_dense')
    end_point = 'resnet18_output'
    end_points[end_point] = inputs
    return inputs, end_points


def resnet_18(inputs, num_classes=21, is_training=False):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]}
  }
  params = model_params[18]
  return resnet_18_generator(inputs,
      params['block'], params['layers'], num_classes, is_training)


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_transpose],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

if __name__ == '__main__':
    sc = resnet_arg_scope()
    inputs = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
    with slim.arg_scope(sc):
        with tf.variable_scope('resnet18'):
            output, end_points = resnet_18(inputs, 20)
            # print(end_points.keys())
            for key in end_points.keys():
                print(key, '-->', end_points[key].shape)