from graph.ntu_rgb_d import Graph
from model.gcn import GraphIsoConvTD
import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(
    scale=2., mode="fan_out", distribution="truncated_normal")


class SpatioTemporalGraphConv(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=[3, 9],
                 stride=1,
                 activation='relu',
                 residual=True):
        super().__init__()
        self.filters = filters
        self.stride = stride
        self.activation = activation
        self.residual = residual

        self.sgcn = GraphIsoConvTD(
            [int(filters / 2), int(filters / 2)], kernel_size=kernel_size[0])

        self.tgcn = tf.keras.Sequential()
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.Activation(self.activation))
        self.tgcn.add(
            tf.keras.layers.Conv2D(self.filters,
                                   kernel_size=[kernel_size[1], 1],
                                   strides=[self.stride, 1],
                                   padding='same',
                                   kernel_initializer=INITIALIZER,
                                   kernel_regularizer=REGULARIZER,
                                   data_format='channels_first'))
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))

        self.act = tf.keras.layers.Activation(self.activation)

    def build(self, input_shape):
        if not self.residual:
            self.residual = lambda x, training=False: 0
        elif (input_shape[1] == self.filters) and (self.stride == 1):
            self.residual = lambda x, training=False: x
        else:
            self.residual = tf.keras.Sequential()
            self.residual.add(
                tf.keras.layers.Conv2D(self.filters,
                                       kernel_size=[1, 1],
                                       strides=[self.stride, 1],
                                       padding='same',
                                       kernel_initializer=INITIALIZER,
                                       kernel_regularizer=REGULARIZER,
                                       data_format='channels_first'))
            self.residual.add(tf.keras.layers.BatchNormalization(axis=1))

    def call(self, x, A, training):
        res = self.residual(x, training=training)
        x, A = self.sgcn(x, A, training=training)
        x = self.tgcn(x, training=training)
        x += res
        x = self.act(x)
        return x, A


"""Spatial temporal graph convolutional networks.
    Args:
        num_class (int): Number of classes for the classification task
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
"""


class Model(tf.keras.Model):
    def __init__(self, num_classes=60):
        super().__init__()

        graph = Graph()
        self.A = tf.Variable(graph.A[:2],
                             dtype=tf.float32,
                             trainable=False,
                             name='adjacency_matrix')

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.STGCN_layers = []
        self.STGCN_layers.append(SpatioTemporalGraphConv(64, residual=False))
        self.STGCN_layers.append(SpatioTemporalGraphConv(64))
        self.STGCN_layers.append(SpatioTemporalGraphConv(64))
        self.STGCN_layers.append(SpatioTemporalGraphConv(64))
        self.STGCN_layers.append(SpatioTemporalGraphConv(128, stride=2))
        self.STGCN_layers.append(SpatioTemporalGraphConv(128))
        self.STGCN_layers.append(SpatioTemporalGraphConv(128))
        self.STGCN_layers.append(SpatioTemporalGraphConv(256, stride=2))
        self.STGCN_layers.append(SpatioTemporalGraphConv(256))
        self.STGCN_layers.append(SpatioTemporalGraphConv(256))

        self.pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(num_classes,
                                             kernel_size=1,
                                             padding='same',
                                             kernel_initializer=INITIALIZER,
                                             kernel_regularizer=REGULARIZER,
                                             data_format='channels_first')

    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]
        M = tf.shape(x)[4]

        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])
        x = tf.reshape(x, [N * M, V * C, T])
        x = self.data_bn(x, training=training)
        x = tf.reshape(x, [N, M, V, C, T])
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        x = tf.reshape(x, [N * M, C, T, V])

        A = self.A
        for layer in self.STGCN_layers:
            x, A = layer(x, A, training=training)

        # N*M,C,T,V
        x = self.pool(x)
        x = tf.reshape(x, [N, M, -1, 1, 1])
        x = tf.reduce_mean(x, axis=1)
        x = self.logits(x)
        x = tf.reshape(x, [N, -1])

        return x
