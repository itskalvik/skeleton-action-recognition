from graph.ntu_rgb_d import Graph
from model.gcn import *
import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,
                                                    mode="fan_out",
                                                    distribution="truncated_normal")


class ProjectionGraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, vertices):
        super().__init__()
        self.vertices = vertices
        self.graph_conv = GraphConv(filters)

    def build(self, input_shape):
        self.centers  = self.add_weight("centers",
                                        shape=[1, int(input_shape[1]), 1, self.vertices])
        self.variance = self.add_weight("variance",
                                        shape=[1, int(input_shape[1]), 1, self.vertices])

    def call(self, x, A, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        z = (tf.reshape(x, [N, C, -1, 1])-self.centers)/tf.sigmoid(self.variance)

        q = tf.maximum(tf.reduce_sum(tf.square(z), axis=1), 1e-12)*(-1/2)
        q = tf.nn.softmax(q, axis=-1)

        z = tf.reduce_sum(tf.expand_dims(q, axis=1)*z, axis=-2)
        z /= tf.reduce_sum(q, axis=-2, keepdims=True)
        z = tf.math.l2_normalize(z, axis=-1)

        A_proj = tf.matmul(z, z, transpose_a=True)

        z, _ = self.graph_conv(z, A_proj, training=training)
        x_proj = tf.matmul(q, tf.transpose(z, perm=[0, 2, 1]))
        x_proj = tf.transpose(x_proj, perm=[0, 2, 1])
        x_proj = tf.reshape(x_proj, [N, -1, T, V])

        x += x_proj
        return x, A


"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        activation (activation function/name, optional): activation function to use
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
"""
class SpatioTemporalGraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=[3, 9], stride=1, activation='relu',
                 residual=True):
        super().__init__()
        self.filters     = filters
        self.stride      = stride
        self.activation  = activation
        self.residual    = residual

        self.sgcn = GraphConvTD(filters, kernel_size=kernel_size[0])

        self.tgcn = tf.keras.Sequential()
        self.tgcn.add(tf.keras.layers.BatchNormalization(axis=1))
        self.tgcn.add(tf.keras.layers.Activation(self.activation))
        self.tgcn.add(tf.keras.layers.Conv2D(self.filters,
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
        elif (input_shape[1]==self.filters) and (self.stride == 1):
            self.residual = lambda x, training=False: x
        else:
            self.residual = tf.keras.Sequential()
            self.residual.add(tf.keras.layers.Conv2D(self.filters,
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
        self.A = tf.Variable(graph.A,
                             dtype=tf.float32,
                             trainable=False,
                             name='adjacency_matrix')

        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.STGCN_layers = []
        self.STGCN_layers.append(SpatioTemporalGraphConv(64, residual=False))
        self.STGCN_layers.append(ProjectionGraphConv(64, 32))
        self.STGCN_layers.append(SpatioTemporalGraphConv(64))
        self.STGCN_layers.append(SpatioTemporalGraphConv(64))
        self.STGCN_layers.append(SpatioTemporalGraphConv(64))
        self.STGCN_layers.append(SpatioTemporalGraphConv(128, stride=2))
        self.STGCN_layers.append(SpatioTemporalGraphConv(128))
        self.STGCN_layers.append(SpatioTemporalGraphConv(128))
        self.STGCN_layers.append(SpatioTemporalGraphConv(256, stride=2))
        self.STGCN_layers.append(SpatioTemporalGraphConv(256))
        self.STGCN_layers.append(SpatioTemporalGraphConv(256))

        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(num_classes,
                                             kernel_size=1,
                                             padding='same',
                                             kernel_initializer=INITIALIZER,
                                             data_format='channels_first',
                                             kernel_regularizer=REGULARIZER)

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
