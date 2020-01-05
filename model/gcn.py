import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.,
                                                    mode="fan_out",
                                                    distribution="truncated_normal")


"""Graph Conv for graph data with temporal dimension
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph
                             convolving kernel
        einsum (string, optional): einsum string used to apply adjacency matrix
                                   after conv, used to override default behaviour
    Shape:
        - Input[0]: Input graph sequence in :(N, in_channels, T_{in}, V)
        - Input[1]: Input graph adjacency matrix in :(K, V, V)
        - Output[0]: Outpu graph sequence in :(N, out_channels, T_{out}, V)
        - Output[1]: Graph adjacency matrix for output data in :(K, V, V)
        where
            N is a batch size,
            K is the spatial kernel size
            T_{in}/T_{out} is a length of input/output sequence,
            V is the number of graph nodes.
"""
class GraphConvTD(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, einsum=None):
        super().__init__()
        self.kernel_size = kernel_size

        if einsum is not None:
            self.einsum = einsum
        else:
            self.einsum = 'nkctv,kvw->nctw'

        self.conv = tf.keras.layers.Conv2D(filters*self.kernel_size,
                                           kernel_size=1,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=REGULARIZER,
                                           data_format='channels_first')

    # N, C, T, V
    def call(self, x, A, training):
        x = self.conv(x)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C//self.kernel_size, T, V])
        x = tf.einsum(self.einsum, x, A)
        return x, A

"""Vanilla Graph Conv for graph data
    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph
                             convolving kernel
        einsum (string, optional): einsum string used to apply adjacency matrix
                                   after conv, used to override default behaviour
    Shape:
        - Input[0]: Input graph sequence in :(N, in_channels, V)
        - Input[1]: Input graph adjacency matrix in :(K, V, V)
        - Output[0]: Outpu graph sequence in :(N, out_channels, V)
        - Output[1]: Graph adjacency matrix for output data in :(K, V, V)
        where
            N is a batch size,
            K is the spatial kernel size
            V is the number of graph nodes.
"""
class GraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, einsum=None):
        super().__init__()
        if einsum is not None:
            self.einsum = einsum
        else:
            self.einsum = 'ncv,nvw->ncw'
        self.conv = tf.keras.layers.Conv1D(filters,
                                           kernel_size=1,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=REGULARIZER,
                                           data_format='channels_first')

    # N, C, V
    def call(self, x, A, training):
        x = self.conv(x)
        x = tf.einsum(self.einsum, x, A)
        return x, A


"""Graph Conv for graph data with temporal dimension and trainable adj matrix
    Args:
        filters (int)      : Number of channels produced by the convolution
        adjacency_matrix   : adjacency matrix
        kernel_size (tuple): Size of the temporal convolving kernel and graph
                             convolving kernel
        einsum (string), optional: einsum string used to apply adjacency matrix
                                   after conv, used to override default behaviour
    Shape:
        - Input[0]: Input graph sequence in :(N, in_channels, T_{in}, V)
        - Output[0]: Outpu graph sequence in :(N, out_channels, T_{out}, V)
        where
            N is a batch size,
            K is the spatial kernel size
            T_{in}/T_{out} is a length of input/output sequence,
            V is the number of graph nodes.
"""
class AdjGraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, adjacency_matrix, einsum=None):
        super().__init__()
        self.A = tf.Variable(initial_value=adjacency_matrix,
                             trainable=True,
                             name='adjacency_matrix')
        self.kernel_size = int(tf.shape(self.A)[-3])

        if einsum is not None:
            self.einsum = einsum
        else:
            self.einsum = 'nkctv,kvw->nctw'

        self.conv = tf.keras.layers.Conv2D(filters*self.kernel_size,
                                           kernel_size=1,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=REGULARIZER,
                                           data_format='channels_first')

    # N, C, T, V
    def call(self, x, training):
        x = self.conv(x)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C//self.kernel_size, T, V])
        x = tf.einsum(self.einsum, x, self.A)
        return x
