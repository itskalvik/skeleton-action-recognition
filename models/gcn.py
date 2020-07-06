import tensorflow as tf
import numpy as np

REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(
    scale=2., mode="fan_out", distribution="truncated_normal")
"""Graph Conv for graph data https://arxiv.org/pdf/1609.02907.pdf
    Args:
        filters (int): Number of channels produced by the convolution
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
    def __init__(self, filters, einsum='ncv,nvw->ncw'):
        super().__init__()
        self.einsum = einsum
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


"""Graph Isomorphism Conv for graph data https://arxiv.org/pdf/1810.00826.pdf
    Args:
        filters (list of ints): Number of channels produced by the mlp at each layer
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


class GraphIsoConv(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 activation='relu',
                 return_logits=False,
                 einsum='ncv,nvw->ncw'):
        super().__init__()
        self.einsum = einsum
        self.mlp = tf.keras.Sequential()
        for filter in filters[:-1]:
            self.mlp.add(
                tf.keras.layers.Conv1D(filter,
                                       kernel_size=1,
                                       kernel_initializer=INITIALIZER,
                                       kernel_regularizer=REGULARIZER,
                                       data_format='channels_first'))
            self.mlp.add(tf.keras.layers.BatchNormalization(axis=1))
            self.mlp.add(tf.keras.layers.Activation(activation))
        self.mlp.add(
            tf.keras.layers.Conv1D(filters[-1],
                                   kernel_size=1,
                                   kernel_initializer=INITIALIZER,
                                   kernel_regularizer=REGULARIZER,
                                   data_format='channels_first'))
        if not return_logits:
            self.mlp.add(tf.keras.layers.BatchNormalization(axis=1))
            self.mlp.add(tf.keras.layers.Activation(activation))

        self.epsilon = tf.Variable(initial_value=0.0,
                                   dtype=tf.float32,
                                   trainable=True,
                                   name='epsilon')

    # x - (N, C, V)
    # A - (V, V) no self connections and binary adj matrix
    def call(self, x, A, training):
        A_ = A + tf.linalg.tensor_diag(tf.ones(tf.shape(A)[-1]) + self.epsilon)
        x = tf.einsum(self.einsum, x, A_)
        x = self.mlp(x, training=training)
        return x, A


"""Graph Isomorphism Conv for graph data with temporal dimension
   https://arxiv.org/pdf/1810.00826.pdf, https://arxiv.org/pdf/1801.07455.pdf
    Args:
        filters (list of ints): Number of channels produced by the mlp at each layer
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


class GraphIsoConvTD(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 activation='relu',
                 return_logits=False,
                 einsum='nctv,kvw->nkctw'):
        super().__init__()
        assert isinstance(filters, list)
        self.kernel_size = kernel_size
        self.einsum = einsum
        self.mlps = []
        for k in range(kernel_size):
            self.mlps.append(tf.keras.Sequential())
            for filter in filters[:-1]:
                self.mlps[-1].add(
                    tf.keras.layers.Conv2D(filter,
                                           kernel_size=1,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=REGULARIZER,
                                           data_format='channels_first'))
                self.mlps[-1].add(tf.keras.layers.BatchNormalization(axis=1))
                self.mlps[-1].add(tf.keras.layers.Activation(activation))
            self.mlps[-1].add(
                tf.keras.layers.Conv2D(filters[-1],
                                       kernel_size=1,
                                       kernel_initializer=INITIALIZER,
                                       kernel_regularizer=REGULARIZER,
                                       data_format='channels_first'))
            if not return_logits:
                self.mlps[-1].add(tf.keras.layers.BatchNormalization(axis=1))
                self.mlps[-1].add(tf.keras.layers.Activation(activation))

        self.epsilon = tf.Variable(initial_value=0.0,
                                   dtype=tf.float32,
                                   trainable=True,
                                   name='epsilon')

    # x - (N, C, T, V)
    # A - (k-1, V, V) no self connections and binary adj matrix
    def call(self, x, A, training):
        self_connections = tf.ones(tf.shape(A)[-1]) + self.epsilon
        self_connections = tf.linalg.tensor_diag(self_connections)
        self_connections = tf.expand_dims(self_connections, axis=0)
        A_ = tf.concat([A, self_connections], axis=0)
        x = tf.einsum(self.einsum, x, A_)

        x = tf.unstack(x, axis=1)
        for i in range(self.kernel_size):
            x[i] = self.mlps[i](x[i], training=training)
        x = tf.reduce_sum(x, axis=0)
        return x, A


"""Graph Conv for graph data with temporal dimension
   https://arxiv.org/pdf/1801.07455.pdf
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
    def __init__(self, filters, kernel_size=3, einsum='nkctv,kvw->nctw'):
        super().__init__()
        self.kernel_size = kernel_size
        self.einsum = einsum
        self.conv = tf.keras.layers.Conv2D(filters * self.kernel_size,
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

        x = tf.reshape(x, [N, self.kernel_size, C // self.kernel_size, T, V])
        x = tf.einsum(self.einsum, x, A)
        return x, A


class AdjGraphConv(tf.keras.layers.Layer):
    def __init__(self, filters, adjacency_matrix, einsum='nkctv,kvw->nctw'):
        super().__init__()
        self.A = tf.Variable(initial_value=adjacency_matrix,
                             trainable=True,
                             name='adjacency_matrix')
        self.kernel_size = int(tf.shape(self.A)[-3])
        self.einsum = einsum

        self.conv = tf.keras.layers.Conv2D(filters * self.kernel_size,
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

        x = tf.reshape(x, [N, self.kernel_size, C // self.kernel_size, T, V])
        x = tf.einsum(self.einsum, x, self.A)
        return x
