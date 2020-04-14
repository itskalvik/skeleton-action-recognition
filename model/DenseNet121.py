import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, num_classes=60):
        super().__init__()
        base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                                weights=None,
                                                                input_shape=(224, 224, 1),
                                                                pooling='avg')
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(base_model.output)
        self.model = tf.keras.Model(base_model.input, x)

    def call(self, x, training):
        return self.model(x, training=training)
