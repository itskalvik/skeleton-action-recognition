class TemporalSampler(tf.keras.Model):
    def __init__(self, num_hidden, top_k=200):
        super().__init__()
        self.top_k = top_k
        self.lstm_model = tf.keras.models.Sequential()
        for units in num_hidden:
            self.lstm_model.add(tf.keras.layers.LSTM(units,
                                                     return_sequences=True))
        self.lstm_model.add(tf.keras.layers.LSTM(1, return_sequences=True))

    def call(self, x, training):
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.transpose(x, [0, 2, 3, 1])
        x_lstm = tf.reshape(x, [-1, T, V*C])
        confidence_scores = self.lstm_model(x_lstm, training=training)
        confidence_scores = tf.squeeze(confidence_scores)
        values, indices = tf.math.top_k(confidence_scores,
                                        k=self.top_k,
                                        sorted=False)
        x = tf.gather(x, indices, batch_dims=1)
        x = tf.math.multiply(x, tf.reshape(values, [N, self.top_k, 1, 1]))
        x = tf.transpose(x, [0, 3, 1, 2])
        return x
