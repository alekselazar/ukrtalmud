from cmath import sqrt
import tensorflow as tf
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Layer, Dense

class ScaledDotAttention(Layer):
    def __init__(self, **kwargs):
        super(ScaledDotAttention, self).__init__(**kwargs)

    def call(q, k, v, d_k, mask=None):
        scores = tf.matmul(q, k, transpose_b=True) / sqrt(d_k)
        if mask is not None:
            scores += -1e9 * mask
        weights = softmax(scores)
        return tf.matmul(weights, v)

class MultiHeadAttention(Layer):
    def __init__(self, num_h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_h = num_h
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = Dense(d_k)
        self.W_k = Dense(d_k)
        self.W_v = Dense(d_v)
        self.W_o = Dense(d_model)
        self.att = ScaledDotAttention()

    def reshape_tensor(self, x, h, flag):
        if flag:
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], h, -1))
            x = tf.transpose(x, perm=(0,2,1,3))
        else:
            x = tf.transpose(x, perm=(0,2,1,3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], -1))
        return x
    
    def call(self, queries, keys, values, mask=None):
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.num_h, True)
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.num_h, True)
        v_reshaped = self.reshape_tensor(self.W_v(values), self.num_h, True)
        o_reshaped = self.att(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        return self.W_o(output)
