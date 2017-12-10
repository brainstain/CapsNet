import tensorflow as tf
from keras import initializers, layers, backend
from keras.utils import conv_utils
import numpy as np


def get_activation(act_func):
    return squash


def squash(vectors, axis=-1):
    s_squared_norm = backend.sum(backend.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / backend.sqrt(s_squared_norm + backend.epsilon())
    return scale * vectors


class PrimaryCapsule(layers.Layer):
    """
    This applies Conv2D to channels then concatenates all capsules
    The intention is to create feeds into a secondary capsule layer

    """

    def __init__(self, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid', dilation_rate=(1, 1),
                 kernel_initializer='glorot_uniform', conv_activation=None, capsule_activation='squash',
                 **kwargs):
        super().__init__(**kwargs)
        self.dilation_rate = dilation_rate
        self.capsule_activation = get_activation(capsule_activation)
        self.kernel_initializer = kernel_initializer
        self.conv_activation = conv_activation
        self.padding = padding
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernal_size')
        self.n_channels = n_channels
        self.dim_capsule = dim_capsule


    def call(self, inputs, **kwargs):
        output = layers.Conv2D(filters=self.dim_capsule * self.n_channels, kernel_size=self.kernel_size,
                               strides=self.strides, padding=self.padding, name='primarycapsule_conv',
                               activation=self.conv_activation, kernel_initializer=self.kernel_initializer)
        output = output(inputs)
        output = layers.Reshape(target_shape=[-1, self.dim_capsule], name='primarycapsule_reshape')(output)
        output = layers.Lambda(self.capsule_activation, name='primarycapsule_squash')(output)
        return output

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        size = 0
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            if size == 0:
                size = new_dim
            else:
                size = size*new_dim
        return tuple([None, self.n_channels*size, self.dim_capsule])


class DigitCaps(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, num_routing=3, kernel_initializer='glorot_uniform', routing='squash',
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.dim_capsule = dim_capsule  # 16
        self.num_routing = num_routing  # 3
        self.num_capsule = num_capsule  # 10
        self.routing = self.routing_softmax

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        # shape = [10, ?, 16, ?]
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        super(DigitCaps, self).build(input_shape)

    def routing_softmax(self, inputs_hat):
        inputs_hat_stop = backend.stop_gradient(inputs_hat)
        # initialize logits as zeros
        b = tf.zeros(shape=[backend.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        # routing
        for i in range(self.num_routing - 1):
            c = tf.nn.softmax(b, dim=1)
            v = squash(backend.batch_dot(c, inputs_hat_stop, [2, 2]))
            b += backend.batch_dot(v, inputs_hat_stop, [2, 3])

        return squash(backend.batch_dot(tf.nn.softmax(b, dim=1), inputs_hat, [2, 2]))

    def call(self, inputs, **kwargs):
        # Converts input to 4D tensor
        inputs_expand = backend.expand_dims(inputs, 1)

        # Replicate the inputs for the number of capsules
        inputs_tiled = backend.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        # Get prediction vectors from input layer
        inputs_hat = backend.map_fn(lambda x: backend.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled,
                                    name='prediction_vector')
        return self.routing(inputs_hat)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


# convert masking to configurable
class Mask(layers.Layer):
    def __init__(self, y_true=None, **kwargs):
        super().__init__(**kwargs)
        self.mask = y_true
        self.prediction = y_true is None

    def call(self, inputs, **kwargs):
        if self.prediction:
            x = backend.sqrt(backend.sum(backend.square(inputs), -1))
            self.mask = backend.one_hot(indices=backend.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        return backend.batch_flatten(inputs * backend.expand_dims(self.mask))

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[1] * input_shape[2]])
