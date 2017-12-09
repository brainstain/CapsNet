import tensorflow as tf
from keras import initializers, layers

class PrimaryCapsule(layers.Layer):
    """
    This applies Conv2D to channels then concatenates all capsules
    The intention is to create feeds into a secondary capsule layer

    """
    def __init__(self, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid'):
        self.padding = padding
        self.strides = strides
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dim_capsule = dim_capsule
