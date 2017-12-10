import capsnet as cn
import datasets
import keras.backend as K
from tensorflow.python import debug as tf_debug

if __name__ == "__main__":
    data = datasets.MNIST(batch_size=100)

    capsnet = cn.CapsNet(input_shape=[784], n_class=10, reshape=[28, 28, 1])

    capsnet.train_model.summary()

    capsnet.train(data, 100, 55000 // 100, epochs=25)

    capsnet.test(data)
