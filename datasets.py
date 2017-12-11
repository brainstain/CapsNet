from tensorflow.examples.tutorials.mnist import input_data
import model


class MNIST(model.TrainableData):

    def __init__(self, batch_size, data_dir='/tmp/data/MNIST_data/'):
        self.batch_size = batch_size
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)

    def data_generator(self):
        while 1:
            x_batch, y_batch = self.mnist.train.next_batch(batch_size=self.batch_size)
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def get_next_test_batch(self):
        return self.mnist.test.next_batch(batch_size=self.batch_size)

    def get_next_train_batch(self):
        return self.mnist.train.next_batch(batch_size=self.batch_size)

    def get_all_test_data(self):
        return [self.mnist.test.images, self.mnist.test.labels]

    def get_data_element(self):
        return self.mnist
