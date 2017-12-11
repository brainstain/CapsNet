import capsnet as cn
import datasets
import os

if __name__ == "__main__":
    working_directory = '/tmp/models/mnistcaps/'

    batch_size = 100
    data = datasets.MNIST(batch_size=batch_size)

    capsnet = cn.CapsNet(input_shape=[784], n_class=10, reshape=[28, 28, 1],
                         save_dir=working_directory)

    capsnet.train_model.summary()
    if not os.path.exists(working_directory + 'weights'):
        os.makedirs(working_directory + 'weights/')
    else:  # if there are weights load them
        capsnet.load_weights(working_directory + 'weights/trained_model.h5')

    model = capsnet.train(data, batch_size, 55000 // batch_size, epochs=50)

    model.save_weights(working_directory + 'weights/trained_model.h5', overwrite=True)
    print('Trained model saved to \'%sweights/trained_model.h5\'' % working_directory)
    capsnet.test(data)
