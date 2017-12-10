import numpy as np
from keras import layers, models, optimizers, backend, callbacks
from PIL import Image
import capslayer as cl
import model
import matplotlib.pyplot as plt

from utils import combine_images


class CapsNet:
    def __init__(self, input_shape, n_class, routings=3, kernal_size=9, conv_filters=256,
                 dim_primarycaps=8, n_channels=32, digit_caps_dim=16, save_dir='/tmp/models/',
                 learning_rate=.001, lam_recon=0.392, reshape=None):
        """

        :type reshape: [-1, rows, columns, channels]
        """
        self.lam_recon = lam_recon
        self.lr = learning_rate
        self.save_dir = save_dir

        x = layers.Input(shape=input_shape)

        if reshape is not None:
            reshape = layers.Reshape(reshape)(x)
            conv1 = layers.Conv2D(filters=conv_filters, kernel_size=kernal_size, strides=1, padding='valid',
                                  activation='relu', name='conv1')(reshape)
        else:
            # ReLU Conv1 - converts pixels to activities of local feature detectors
            conv1 = layers.Conv2D(filters=conv_filters, kernel_size=kernal_size, strides=1, padding='valid',
                                  activation='relu', name='conv1')(x)

        # Primary Caps Layer
        primarycaps = cl.PrimaryCapsule(dim_capsule=dim_primarycaps,
                                        n_channels=n_channels,
                                        kernel_size=kernal_size)(conv1)

        # Digit Caps Layer
        digitcaps = cl.DigitCaps(num_capsule=n_class,
                                 dim_capsule=digit_caps_dim,
                                 num_routing=routings)(primarycaps)

        # replace each capsule with it's length
        out_caps = layers.Lambda(lambda x1: backend.sqrt(backend.sum(backend.square(x1), -1)))(digitcaps)

        # Decoder Network
        y = layers.Input(shape=(n_class,))

        masked_by_y = cl.Mask(y_true=y)(digitcaps)
        masked = cl.Mask()(digitcaps)

        # Shared Decoder model in training and prediction
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
        decoder.add(layers.Dense(1024, activation='relu'))
        decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

        # Finish Models
        self.train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
        self.eval_model = models.Model(x, decoder(masked))

    @staticmethod
    def margin_loss(y_true, y_pred):
        """
        Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
        """
        L = y_true * backend.square(backend.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * backend.square(backend.maximum(0., y_pred - 0.1))

        return backend.mean(backend.sum(L, 1))

    def train(self, data, batch_size, steps_per_epoch, epochs=1):
        assert issubclass(type(data), model.TrainableData)

        log = callbacks.CSVLogger(self.save_dir + 'log.csv')
        tb = callbacks.TensorBoard(log_dir=self.save_dir + 'tensorboard-logs',
                                   batch_size=batch_size)
        checkpoint = callbacks.ModelCheckpoint(self.save_dir + '/weights-{epoch:02d}.h5',
                                               monitor='val_capsnet_acc',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: self.lr * (0.9 ** epoch))

        self.train_model.compile(optimizer=optimizers.Adam(lr=self.lr),
                                 loss=[self.margin_loss, 'mse'],
                                 loss_weights=[1., self.lam_recon],
                                 metrics={'capsnet': 'accuracy'})

        self.train_model.fit_generator(generator=data.data_generator(),
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       callbacks=[log, tb, checkpoint, lr_decay])

        self.train_model.save_weights(self.save_dir + 'weights/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % self.save_dir)

        return self.train_model

    def test(self, data):
        assert issubclass(data, model.TrainableData)

        x_test, y_test = data.get_all_test_data()
        y_pred, x_recon = model.predict(x_test, batch_size=100)

        print('-' * 30 + 'Begin: test' + '-' * 30)
        print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

        img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(self.save_dir + "/real_and_recon.png")
        print()
        print('Reconstructed images are saved to %s/real_and_recon.png' % self.save_dir)
        print('-' * 30 + 'End: test' + '-' * 30)
        plt.imshow(plt.imread(self.save_dir + "/real_and_recon.png"))
        plt.show()
