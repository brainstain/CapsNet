from abc import ABC, abstractmethod


class TrainableData(ABC):

    @abstractmethod
    def data_generator(self):
        ...

    @abstractmethod
    def get_next_test_batch(self):
        ...

    @abstractmethod
    def get_next_train_batch(self):
        ...

    @abstractmethod
    def get_all_test_data(self):
        ...