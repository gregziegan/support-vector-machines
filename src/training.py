import numpy as np
import threading
import Queue
from svm import SupportVectorMachine


def get_cross_validation_sets(data, number_of_tests):
    """

    :param data: data to be tested in a number of combinations
    :type data: ndarray
    :param number_of_tests: the number of training/validation set pairs to return
    :type number_of_tests: int
    :return:
    """
    for k in xrange(number_of_tests):
        training = [x for i, x in enumerate(data) if i % number_of_tests != k]
        validation = [x for i, x in enumerate(data) if i % number_of_tests == k]
        yield training, validation


def train_async(data, number_of_tests, svm_class, c):
    """

    :param cross_validation: contains the training sets
    :type cross_validation: dict
    :param number_of_tests: how many tests to perform/threads to spawn
    :type number_of_tests: int
    :param c: C value, trade off between generalization and error
    :return:
    """
    queue = Queue.Queue()

    for training_set, validation_set in get_cross_validation_sets(data, number_of_tests):
        svm = svm_class(c)
        t = threading.Thread(target=train_and_classify, args=(svm, training_set, validation_set, queue))
        t.daemon = True
        t.start()

    return queue.get()


def train_and_classify(svm, training_set, validation_set, queue):
    """
    Trains and tests a set of data and stores its result to a queue.

    :param svm: svm object to train with
    :type svm: SupportVectorMachine
    :param queue: where to put the results of the training session
    :type queue: Queue
    :param thread_id: metadata for the thread
    :type thread_id: int
    :return:
    """
    training_data = training_set[:, 1:-1]
    class_labels = training_set[:, -1]

    svm.train(training_data, class_labels)

    result = svm.classify(validation_set)
    queue.put(result)
