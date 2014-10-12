import numpy as np
import threading
import Queue
from svm import SupportVectorMachine


def get_cross_validation_sets(data, number_of_tests):
    """

    :param data: data to be tested in a number of combinations
    :type data: ndarray
    :param number_of_tests
    :type number_of_tests: int
    :return:
    """

    cross_validation = []

    stop_index = len(data) / 5
    data_folds = [data[(i-1) * stop_index:i * stop_index] for i in range(1, 6)]

    shape = data.shape
    training_set_shape = ((data.shape[0] / 5), data.shape[1])

    for test_index in range(number_of_tests):
        validation_set_fold_index = number_of_tests - test_index - 1
        validation_set = data_folds[validation_set_fold_index]
        training_set = np.ndarray(training_set_shape)
        for fold_index in range(len(data_folds)):
            if fold_index == validation_set_fold_index:
                continue
            np.add(training_set, data_folds[fold_index])    # build training set array

        cross_validation.append({'training_set': training_set, 'validation_set': validation_set})

    return cross_validation


def train_async(cross_validation, number_of_tests, svm_class, tradeoff):
    """

    :param cross_validation: contains the training sets
    :type cross_validation: dict
    :param number_of_tests: how many tests to perform/threads to spawn
    :type number_of_tests: int
    :return:
    """
    queue = Queue.Queue()

    for thread_id in range(number_of_tests):
        training_set = cross_validation[thread_id]['training_set']
        validation_set = cross_validation[thread_id]['validation_set']
        svm = svm_class(training_set, validation_set, tradeoff)
        t = threading.Thread(target=train_and_validate, args=(svm, queue, thread_id))
        t.daemon = True
        t.start()

    return queue.get()


def train_and_validate(svm, queue, thread_id):
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
    train(svm, thread_id)

    result = validate(svm, queue, thread_id)
    queue.put(result)


def train(svm, thread_id):
    for example in svm.training_set:
        pass
        # TODO implement training


def validate(svm, queue, thread_id):
    result = None
    for example in svm.validation_set:
        pass
        # TODO implement classification

    return result