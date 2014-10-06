import numpy as np
import threading
import Queue


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
    data_folds = [data[(i-1) * stop_index:i * stop_index] for i in range(1, 5)]

    for test_index in range(number_of_tests):
        validation_set_fold_index = number_of_tests - test_index - 1
        validation_set = data_folds[validation_set_fold_index]
        training_set = np.array([])
        for fold_index in range(len(data_folds)):
            if fold_index == validation_set_fold_index:
                continue
            np.add(training_set, data_folds[fold_index])    # build training set array

        cross_validation.append({'training_set': np.ndarray(training_set), 'validation_set': validation_set})

    return cross_validation


def train_async(cross_validation, number_of_tests):
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
        t = threading.Thread(target=train_and_validate, args=(training_set, validation_set, queue, thread_id))
        t.daemon = True
        t.start()

    return queue.get()


def train_and_validate(training_set, validation_set, queue, thread_id):
    """
    Trains and tests a set of data and stores its result to a queue.

    :param training_set: data to be given to the learning algorithm
    :type training_set: ndarray
    :param validation_set: data to be classified by the learning algorithm
    :param queue: where to put the results of the training session
    :type queue: Queue
    :param thread_id: metadata for the thread
    :type thread_id: int
    :return:
    """
    train(training_set, thread_id)

    result = validate(validation_set, queue, thread_id)
    queue.put(result)


def train(training_set, thread_id):
    for example in training_set:
        pass
        # TODO implement training


def validate(validation_set, queue, thread_id):
    result = None
    for example in validation_set:
        pass
        # TODO implement classification

    return result