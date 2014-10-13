import threading
from Queue import Queue
from svm import SupportVectorMachine
import numpy as np
from numpy import ndarray
import logging
logging.basicConfig(filename='training.log', level=logging.DEBUG)


def get_cross_validation_sets(data, number_of_tests):
    """

    :param data: data to be tested in a number of combinations
    :type data: ndarray
    :param number_of_tests: the number of training/validation set pairs to return
    :type number_of_tests: int
    :return:
    """
    for k in xrange(number_of_tests):
        training = np.array([x for i, x in enumerate(data) if i % number_of_tests != k], np.float64)
        validation = np.array([x for i, x in enumerate(data) if i % number_of_tests == k], np.float64)
        yield training, validation


def train_async(data, number_of_tests, svm_class, c):
    """
    :param data: data set to learn
    :type data: ndarray
    :param number_of_tests: how many tests to perform/threads to spawn
    :type number_of_tests: int
    :param c: C value, trade off between generalization and error
    :return:
    """
    q = Queue()

    logging.debug("Spawning threads...")
    threads = []
    thread_id = 0
    for training_set, validation_set in get_cross_validation_sets(data, number_of_tests):
        svm = svm_class(c)
        t = threading.Thread(target=train_and_classify, args=(svm, training_set, validation_set, q, thread_id))
        t.daemon = True
        t.start()
        threads.append(t)

        thread_id += 1

    for t in threads:
        t.join()

    logging.debug("Threads finished executing.")

    results = []
    while not q.empty():
        results.append(q.get())
    return results


def get_usable_data_and_class_labels(data):
    data_column_indices = [i for i in range(1, len(data[0]) - 1)]
    usable_data = data[:, data_column_indices]
    class_labels = data[:, len(data[0]) - 1]
    class_labels = np.array([y if y != 0 else -1 for y in class_labels])
    return usable_data, class_labels


def train_and_classify(svm, training_set, validation_set, q, thread_id):
    """
    Trains and tests a set of data and stores its result to a queue.

    :param svm: svm object to train with
    :type svm: SupportVectorMachine
    :param q: where to put the results of the training session
    :type q: Queue
    :return:
    """

    training_data, class_labels = get_usable_data_and_class_labels(training_set)
    svm.train(training_data, class_labels)

    validation_data, class_labels = get_usable_data_and_class_labels(validation_set)
    predictions = svm.classify(validation_data)
    num_correct = np.sum(predictions == class_labels)
    print "{}/{} correct predictions".format(num_correct, len(predictions))
    q.put(predictions)
    q.task_done()
