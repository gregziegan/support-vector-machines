import argparse
import numpy as np
import scipy.io
import time
import os

from mldata import parse_c45

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '../data/')


def get_svm_inputs():
    parser = argparse.ArgumentParser(description="SVM Classifier")
    parser.add_argument('data_file_name')
    parser.add_argument('c', type=float)
    args = parser.parse_args()

    if args.data_file_name.endswith(".mat"):
        data_dict = scipy.io.loadmat(DATA_DIRECTORY + args.data_file_name)
        data_set_key = args.data_file_name.replace('.mat', '')
        data_set = (data_dict[data_set_key]).astype(float)
    else:
        example_set = parse_c45(args.data_file_name, DATA_DIRECTORY)
        data_set = np.array(example_set.to_float())
    return normalize(data_set), args.c


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f seconds' % (f.func_name, (time2-time1))
        return ret
    return wrap


def get_accuracy(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):
    return (num_true_positives + num_true_negatives) / (num_true_positives + num_true_negatives +
                                                        num_false_positives + num_false_negatives)


def get_precision(num_true_positives, num_false_positives):
    if num_false_positives == 0:
        return 1.0
    return num_true_positives / (num_true_positives + num_false_positives)


def get_recall(num_true_positives, num_false_negatives):
    if num_false_negatives == 0:
        return 1.0
    return num_true_positives / (num_true_positives + num_false_negatives)


def normalize(data):
    stds = data.std(axis=0)
    means = data.mean(axis=0)
    for example in range(0, len(data)):
        for i in range(1, data[example].size - 1):
            data[example][i] = (data[example][i] - means[i]) / stds[i]
    return data


def print_performance(results):
    num_true_positives, num_false_positives, num_true_negatives, num_false_negatives = 0.0, 0.0, 0.0, 0.0

    accuracies, precisions, recalls = [], [], []
    for result in results:
        predictions, class_labels = result['predictions'], result['class_labels']
        for prediction, class_label in zip(predictions, class_labels):
            if prediction > 0:
                if class_label > 0:
                    num_true_positives += 1
                else:
                    num_false_positives += 1
            else:
                if class_label <= 0:
                    num_true_negatives += 1
                else:
                    num_false_negatives += 1

            accuracies.append(get_accuracy(num_true_positives, num_false_positives,
                                           num_true_negatives, num_false_negatives))
            precisions.append(get_precision(num_true_positives, num_false_positives))
            recalls.append(get_recall(num_true_positives, num_false_negatives))

    print "Accuracy: {:0.3f} {:0.3f}".format(np.mean(accuracies), np.std(accuracies))
    print "Precision: {:0.3f} {:0.3f}".format(np.mean(precisions), np.std(precisions))
    print "Recall: {:0.3f} {:0.3f}".format(np.mean(recalls), np.std(recalls))