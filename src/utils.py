import argparse
import scipy.io
import time

DATA_DIRECTORY = '../data/'


def get_svm_inputs():
    parser = argparse.ArgumentParser(description="SVM Classifier")
    parser.add_argument('data_file_name')
    parser.add_argument('c', type=float)
    args = parser.parse_args()

    data_dict = scipy.io.loadmat(DATA_DIRECTORY + args.data_file_name)
    data_set_key = args.data_file_name.replace('.mat', '')
    data_set = data_dict[data_set_key]
    return data_set, args.c


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f seconds' % (f.func_name, (time2-time1))
        return ret
    return wrap
