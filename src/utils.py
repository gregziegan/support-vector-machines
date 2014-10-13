import argparse
import scipy.io

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
