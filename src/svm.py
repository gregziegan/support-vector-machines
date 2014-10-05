from cvxopt import matrix
from cvxopt.solvers import qp
import argparse
import pylab
import scipy.io

DATA_DIRECTORY = '../data/'

def solve_svm(data):
    S = matrix(data.tolist())
    print(S)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute SVM Solver")
    parser.add_argument('data_file_name')
    args = parser.parse_args()

    data = scipy.io.loadmat(DATA_DIRECTORY + args.data_file_name)
    data_key = args.data_file_name.replace('.mat', '')
    solve_svm(data[data_key])