from cvxopt import matrix
from cvxopt.solvers import qp
import pylab
from utils import get_svm_inputs

import training

DATA_DIRECTORY = '../data/'
NUMBER_OF_INDEPENDENT_TESTS = 5


class SupportVectorMachine(object):

    def __init__(self, training_set, validation_set, tradeoff):
        self.training_set = training_set
        self.validation_set = validation_set
        self.tradeoff = tradeoff

    @classmethod
    def solve_svm(cls, data, tradeoff):
        S = matrix(data.tolist())
        cross_validation = training.get_cross_validation_sets(data, NUMBER_OF_INDEPENDENT_TESTS)
        results = training.train_async(cross_validation, NUMBER_OF_INDEPENDENT_TESTS, cls, tradeoff)
        print(results)


if __name__ == '__main__':
    SupportVectorMachine.solve_svm(*get_svm_inputs())