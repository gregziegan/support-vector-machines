from svm import SupportVectorMachine
from utils import get_svm_inputs
import numpy.linalg
import math

DATA_DIRECTORY = '../data/'


class SmoothSupportVectorMachine(SupportVectorMachine):

    def __init__(self):
        super(SupportVectorMachine, self).__init__()

    def smoothing_function(self, x, alpha):
        assert alpha > 0
        return x + ((1/alpha) * math.log(1 + math.pow(math.e, (-1 * alpha * x))))


if __name__ == '__main__':
    SmoothSupportVectorMachine.solve_svm(*get_svm_inputs())
