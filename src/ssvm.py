from svm import SupportVectorMachine
from utils import get_svm_inputs
import numpy.linalg

DATA_DIRECTORY = '../data/'


class SmoothSupportVectorMachine(SupportVectorMachine):

    def __init__(self):
        super(SupportVectorMachine, self).__init__()

if __name__ == '__main__':
    SmoothSupportVectorMachine.solve_svm(*get_svm_inputs())
