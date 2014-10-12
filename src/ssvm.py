from svm import SupportVectorMachine
from utils import get_svm_inputs

DATA_DIRECTORY = '../data/'


class SmoothSupportVectorMachine(SupportVectorMachine):

    @classmethod
    def solve_svm(cls, data, tradeoff):
        # TODO more smoothing stuff
        pass


if __name__ == '__main__':
    SmoothSupportVectorMachine.solve_svm(*get_svm_inputs())
