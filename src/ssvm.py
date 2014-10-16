from svm import SupportVectorMachine
from utils import get_svm_inputs
import numpy.linalg
import math
import training

DATA_DIRECTORY = '../data/'
NUMBER_OF_INDEPENDENT_TESTS = 5


class SmoothSupportVectorMachine(SupportVectorMachine):

    @staticmethod
    def smoothing_function(x, alpha):
        assert alpha > 0
        return x + ((1/alpha) * math.log(1 + math.pow(math.e, (-1 * alpha * x))))

    def __init__(self, c, a):
        super(SmoothSupportVectorMachine, self).__init__(c)
        w0 = numpy.array([[-977], [10000], [1]])  # starting point for newton method
        b = 0
        self.a = numpy.vstack([numpy.hstack([A_pos, -numpy.ones([A_pos.shape[0], 1])]), numpy.hstack([-A_neg, numpy.ones([A_neg.shape[0], 1])])])
        self.w = numpy.vstack((w0, b))

    def objf(self, w):
        temp = numpy.ones((self.a.shape[0], 1)) - numpy.dot(self.a, w)
        v = numpy.maximum(temp, 0)
        return 0.5 * (numpy.dot(v.transpose(), v) + numpy.dot(w.transpose(), w) / self.C)

    def train(self, data, y):
        e = numpy.ones((self.a.shape[0], 1))

        distance = 1
        convergence_value = 1e-5
        while distance > convergence_value:
            d = e - numpy.dot(self.a, self.w)

            are_all_attributes_zero = d[:, 0] > 0

            if not are_all_attributes_zero.all:
                return

            hessian = numpy.eye(self.a.shape[1]) / self.c + numpy.dot(self.a[are_all_attributes_zero, :].transpose(),
                                                                      self.a[are_all_attributes_zero, :])
            z_gradient = self.w / self.c - numpy.dot(self.a[are_all_attributes_zero, :].transpose(),
                                                     d[are_all_attributes_zero])

            del d
            del are_all_attributes_zero

            if numpy.dot(z_gradient.transpose(), z_gradient) / self.a.shape[1] > convergence_value:
                z = numpy.linalg.solve(-hessian, z_gradient)

                del hessian

                obj1 = self.objf(self.w)
                w1 = self.w + z
                obj2 = self.objf(w1)

                if (obj1 - obj2) <= self.convergSpeed:
                    gap = numpy.dot(z.transpose(), z_gradient) # Compute the gap
                    stepsize = self.armijo(self.w, z, gap, obj1)
                    self.w = self.w + stepsize * z
                else:
                    # Use the Newton method
                   self.w = w1

                distance = numpy.linalg.norm(z)

        #print self.w.shape
        return {'w': self.w[0: self.w.shape[0] - 1], 'b': self.w[self.w.shape[0] - 1]}

    def classify(self, ):

if __name__ == '__main__':
    SmoothSupportVectorMachine.solve_svm(*get_svm_inputs())
