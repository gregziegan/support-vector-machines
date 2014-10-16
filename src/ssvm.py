import numpy as np
from numpy import linalg
import math

from svm import SupportVectorMachine
from utils import get_svm_inputs

DATA_DIRECTORY = '../data/'
NUMBER_OF_INDEPENDENT_TESTS = 5


class SmoothSupportVectorMachine(SupportVectorMachine):

    def __init__(self, c):
        super(SmoothSupportVectorMachine, self).__init__(c)

        # instance variables to be calculated during training:
        self._a = None
        self._d = None
        self._e = None
        self._gamma = None
        self._hessian = None
        self._num_features = None
        self._w = None
        self._z_gradient = None

    @staticmethod
    def _smoothing_function(x, alpha):
        assert alpha > 0
        return x + 1.0/alpha * np.log(1 + np.exp(-alpha*x))

    def _get_next_w(self, current_w, step, step_direction):
        return current_w + step * step_direction[0:self._num_features]

    def _get_next_gamma(self, current_gamma, step, step_direction):
        return current_gamma + step * step_direction[self._num_features, 0]

    def _objective_function(self, alpha, gamma, w):
        x = self._e - self._d * (self._a * w - gamma * self._e)
        p = self._smoothing_function(x, alpha)
        return 0.5 * (self.c * math.pow(linalg.norm(p), 2) + (np.transpose(w) * w)[0, 0] + math.pow(gamma, 2))

    def _calculate_gradient(self, gamma, w):
        temp = ((self._d * self._a) * w)
        temp2 = temp - ((self._d * self._e) * gamma)
        rv = self._e - temp2
        #rv = self._e - ((self._d * w * self._a) - (self._d * gamma * self._e))  # e - D(Aw - e * gamma0)
        self._calculate_hessian(rv)
        plus_function = (rv < 0).choose(rv, 0)  # (7) in the paper

        z_gradient = np.vstack((
            (w - self.c * np.transpose((self._d * self._a)) * plus_function),
            gamma + self.c * np.transpose((self._d * self._e)) * plus_function
        ))

        self._z_gradient = np.transpose(np.matrix(z_gradient))

    def _calculate_hessian(self, rv):
        h = 0.5 * (self._e + np.sign(rv))
        t = np.identity(h.shape[0])
        sh = np.transpose((self._d * self._a)) * t
        p = sh * (self._d * self._a)
        q = sh * (self._d * self._e)

        tmp1 = np.identity(self._w.shape[0] + 1)
        tmp2 = self.c * np.vstack((np.hstack((p, -q)), np.hstack((np.transpose(-q), np.matrix([linalg.norm(h)])))))
        self._hessian = tmp1 + tmp2

    def _get_next_armijo_step(self, w, alpha, gamma, step_direction, step_gap):
        step = 1
        objective = self._objective_function(alpha, gamma, w)

        next_w = self._get_next_w(w, step, step_direction)
        next_gamma = self._get_next_gamma(gamma, step, step_direction)
        next_objective = self._objective_function(alpha, next_gamma, next_w)
        objective_difference = objective - next_objective

        convergence = -.05 * step_gap * step

        while objective_difference < -0.05 * step * step_gap:
            step *= 0.5
            next_w = self._get_next_w(w, step, step_direction)
            next_gamma = self._get_next_gamma(gamma, step, step_direction)
            next_objective = self._objective_function(alpha, next_gamma, next_w)
            objective_difference = objective - next_objective

        return step

    def train(self, data, class_labels):

        num_examples, self._num_features = data.shape

        self._a = np.matrix(data)
        self._e = np.matrix(np.ones((num_examples, 1)))
        d_tmp = np.array(np.identity(num_examples))
#        self._d = d_tmp * class_labels
        self._d = np.matrix(d_tmp * class_labels) # np.array(np.hstack((class_labels,) * num_examples)))
        self._w = np.matrix(np.zeros((self._num_features, 1)))
        self._gamma = 0

        alpha = 5

        self._calculate_gradient(self._gamma, self._w)

        step_direction = linalg.inv(self._hessian) * -1 * np.transpose(self._z_gradient)
        step_gap = np.transpose(step_direction) * np.transpose(self._z_gradient)
        step = self._get_next_armijo_step(self._w, alpha, self._gamma, step_direction, step_gap)

        distance_to_solution = step * (self._z_gradient * np.transpose(self._z_gradient))[0, 0]
        convergence = 0.01
        while distance_to_solution >= convergence:
            self._w = self._get_next_w(self._w, step, step_direction)
            self._gamma = self._get_next_gamma(self._gamma, step, step_direction)
            self._calculate_gradient(self._gamma, self._w)
            step_direction = linalg.inv(self._hessian) * -1 * np.transpose(self._z_gradient)
            step_gap = np.transpose(step_direction) * np.transpose(self._z_gradient)
            step = self._get_next_armijo_step(self._w, alpha, self._gamma, step_direction, step_gap)
            distance_to_solution = step * (self._z_gradient * np.transpose(self._z_gradient))[0, 0]

    def classify(self, data):

        return (np.dot(np.transpose(self._w), data) - self._gamma)[0, 0]

if __name__ == '__main__':
    SmoothSupportVectorMachine.solve_svm(*get_svm_inputs())
