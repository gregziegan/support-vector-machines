from cvxopt import matrix, solvers
import numpy as np
from utils import get_svm_inputs

import training

DATA_DIRECTORY = '../data/'
NUMBER_OF_INDEPENDENT_TESTS = 5


class SupportVectorMachine(object):

    def __init__(self, c):
        self.c = float(c)
        self.kernel = lambda x1, x2: np.dot(x1, x2)  # linear kernel

        # private variables to be used by train/classify
        self._weight_vector = None
        self._intercept = None
        self._gramian_matrix = None

    @classmethod
    def solve_svm(cls, data, c):
        results = training.train_async(data, NUMBER_OF_INDEPENDENT_TESTS, cls, c)
        return results

    def train(self, data, class_labels):
        """

        :param data: training data
        :type data: ndarray
        :param class_labels: class labels
        :return:
        """
        qp_solution = self.solve_quadratic_problem(data, class_labels)
        self._calculate_intercept_and_weight_vector(data, class_labels, qp_solution)

    def solve_quadratic_problem(self, data, class_labels):
        num_samples, num_features = data.shape
        q_arr = np.ones(num_samples) * -1
        self._calculate_gramian_matrix(data)

        # set up qp parameters
        p = matrix(np.outer(class_labels, class_labels) * self._gramian_matrix)
        q = matrix(q_arr)
        g = matrix(np.vstack((np.diag(q_arr), np.identity(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.c)))
        a = matrix(class_labels, (1, num_samples))
        b = matrix(0.0)

        return solvers.qp(p, q, g, h, a, b)

    def _calculate_gramian_matrix(self, data):
        num_samples = data.shape[0]
        self._gramian_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                self._gramian_matrix[i, j] = self.kernel(data[i], data[j])

    def _calculate_intercept_and_weight_vector(self, data, class_labels, qp_solution):
        self._intercept = 0

        solution_array = np.ravel(qp_solution['x'])  # a
        support_vector_booleans = solution_array > 1e-6  # sv create ndarray of True/False for lagrange multipliers
        sv_indices = np.arange(len(solution_array))[support_vector_booleans]  # ind get indices of nonzero
        solution_array = solution_array[support_vector_booleans]  # self.a
        support_vectors = data[support_vector_booleans]  # self.sv
        support_vector_labels = class_labels[support_vector_booleans]  # self.sv_y

        for solution_index in range(len(solution_array)):
            self._intercept += support_vector_labels[solution_index]
            self._intercept -= np.sum(solution_array * support_vector_labels *
                                      self._gramian_matrix[sv_indices[solution_index], support_vector_booleans])
            self._intercept /= len(solution_array)

        self._calculate_weight_vector(solution_array, support_vector_labels, support_vectors, data.shape[1])

    def _calculate_weight_vector(self, solution_array, support_vector_labels, support_vectors, num_features):
        self._weight_vector = np.zeros(num_features)
        for i in range(len(solution_array)):
            self._weight_vector += solution_array[i] * support_vector_labels[i] * support_vectors[i]

    def classify(self, validation_data):
        return np.sign(np.dot(validation_data, self._weight_vector) + self._intercept)


if __name__ == '__main__':
    results = SupportVectorMachine.solve_svm(*get_svm_inputs())
    print results