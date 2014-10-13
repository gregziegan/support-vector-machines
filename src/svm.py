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
        self.weight_vector = None
        self.intercept = None

    @classmethod
    def solve_svm(cls, data, c):
        cross_validation = training.get_cross_validation_sets(data, NUMBER_OF_INDEPENDENT_TESTS)
        results = training.train_async(cross_validation, NUMBER_OF_INDEPENDENT_TESTS, cls, c)
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

        # set up qp parameters
        p = matrix(np.outer(class_labels, class_labels) * self.get_gramian_matrix(data))
        q = matrix(q_arr)
        g = matrix(np.vstack((np.diag(q_arr), np.identity(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.c)))
        a = matrix(class_labels, (1, num_samples))
        b = matrix(0.0)

        return solvers.qp(p, q, g, h, a, b)

    def get_gramian_matrix(self, data):
        num_samples = data.shape[0]
        gramian_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                gramian_matrix[i, j] = self.kernel(data[i], data[j])
        return gramian_matrix

    def _calculate_intercept_and_weight_vector(self, data, class_labels, qp_solution):
        self.intercept = 0
        solution_array = np.ravel(qp_solution['x'])
        support_vector_booleans = np.nonzero(solution_array)  # create array of True/False for lagrange multipliers
        sv_indices = np.arange(len(solution_array))[support_vector_booleans]
        support_vectors = data[support_vector_booleans]
        support_vector_labels = class_labels[support_vector_booleans]

        for solution_index in range(len(solution_array)):
            self.intercept += support_vector_labels[solution_index]
            gramian_matrix = self.get_gramian_matrix(data)
            self.intercept -= np.sum(solution_array * support_vector_labels *
                                     gramian_matrix[sv_indices[solution_index], support_vectors])
            self.intercept /= len(solution_array)

        self._calculate_weight_vector(solution_array, support_vector_labels, support_vectors, data.shape[1])

    def _calculate_weight_vector(self, solution_array, support_vector_labels, support_vectors, num_features):
        self.weight_vector = np.zeros(num_features)
        for i in range(len(solution_array)):
            self.weight_vector += solution_array[i] * support_vector_labels[i] * support_vectors[i]

    def classify(self, validation_data):
        return np.dot(validation_data, self.weight_vector) + self.intercept


if __name__ == '__main__':
    results = SupportVectorMachine.solve_svm(*get_svm_inputs())
    print results