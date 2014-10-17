from cvxopt import matrix, solvers
import numpy as np
from utils import get_svm_inputs, timing, print_performance
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
    @timing
    def solve_svm(cls, data, c):
        return training.train_async(data, NUMBER_OF_INDEPENDENT_TESTS, cls, c)

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
        negative_dense_arr = np.ones(num_samples) * -1
        self._calculate_gramian_matrix(data)

        # set up qp parameters for the qp solver which takes the form:
        # (1/2) * x.T * P * x + q.T * x; G * x <= h
        p = matrix(np.outer(class_labels, class_labels) * self._gramian_matrix)
        q = matrix(negative_dense_arr)
        g = matrix(np.vstack((np.diag(negative_dense_arr), np.identity(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.c)))

        solvers.options['show_progress'] = False
        return solvers.qp(p, q, g, h)

    def _calculate_gramian_matrix(self, data):
        """

        :param data: training data to be used on the svm
        :type data: ndarray
        :return:
        """
        num_samples = data.shape[0]
        self._gramian_matrix = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                self._gramian_matrix[i, j] = self.kernel(data[i], data[j])

    def _calculate_intercept_and_weight_vector(self, data, class_labels, qp_solution):
        """

        Calculates b and w for the decision boundary vector based on the qp_solution.
        These values will be used to classify new data.

        :param data: the data the svm was trained on to converge on a solution via the qp solver.
        :type data: ndarray
        :param class_labels: the true y labels of the training data set. A flat array.
        :type class_labels: ndarray
        :param qp_solution: the result of the qp_solver
        :type qp_solution: dict
        :return:
        """
        self._intercept = 0

        solution_array = np.ravel(qp_solution['x'])
        support_vector_booleans = solution_array > 1e-6  # create ndarray of True/False for lagrange multipliers
        sv_indices = np.arange(len(solution_array))[support_vector_booleans]  # get indices of non-zero elements
        solution_array = solution_array[support_vector_booleans]
        support_vectors = data[support_vector_booleans]
        support_vector_labels = class_labels[support_vector_booleans]

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
    for result in results:
        num_correct = np.sum(result['predictions'] == result['class_labels'])
        print "{}/{} correct predictions".format(num_correct, len(result['predictions']))

    print_performance(results)
