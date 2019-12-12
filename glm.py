#!/usr/local/bin/python3

import numpy as np
import copy
import random
from matplotlib import pyplot as plt
import time


# Reads data from given filename
def read_data(filename):
    return np.genfromtxt(filename, delimiter=',', dtype='float32')


# Dataset class, holds name, data and labels
class Dataset:
    def __init__(self, name):
        self.name = name
        self.data = read_data(self.name + '.csv')
        self.data = np.hstack((np.ones((self.data.shape[0], 1)), self.data))
        self.labels = read_data('labels-' + self.name + '.csv')


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# GLM class, implements a Logistic, Poisson and Ordinal regression
class GLM:
    def __init__(self, type, alpha, dataset):
        self.dataset = dataset
        self.type = type
        self.data = self.dataset.data
        self.labels = np.reshape(self.dataset.labels, (-1, 1))
        self.phi = self.data
        self.t = self.labels
        self.K = 5
        self.s = 1
        self.levels = (-float('inf'), -2, -1, 0, 1, float('inf'))
        self.alpha = alpha
        self.t = np.reshape(self.labels, (self.labels.shape[0], 1))
        self.d = None
        # Set R and w to zeros
        self.R = np.zeros((self.phi.shape[0], self.phi.shape[0]), dtype='float32')
        self.w = np.zeros((self.phi.shape[1], 1), dtype='float32')
        self.y = sigmoid(self.phi @ self.w)
        self.evaluation_function = None
        self.prediction_function = None
        self.gradient = None
        self.previous_w = None
        self.t_hat = None
        self.y_hat = None
        self.hessian = None
        self.w_map = None
        self.count = 0
        self.avg_conv = -1
        self.all_error = None

        # Set-up update and prediction functions
        if self.type == "logistic":
            self.update = self.logistic_updates
            self.prediction_function = self.logistic_prediction

        elif self.type == "poisson":
            self.update = self.poisson_updates
            self.prediction_function = self.poisson_prediction

        else:  # self.type == "ordinal"
            self.update = self.ordinal_updates
            self.prediction_function = self.ordinal_prediction

    def logistic_updates(self):
        self.y = sigmoid(self.phi @ self.w)
        self.d = self.t - self.y
        np.fill_diagonal(self.R, (self.y * (1 - self.y)).T)

    def logistic_prediction(self, phi, w, t):
        y_hat = sigmoid(phi @ w)
        t_hat = np.where(y_hat >= 0.5, 1, 0)
        return np.sum(t_hat == t) / t.shape[0]

    def poisson_updates(self):
        self.y = np.exp(self.phi @ self.w)
        self.d = self.t - self.y
        np.fill_diagonal(self.R, self.y)

    def poisson_prediction(self, phi, w, t):
        y_hat = np.exp(phi @ w)
        t_hat = np.floor(y_hat)
        return np.sum(np.abs(t_hat - t)) / t.shape[0]

    def ordinal_updates(self):
        a = self.phi @ self.w
        self.y = np.asarray([sigmoid(self.s * (level - a)) for level in self.levels]).T
        [self.y] = self.y
        self.d = np.asarray(
            [self.y[i, int(self.t[i])] + self.y[i, int(self.t[i] - 1)] - 1 for i in range(self.phi.shape[0])]
        )
        self.d = np.reshape(self.d, (self.phi.shape[0], 1))
        r = np.array([self.s ** 2 * (self.y[i, int(self.t[i])] * (1 - self.y[i, int(self.t[i])]) +
                                     self.y[i, int(self.t[i] - 1)] * (1 - self.y[i, int(self.t[i] - 1)])
                                     ) for i in range(self.phi.shape[0])])
        r = np.reshape(r, (self.phi.shape[0], 1))
        np.fill_diagonal(self.R, r)

    def ordinal_prediction(self, phi, w, t):
        a = phi @ w
        y_hat = np.asarray([sigmoid(self.s * (level - a)) for level in self.levels]).T
        [y_hat] = y_hat
        p = np.array([y_hat[:, i] - y_hat[:, i - 1] for i in range(1, y_hat.shape[1])]).T
        t_hat = np.argmax(p, axis=1) + 1
        t_hat = np.reshape(t_hat, (phi.shape[0], 1))
        return np.sum(np.abs(t_hat - t)) / t.shape[0]

    # Trains GLM on selected regression
    def train(self):

        self.count = 0
        self.w = np.zeros((self.phi.shape[1], 1), dtype='float32')
        while True:
            self.count += 1
            self.update()
            self.hessian = - self.phi.T @ self.R @ self.phi - self.alpha * np.identity(self.phi.shape[1])
            self.gradient = self.phi.T @ self.d - self.alpha * self.w
            self.previous_w = copy.deepcopy(self.w)
            self.w -= np.linalg.inv(self.hessian) @ self.gradient
            num = np.linalg.norm(self.w - self.previous_w)
            den = np.linalg.norm(self.previous_w) + 1e-6
            change = num / den
            if change < 1e-3 or self.count >= 100:
                self.w_map = self.w
                return self.w_map

    # Evaluates GLM based on selected regression and 1/3 data as test set
    def evaluate(self):
        data = np.hstack((self.data, self.labels))
        all_error = []
        count = 0
        ratio = 2 / 3
        n_train_samples = int(ratio * self.labels.shape[0])
        for experiment in range(30):
            random.shuffle(data)
            train_data = data[:n_train_samples]
            test_data = data[n_train_samples:]
            single_error = []
            phi_test = test_data[:, :-1]
            t_test = np.reshape(test_data[:, -1], (-1, 1))
            for n in range(1, 11):
                n_train = int(n * 0.1 * train_data.shape[0])
                self.phi = data[:n_train, :-1]
                self.t = np.reshape(data[:n_train, -1], (-1, 1))
                self.R = np.zeros((self.phi.shape[0], self.phi.shape[0]), dtype='float32')
                w_map = self.train()
                count += self.count
                single_error.append(self.prediction_function(phi_test, w_map, t_test))
            all_error.append(single_error)
        all_error = np.asarray(all_error)
        self.all_error = all_error
        avg_error = np.mean(all_error, axis=0)
        std_error = np.std(all_error, axis=0)
        std_error /= len(all_error)**0.5
        # self.err_plot(avg_error, std_error)
        self.avg_conv = str(count / 300)

    # Plotting function
    def err_plot(self, avg_error, std_error):
        plt.clf()
        plt.xlabel('Portion of training data used from the ' + self.dataset.name + ' dataset')
        plt.ylabel('Mean Error')
        plt.title(self.type.capitalize() + ' Regression')
        plt.xticks([0.1 * x for x in range(1, 11)])
        plt.errorbar([x * 0.1 for x in range(1, 11)],
                     avg_error,
                     std_error,
                     label=self.type,
                     capsize=10,
                     elinewidth=1,
                     ecolor='red'
                     )
        plt.savefig(self.type + '_' + self.dataset.name + '.png')


def run_experiment(model, dataset, alpha):
    D = Dataset(dataset)
    model = GLM(model, alpha, D)
    start = time.time()
    model.evaluate()
    end = time.time()
    print(model.type.capitalize() + ' Regression on ' + dataset + ' dataset took ' + str(
        end - start) + ' seconds and converged in about ' + model.avg_conv + ' iterations')


# def main():
#     alpha = 10
#     run_experiment('logistic', 'A', alpha)
#     run_experiment('logistic', 'usps', alpha)
#     run_experiment('logistic', 'irlstest', alpha)
#     run_experiment('poisson', 'AP', alpha)
#     run_experiment('ordinal', 'AO', alpha)
#
#
# if __name__ == '__main__':
#     main()
