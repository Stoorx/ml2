import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import Normalizer

import GeneticRegressor

np.random.seed(int(time.time_ns() and (2 ** 32 - 1)))


class Entry:
    def __init__(self, x: np.ndarray, y: float):
        self.x = x
        self.y = y


def smape(predict, real):
    _smape = 0.0
    for i in range(len(predict)):
        _smape += abs(predict[i] - real[i]) / ((abs(predict[i]) + abs(real[i])) / 2)

    _smape /= len(predict)
    return _smape


if __name__ == '__main__':
    trainSet = []
    testSet = []
    with open('4.txt', 'r') as file:
        feature_count = int(file.readline())
        train_size = int(file.readline())
        for i in range(train_size):
            line = list(map(lambda x: float(x), file.readline().split(' ')))
            trainSet.append(Entry(np.array(line[:-1]), line[-1]))
        test_size = int(file.readline())
        for i in range(test_size):
            line = list(map(lambda x: float(x), file.readline().split(' ')))
            testSet.append(Entry(np.array(line[:-1]), line[-1]))

    normalizer = Normalizer()
    x = normalizer.fit_transform(np.array(
        list(map(lambda o: o.x, trainSet))
    ))
    y = np.array(
        list(map(lambda o: o.y, trainSet))
    )

    test_x = normalizer.transform(np.array(
        list(map(lambda o: o.x, testSet))
    ))
    test_y = np.array(
        list(map(lambda o: o.y, testSet))
    )

    # Ridge
    ridge = Ridge(solver='svd')
    ridge.fit(x.copy(), y.copy())
    ridge_predicted = ridge.predict(test_x.copy())
    ridge_smape = smape(ridge_predicted, test_y)
    print(f"Ridge regression error: {ridge_smape :.6f}")

    # Gradient descent
    smape_list_grad = []
    iterations_list = []
    for maxit in range(50, 1030, 50):
        sgd = SGDRegressor(max_iter=maxit, n_iter_no_change=100, epsilon=0.01)
        sgd.fit(x.copy(), y.copy())
        sgd_predicted = sgd.predict(test_x.copy())
        sgd_smape = smape(sgd_predicted, test_y)
        smape_list_grad.append(sgd_smape)
        iterations_list.append(maxit)
        print(f"Gradient descent with {maxit} iterations, SMAPE: {sgd_smape :.6f}")

    plt.plot(iterations_list, smape_list_grad)

    # Genetic descent
    smape_list_gr = []
    gr_iterations_list = []
    for maxit in range(50, 1030, 50):
        gr = GeneticRegressor.GeneticRegressor(feature_count,
                                               start_population=500,
                                               iterations_count=maxit,
                                               validation_rate=0.005,
                                               mutation_rate=0.07,
                                               alive_rate=0.41,
                                               normal_population=20)
        gr.fit(x.copy(), y.copy())
        gr_predicted = gr.predict(test_x.copy())
        gr_smape = smape(gr_predicted, test_y)
        smape_list_gr.append(gr_smape)
        gr_iterations_list.append(maxit)
        print(f"Genetic descent with {maxit} iterations, SMAPE: {gr_smape :.6f}")

    plt.plot(gr_iterations_list, smape_list_gr)

    plt.show()
