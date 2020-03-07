import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import Normalizer


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
    smape_list = []
    iterations_list = []
    for maxit in range(10, 1000, 10):
        sgd = SGDRegressor(max_iter=maxit, n_iter_no_change=100, epsilon=0.01)
        sgd.fit(x.copy(), y.copy())
        sgd_predicted = sgd.predict(test_x.copy())
        sgd_smape = smape(sgd_predicted, test_y)
        smape_list.append(sgd_smape)
        iterations_list.append(maxit)
        print(f"Gradient descent with {maxit} iterations, SMAPE: {sgd_smape :.6f}")

    plt.plot(iterations_list, smape_list)
    plt.show()
