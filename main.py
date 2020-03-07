import numpy as np
from sklearn.linear_model import Ridge


class Entry:
    def __init__(self, x: np.ndarray, y: float):
        self.x = x
        self.y = y


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
    # Ridge
    x = np.array(
        list(map(lambda o: o.x, trainSet))
    )
    y = np.array(
        list(map(lambda o: o.y, trainSet))
    )
    ridge = Ridge(solver='svd')
    ridge.fit(x, y)

    test_x = np.array(
        list(map(lambda o: o.x, testSet))
    )
    test_y = np.array(
        list(map(lambda o: o.y, testSet))
    )

    ridge_predicted = ridge.predict(test_x)

    ridge_smape = 0.0
    for i in range(len(ridge_predicted)):
        ridge_smape += abs(ridge_predicted[i] - test_y[i]) / ((abs(ridge_predicted[i]) + abs(test_y[i])) / 2)

    ridge_smape /= len(ridge_predicted)

    print(f"Ridge regression error: {ridge_smape :.3f}")


