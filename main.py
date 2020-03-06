import numpy as np

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





