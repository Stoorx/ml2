import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    ds = pd.read_csv('geyser.csv')
    fullX = ds.drop(['class'], axis=1)
    fullY = ds['class'].replace('N', 0).replace('P', 1)
    trainX, testX, trainY, testY = train_test_split(fullX, fullY, test_size=0.3)

    icount = []
    acc = []
    for i in range(1, 200):
        cls = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=i)
        cls.fit(trainX, trainY)

        print(f'geyser: {i}')
        icount.append(i)
        acc.append(cls.score(testX, testY))
    plt.plot(icount, acc)
    plt.title('geyser')
    plt.savefig('geyser.png')

    ds = pd.read_csv('chips.csv')
    fullX = ds.drop(['class'], axis=1)
    fullY = ds['class'].replace('N', 0).replace('P', 1)
    trainX, testX, trainY, testY = train_test_split(fullX, fullY, test_size=0.75)

    plt.clf()
    icount = []
    acc = []
    for i in range(1, 200):
        cls = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=i)
        cls.fit(trainX, trainY)

        print(f'chips: {i}')
        icount.append(i)
        acc.append(cls.score(testX, testY))
    plt.plot(icount, acc)
    plt.title('chips')
    plt.savefig('chips.png')
