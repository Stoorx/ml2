import re
from os import listdir, getcwd
from os.path import isfile, join, basename

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB


class Message:
    def __init__(self, fileContent: list, fileName: str):
        self.subjectWords: list = fileContent[0].rstrip().split(' ')[1:]
        self.contentWords: list = fileContent[2].rstrip().split(' ')
        self.fileName = fileName
        self.isLegit = re.match(r'^\d+spmsg\d+\.txt', basename(fileName)) is None


if __name__ == '__main__':
    parts = []
    cwd = getcwd()
    messagesDir = join(cwd, 'messages')
    for i in range(1, 10):
        parts.append(
            list(
                map(
                    lambda fpath: Message(open(fpath, 'r').readlines(), fpath),
                    [join(join('messages', 'part' + str(i)), f) for f in listdir('messages/part' + str(i)) if
                     isfile(join('messages/part' + str(i), f))]
                )
            )
        )

    larravg = []
    for i in range(len(parts)):
        larr = []
        f1arr = []
        for lamb in range(50, 100, 1):
            trainSetX = []
            trainSetY = []
            for j in range(len(parts)):
                if i == j:
                    continue
                for oi in range(len(parts[j])):
                    trainSetX.append([*parts[j][oi].subjectWords, *parts[j][oi].contentWords])
                    trainSetY.append(1 if parts[j][oi].isLegit else 0)

            wordIndexes = {}
            curri = 0
            for o in trainSetX:
                for w in o:
                    if w in wordIndexes:
                        pass
                    else:
                        wordIndexes[w] = curri
                        curri += 1

            trainSetXFeatured = []
            for o in trainSetX:
                tmp = [0] * len(wordIndexes)
                for w in o:
                    tmp[wordIndexes[w]] = 1
                trainSetXFeatured.append(tmp)

            legitClassPrior = lamb / 100
            cl = BernoulliNB(binarize=None, class_prior=[legitClassPrior, 1.0 - legitClassPrior])
            cl.fit(trainSetXFeatured, trainSetY)

            testSetX = []
            testSetY = []
            for oi in range(len(parts[i])):
                testSetX.append([*parts[i][oi].subjectWords, *parts[i][oi].contentWords])
                testSetY.append(1 if parts[i][oi].isLegit else 0)

            testSetXFeatured = []
            for o in testSetX:
                tmp = [0] * len(wordIndexes)
                for w in o:
                    if w in wordIndexes:
                        tmp[wordIndexes[w]] = 1
                testSetXFeatured.append(tmp)

            predY = cl.predict(testSetXFeatured)

            f1 = f1_score(testSetY, predY)
            larr.append(lamb)
            f1arr.append(f1)
            print(f"{i} -> {lamb}: {f1}")
        plt.plot(larr, f1arr)
        larravg.append(f1arr)

    avg = []
    for i in range(len(larravg[0])):
        val = 0.0
        for j in range(len(larravg)):
            val += larravg[j][i]
        avg.append(val / len(larravg))

    plt.plot(larr, avg)
    plt.show()
