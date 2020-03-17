import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    # with open('geyser.csv', 'r') as file:
    with open('chips.csv', 'r') as file:
        lines = file.readlines()

    dataset = list(
        map(
            lambda o: list(
                map(
                    lambda v: float(v),
                    o
                )
            ),
            list(
                map(
                    lambda o: [*o[:-1], '1.0' if o[-1] == 'P' else '-1.0'],
                    list(
                        map(
                            lambda l: l[:-1].split(','),
                            lines[1:]
                        )
                    )
                )
            )
        )
    )

    x = np.array(
        list(
            map(
                lambda o: o[:-1],
                dataset
            )
        )
    )

    y = np.array(
        list(
            map(
                lambda o: o[-1],
                dataset
            )
        )
    )

    xp = list(
        map(
            lambda o: o[:-1],
            list(
                filter(
                    lambda o: o[-1] == 1.0,
                    dataset
                )
            )
        )
    )

    xn = list(
        map(
            lambda o: o[:-1],
            list(
                filter(
                    lambda o: o[-1] == -1.0,
                    dataset
                )
            )
        )
    )

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    svc = SVC(kernel='rbf', C=1000)
    # svc = SVC(kernel='sigmoid', C=1000)
    svc.fit(x, y)
    score = svc.score(x, y)

    ax = 1.2
    heatmap = []
    # for i in range(0, 700, 1):
    for i in range(-100, 100, 1):
        heatmap.append(
            svc.decision_function(
                # [[j / 100, i / 100] for j in range(0, 2500, 1)]
                [[ax * j / 100, ax * i / 100] for j in range(-100, 100, 1)]
            )
        )

    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):
            heatmap[i][j] = (heatmap[i][j] ** 0.9)

    # plt.axis([0, 25, 0, 7])
    plt.axis([-ax, ax, -ax, ax])
    # plt.imshow(heatmap, extent=(0, 25.0, 7, 0))
    plt.imshow(heatmap, extent=(-ax, ax, ax, -ax))
    plt.scatter(
        list(map(lambda e: e[0], xp)),
        list(map(lambda e: e[1], xp)),
        c='red'
    )

    plt.scatter(
        list(map(lambda e: e[0], xn)),
        list(map(lambda e: e[1], xn)),
        c='blue'
    )
    plt.title(f"{svc.kernel}, C: {svc.C}, acc: {score:.6f}")
    plt.show()
