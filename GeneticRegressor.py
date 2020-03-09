import math

import numpy as np


class GeneticRegressor:
    def __init__(self,
                 features_count,
                 iterations_count=100,
                 start_population=40,
                 normal_population=20,
                 mutation_rate=0.5,
                 alive_rate=0.4,
                 validation_rate=0.25):
        self.features_count = features_count
        self.iterations_count = iterations_count
        self.start_population = start_population
        self.normal_population = normal_population
        self.mutation_rate = mutation_rate
        self.alive_rate = alive_rate
        self.validation_rate = validation_rate
        self.w = []
        self.population = []

    @staticmethod
    def _calculate(x: list, w: list):
        res = 0.0
        for i in range(len(x)):
            res += x[i] * w[i]
        res += w[len(w) - 1]
        return res

    def _calculateMeanError(self, X, y, w):
        validation_x = []
        validation_y = []
        validation_count = int(len(X) * self.validation_rate)
        validation_count = validation_count if validation_count >= 1 else 1
        for i in range(validation_count):
            entry_idx = int(np.random.uniform(0, len(X) - 1))
            validation_x.append(X[entry_idx])
            validation_y.append(y[entry_idx])

        sq_error = 0.0
        for i in range(validation_count):
            sq_error += (validation_y[i] - self._calculate(validation_x[i], w)) ** 2
        return math.sqrt(sq_error / validation_count)

    @staticmethod
    def _crossover(w1, w2):
        result = []
        for i in range(len(w1)):
            result.append(
                w1[i] + w2[i] / 2 if np.random.uniform() < 0.5
                else math.sqrt(abs(w1[i] * w2[i])) * (
                    1.0 if np.random.uniform() < 0.5 else -1.0
                )
            )
        return result

    def _mutate(self, w):
        result = []
        for i in w:
            result.append(i * np.random.normal(scale=1000.0) if np.random.uniform() < self.mutation_rate else i)
        return result

    def fit(self, X, y):
        for i in range(self.start_population):
            self.population.append(
                np.random.uniform(low=-1.0, high=1.0, size=self.features_count + 1)
            )
        for iteration in range(self.iterations_count):
            new_items = []
            for it in self.population:
                new_items.append(
                    self._mutate(
                        self._crossover(
                            it,
                            self.population[int(np.random.uniform(0, len(self.population) - 1))]
                        )
                    )
                )

            for i in new_items:
                self.population.append(i)

            if len(self.population) > self.normal_population:
                calculated_errors = []
                for item in self.population:
                    calculated_errors.append(
                        (item,
                         self._calculateMeanError(X, y, item))
                    )
                calculated_errors.sort(key=lambda o: o[1])

                self.population.clear()
                for i in range(int(len(calculated_errors) * self.alive_rate)):
                    self.population.append(
                        calculated_errors[i][0]
                    )

        self.w = self.population[0]

    def predict(self, X):
        predict_list = []
        for x in X:
            predict_list.append(self._calculate(x, self.w))
        return predict_list
