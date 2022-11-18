# Первая задача

import numpy as np
from numpy.linalg import matrix_power as mp
import matplotlib.pyplot as plt

n = 8
a1 = 1 / n
a2 = round(np.random.rand() * 0.5, 2)
a3 = 1 - a1 - a2

# Матрица переходных вероятностей
transferProbabilities = np.array([
  [0.5, 0.25, 0.25],
  [0.5 - 1 / (10 + n), 0, 0.5 + 1 / (10 + n)],
  [0.25, 0.25, 0.5]
])
pi_transposed = transferProbabilities.T

transferProbabilityNormalized = np.array([
  [pi_transposed[0][0] - 1, pi_transposed[0][1], pi_transposed[0][2]],
  [pi_transposed[1][0], pi_transposed[1][1] - 1, pi_transposed[1][2]],
  [1, 1, 1]
])

probabilitiesLimit = np.linalg.solve(transferProbabilityNormalized, np.array([0, 0, 1]))
pStart = np.array([a1, a2, a3])
precision = 0.0001
n = 0
#P(n) = (A^T)^n x p(0) - Вероятность событий на n-ом временном шаге.
while True:
    n += 1
    pi_transposed_npow = mp(pi_transposed, n)
    iterationP = np.matmul(pi_transposed_npow, pStart)
    if max(abs(np.subtract(iterationP, probabilitiesLimit))) < precision:
        break

print("Iterations:", n)
print("Start Probabilities =", pStart)
print("Iteration Probability Vector =", iterationP)
print("Theoretical Probability =", probabilitiesLimit)

# Вторая задача
P = [ [0.4, 0.3, 0],
      [0.5, 0.6, 0.75],
      [0.1, 0.1, 0.25] ]

birthCoef = [51, 39, 17]
deathCoef = [24, 17, 7]
# Численность (млн. человек)
startPopulation = 658.9

a = [0, 0 ,0]
for s in range(3):
    a[s] = (1000 + birthCoef[s] - deathCoef[s]) / 1000


sumTemp = sum(birthCoef)
startOrderMoment = [x / sumTemp for x in birthCoef]
prevMoment = startOrderMoment
nextMoment = startOrderMoment

print("M_0 =", startOrderMoment)
print("a =", a)

moments = [sum(startOrderMoment) * startPopulation]
# прогноз на 100 лет
years = 100
for k in range(years):
    prevMoment = nextMoment.copy()
    for j in range(3):
        momentSum = 0
        for s in range(3):
            momentSum += P[j][s] * a[s] * prevMoment[s]
        nextMoment[j] = momentSum
    moments.append(sum(nextMoment) * startPopulation)

plt.plot(moments)
plt.show()