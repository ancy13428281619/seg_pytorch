import numpy as np

a = np.array([1, 0, 1])
b = np.array([1, 1, 0])

print(np.bitwise_and(a, b))
print(np.sum(np.bitwise_and(a, b)))
