import numpy as np


def cross_entropy_error(y_arr, t_arr):
    delta = 1e-7  # code on the git says 1e-4
    return - np.sum(t_arr * np.log(y_arr + delta))


# t = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

t = [1, 0]
y = [0.0078, 1.0 - 0.0078]

print(cross_entropy_error(np.array(y), np.array(t)))
