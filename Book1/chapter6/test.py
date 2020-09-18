import sys
import os

sys.path.append(os.pardir)

from Book1.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print("----x_train-----")
print(type(x_train))
print("-----x_test-----")
print(x_test[0])
print("-----t_train-----")
print(t_train[0])
print("-----t_test-----")
print(t_test[0])
