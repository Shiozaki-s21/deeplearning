import sys
import os
sys.path.append(os.pardir)
import numpy as np
from Book1.dataset.mnist import load_mnist
from Book1.chapter5.two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numeric = network.numerical_gradient(x_batch, t_batch)
grad_backpop = network.gradient(x_batch, t_batch)

for key in grad_numeric.keys():
    diff = np.average(np.abs(grad_backpop[key] - grad_numeric[key]))
    print(key + ': ' + str(diff))
