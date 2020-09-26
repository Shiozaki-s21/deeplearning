import sys
sys.path.append('..')

import numpy as np
from Book2.common.layers import MatMul

# Sample context data
c0 = ([1, 0, 0, 0, 0, 0, 0])
c1 = ([0, 0, 1, 0, 0, 0, 0])

# init weight
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# generate layer
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# forward
h0 = in_layer0.forward(c0)
h1 = in_layer0.forward(c1)

h = (h0 + h1) * 0.5
s = out_layer.forward(h)

print(s)

