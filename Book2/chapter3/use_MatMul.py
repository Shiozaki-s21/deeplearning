import numpy as np
import sys
sys.path.append('..')

from Book2.common.layers import MatMul

# Word ID
c = np.array([1, 0, 0, 0, 0, 0, 0])
# Weight
W = np.random.randn(7, 3)
print(W)
layer = MatMul(W)
# took line vector
h = layer.forward(c)

print(h)





