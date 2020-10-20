import numpy as np

# use it instead of matmul
# because, if the sentence has tons of vocabulary, processing is gonna be busy .
# but, matmul is working for taking to one list from tons of values.
# so, let's use numpuy method to take one line from big list easier.


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = [self.params]
        self.idx = idx
        out = W[idx]

        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]

        # you can use it instead of this
        # more faster than for
        np.add.at(dW, self.idx, dout)

    # do I need it? All layers class must be managed by abstraction class
    # Do it on real app
    # return None



