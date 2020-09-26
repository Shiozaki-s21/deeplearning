import numpy as np


def create_context_target(corpus, window_size):
    target = corpus[window_size: -window_size]
    contexts = []
    # loop1
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        # loop 1-1
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)

# loop1 -> target
# loop1-1 -> put each context

