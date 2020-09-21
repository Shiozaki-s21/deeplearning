from Book2.common.util import create_to_matrix, preprocess, cos_similarity, ppmi
import numpy as np

text = 'You say goodbye and I say hallo.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_to_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

