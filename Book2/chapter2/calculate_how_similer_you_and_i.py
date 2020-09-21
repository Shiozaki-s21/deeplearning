# import sys
# from Book2.chapter2 import preprocess, create_co_matrix, cos_similarity
# sys.path.append('')
from Book2.chapter2.preprocess import preprocess

text: str = 'You say goodby and I say hallo'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
