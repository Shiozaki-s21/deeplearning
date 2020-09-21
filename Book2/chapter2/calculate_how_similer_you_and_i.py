from Book2.common.util import preprocess, cos_similarity, create_to_matrix

text: str = 'You say goodbye and I say hallo'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_to_matrix(corpus, vocab_size)

cS = C[word_to_id['you']]
c1 = C[word_to_id['i']]

# how similar you and i in this sentence(or generally?)
# cos similar range is from -1 to 1, so I can say you and I are similar each other
print(cos_similarity(cS, c1))
