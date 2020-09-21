import numpy as np
from Book2.common.util import preprocess, cos_similarity, create_to_matrix

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # get query
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('¥n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # calculate cos similar
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # output ranking
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print('%s: %s' % (id_to_word[i], similarity[i]))

        count += 1

        if count >= top:
            return

