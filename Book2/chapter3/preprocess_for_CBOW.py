import sys
sys.path.append('..')
import numpy as np
from Book2.common.util import preprocess, create_contexts_target, convert_one_hot


# preprocess
text = 'You say goodbye and I say hallo.'

corpus, word_to_id, id_to_word = preprocess(text)

context, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(context, vocab_size)



