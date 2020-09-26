import sys
sys.path.append('..')

from Book2.common.trainer import Trainer
from Book2.common.optimizer import Adam
from Book2.common.util import preprocess, create_contexts_target, convert_one_hot
from Book2.chapter3.SimpleCBOW import SimpleCBOW


# variable
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

# preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# train
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# show all weight
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
