'''
ELMo usage example with character inputs.

'''

import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# Location of pretrained LM.  Here we use the test fixtures.
# datadir = os.path.join('tests', 'fixtures', 'model')
data_dir = '/disk/scratch/mpatsis/eddie/data/phog/js/'
vocab_file = os.path.join(data_dir, 'vocab')
model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/1024/'
options_file = os.path.join(model_dir, 'options.json')
weight_file = os.path.join(model_dir, 'weights/weights.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
code_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

# Get ops to compute the LM embeddings.
code_embeddings_op = bilm(code_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our model includes ELMo at both the input layers
# of the task GRU, so we need 2x ELMo representations for the question
# and code at each of the input and output.
# We use the same ELMo weights for both the question and code
# at each of the input and output.
elmo_code_input = weight_layers('input', code_embeddings_op, l2_coef=0.0)
# with tf.variable_scope('', reuse=True):
#     # the reuse=True scope reuses weights from the code for the question
#     elmo_question_input = weight_layers(
#         'input', question_embeddings_op, l2_coef=0.0
#     )


# Now we can compute embeddings.
raw_code = [
    'STD:function STD:( ID:e STD:, ID:tags STD:) STD:{ ID:tags STD:. ID:should STD:. ' + 
    'ID:have STD:. ID:lengthOf STD:( LIT:1 STD:) STD:; ID:done STD:( STD:) STD:; STD:} '
    'STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; ' +
    'STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) ' +
    'STD:; STD:} STD:) STD:;',
    'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;'
]
tokenized_code = [sentence.split() for sentence in raw_code]
# tokenized_question = [
#     ['What', 'are', 'biLMs', 'useful', 'for', '?'],
# ]

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    code_ids = batcher.batch_sentences(tokenized_code)
    # question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_code_input_ = sess.run(
        [elmo_code_input['weighted_op']],
        feed_dict={code_character_ids: code_ids}
    )
    print(elmo_code_input_)
    # elmo_question_input_ = sess.run(
    #     [elmo_question_input['weighted_op']],
    #     feed_dict={question_character_ids: question_ids}
    # )

