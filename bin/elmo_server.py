'''
ELMo server that listens for queries and sends embedding responses.

'''

from __future__ import print_function

import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import os
import pickle
import socket
import sys

from sys import stderr


# Maximum packet size in characters
MAX_PACKET_SIZE = 1000000
# Port to listen in
PORT = 8888
# Finished packet
END = '<END>'.encode()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == '__main__':
    # Location of pretrained LM.  Here we use the test fixtures.
    # datadir = os.path.join('tests', 'fixtures', 'model')
    data_dir = '/disk/scratch/mpatsis/eddie/data/phog/js/'
    vocab_file = os.path.join(data_dir, 'vocab')
    model_dir = '/disk/scratch/mpatsis/eddie/models/phog/js/elmo/1024/'
    options_file = os.path.join(model_dir, 'query_options.json')
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
    elmo_code_rep_op = weight_layers('input', code_embeddings_op, l2_coef=0.0)

    # Create a Tensorflow session
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())
        # Warm up the LSTM state, otherwise will get inconsistent embeddings.
        raw_code = [
            'STD:function STD:( ID:e STD:, ID:tags STD:) STD:{ ID:tags STD:. ID:should STD:. ' + 
            'ID:have STD:. ID:lengthOf STD:( LIT:1 STD:) STD:; ID:done STD:( STD:) STD:; STD:} '
            'STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; ' +
            'STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) STD:; STD:} STD:) ' +
            'STD:; STD:} STD:) STD:;',
            'STD:var ID:gfm STD:= ID:require STD:( LIT:github-flavored-markdown STD:) STD:;'
        ]
        tokenized_code = [sentence.split() for sentence in raw_code]
        # Create batches of data for warm up.
        code_ids = batcher.batch_sentences(tokenized_code)
        for step in range(100):
            elmo_code_representation = sess.run(
                [elmo_code_rep_op['weighted_op']],
                feed_dict={code_character_ids: code_ids}
            )
        print('ELMo was warmed up.')
        
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        server_address = ('localhost', PORT)
        eprint('starting up on %s port %s' % server_address)
        sock.bind(server_address)

        # Listen for incoming connections
        sock.listen(1)

        while True:
            # Wait for a connection
            eprint('ELMo server is up and ready! Waiting for a connection...')
            connection, client_address = sock.accept()

            try:
                eprint('Received connection from:', client_address)

                # Receive code sequences in small chunks, calculate ELMo representations and send response.
                while True:
                    received_data = ''
                    data = connection.recv(MAX_PACKET_SIZE)
                    while not data or not data[-len(END): ] == END:
                        # eprint('received "%s"' % data.decode())
                        received_data += data
                        data = connection.recv(MAX_PACKET_SIZE)
                    received_data += data[: -len(END)]
                    eprint('received "%s"' % received_data)
                    eprint('Quering elmo!')

                    # Create batches of data.
                    tokenized_code = pickle.loads(received_data)
                    code_ids = batcher.batch_sentences(tokenized_code)
                    
                    # Compute ELMo representations (here for the input only, for simplicity).
                    elmo_code_representation = sess.run(
                        [elmo_code_rep_op['weighted_op']],
                        feed_dict={code_character_ids: code_ids}
                    )
                    print('Representations:', elmo_code_representation)
                    eprint('Sending ELMo representations back to the client.')
                    
                    connection.sendall(pickle.dumps(elmo_code_representation))
                    connection.sendall(END)
                    break                
            finally:
                # Clean up the connection
                connection.close()

